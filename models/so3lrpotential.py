from __future__ import annotations

from functools import partial
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import openmm
import openmm.app as app
import openmmjax
from jax_md import space
from openmm import unit
from openmmjax_export import (
    configure_pjrt_plugin,
    export_jax_model,
)
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .so3lr import SO3LR_MODEL_NAMES, get_sparse_neighbors, load_model

jax.config.update("jax_default_matmul_precision", "highest")


class SO3LRPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, charge: float = 0.0, **args):
        return SO3LRPotentialImpl(name, modelPath=modelPath, charge=charge)


class SO3LRPotentialImpl(MLPotentialImpl):
    def __init__(self, name, modelPath=None, charge: float = 0.0):
        if modelPath is not None:
            raise ValueError("SO3LR only supports the bundled checkpoint.")
        self.name = name
        self.charge = charge

    def addForces(
        self,
        topology: app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        modelPath: Optional[str] = None,
        charge: Optional[float] = None,
        total_charge: Optional[float] = None,
        neighbor_cell_atom_threshold: Optional[int] = None,
        neighbor_cell_capacity_multiplier: Optional[float] = None,
        periodic_neighborlist: bool = True,
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        **args,
    ):
        included_atoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            included_atoms = [included_atoms[i] for i in atoms]
        species = jnp.asarray(
            [atom.element.atomic_number for atom in included_atoms],
            dtype=jnp.int32,
        )
        indices = None if atoms is None else jnp.asarray(atoms, dtype=jnp.int32)
        num_system_atoms = system.getNumParticles() or topology.getNumAtoms()

        if modelPath is not None:
            raise ValueError("SO3LR only supports the bundled checkpoint.")
        if self.name not in SO3LR_MODEL_NAMES:
            raise ValueError(f"Unsupported SO3LR model: {self.name}")
        model = load_model(
            self.name,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
            neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
        )

        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        force_periodic = periodic and periodic_neighborlist
        allocation_box = None
        if force_periodic:
            box_vectors = topology.getPeriodicBoxVectors()
            if box_vectors is None:
                box_vectors = system.getDefaultPeriodicBoxVectors()
            allocation_box = jnp.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                dtype=jnp.float32,
            )
        if preprocessing_positions is None:
            raise ValueError("SO3LR JAX requires preprocessing_positions.")
        if hasattr(preprocessing_positions, "value_in_unit"):
            allocation_positions = preprocessing_positions.value_in_unit(unit.angstrom)
        else:
            scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
            allocation_positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
        allocation_positions = jnp.asarray(allocation_positions, dtype=jnp.float32)
        if atoms is not None:
            allocation_positions = allocation_positions[jnp.asarray(atoms, dtype=jnp.int32)]

        def _allocate_neighbor_lists(box_vectors_angstrom, positions_angstrom):
            if force_periodic:
                positions_angstrom = fractional_coordinates(
                    positions_angstrom,
                    box_vectors_angstrom,
                )
            neighbors = get_sparse_neighbors(
                positions_angstrom,
                box_vectors_angstrom,
                cutoff=float(model.cutoff),
                cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                periodic=force_periodic,
            )
            neighbors_lr = get_sparse_neighbors(
                positions_angstrom,
                box_vectors_angstrom,
                cutoff=float(model.long_range_cutoff),
                cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
                cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                periodic=force_periodic,
            )
            return neighbors, neighbors_lr

        neighbors, neighbors_lr = _allocate_neighbor_lists(
            allocation_box,
            allocation_positions,
        )

        model_charge = self.charge if charge is None else charge
        if total_charge is not None:
            model_charge = total_charge

        energy_fn = partial(
            _energySO3LR,
            model=model,
            species=species,
            total_charge=jnp.asarray(model_charge, dtype=jnp.float32),
            pbc=force_periodic,
            neighbors=neighbors,
            neighbors_lr=neighbors_lr,
        )

        def _energy_kjmol(positions_nm, box_vectors_nm=None):
            selected = positions_nm if indices is None else positions_nm[indices]
            return energy_fn((selected, box_vectors_nm))

        def _energy_and_forces_kjmol(positions_nm, box_vectors_nm=None):
            energy, minus_forces = jax.value_and_grad(_energy_kjmol)(
                positions_nm,
                box_vectors_nm,
            )
            return energy, -minus_forces

        def _forces_kjmol(positions_nm, box_vectors_nm=None):
            return _energy_and_forces_kjmol(positions_nm, box_vectors_nm)[1]

        force_mlir, energy_mlir, energy_and_forces_mlir, compile_options_base64 = export_jax_model(
            num_system_atoms=num_system_atoms,
            force_function=_forces_kjmol,
            energy_function=_energy_kjmol,
            energy_and_forces_function=_energy_and_forces_kjmol,
            periodic=force_periodic,
        )
        force = openmmjax.JaxForce(
            force_mlir,
            energy_mlir,
            energy_and_forces_mlir,
            compile_options_base64,
        )
        configure_pjrt_plugin(force)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(force_periodic)
        force.setOutputsForces(True)
        force.addToSystem(system)


for model_name in SO3LR_MODEL_NAMES:
    MLPotential.registerImplFactory(model_name, SO3LRPotentialImplFactory())


def fractional_coordinates(positions, box_vectors):
    jax_box = jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2)
    return space.transform(_restricted_box_inverse(jax_box), positions)


def _restricted_box_inverse(box):
    a = box[0, 0]
    b = box[0, 1]
    c = box[0, 2]
    d = box[1, 1]
    e = box[1, 2]
    f = box[2, 2]
    return jnp.array(
        [
            [1.0 / a, -b / (a * d), (b * e - c * d) / (a * d * f)],
            [0.0, 1.0 / d, -e / (d * f)],
            [0.0, 0.0, 1.0 / f],
        ],
        dtype=box.dtype,
    )


def _energySO3LR(
    state,
    model,
    species,
    total_charge,
    pbc: bool,
    neighbors,
    neighbors_lr,
):
    positions_nm, box_vectors_nm = state
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if pbc and box_vectors_nm is not None:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        positions = fractional_coordinates(positions, box_vectors)
        positions = positions - jnp.floor(positions)
    energy = model(
        positions,
        species,
        box_vectors=box_vectors,
        neighbors=neighbors,
        neighbors_lr=neighbors_lr,
        periodic=pbc,
        total_charge=total_charge,
    )
    return energy * model.ev_to_kjmol
