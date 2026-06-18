"""OpenMM integration for the bundled MACE-OFF JAX models."""

from __future__ import annotations

import math
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

from .mace import (
    HARTREE_TO_KJMOL,
    MACE_MODEL_NAMES,
    get_neighbors,
    load_model,
)

jax.config.update("jax_default_matmul_precision", "highest")


class MACEPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **_args):
        return MACEPotentialImpl(name, modelPath=modelPath)


class MACEPotentialImpl(MLPotentialImpl):
    def __init__(self, name, modelPath=None):
        self.name = name
        self.modelPath = modelPath

    def addForces(
        self,
        topology: app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        modelPath: Optional[str] = None,
        neighbor_cell_atom_threshold: Optional[int] = None,
        neighbor_cell_capacity_multiplier: Optional[float] = None,
        periodic_neighborlist: bool = True,
        **_args,
    ):
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        species = jnp.array(
            [atom.element.atomic_number for atom in includedAtoms],
            dtype=jnp.int32,
        )
        indices = None if atoms is None else jnp.array(atoms, dtype=jnp.int32)
        numSystemAtoms = system.getNumParticles()

        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        forcePeriodic = periodic and periodic_neighborlist
        model_ref = modelPath if modelPath is not None else self.modelPath
        if model_ref is None:
            if self.name in MACE_MODEL_NAMES:
                model_ref = self.name
            else:
                raise ValueError("modelPath must be provided for custom MACE models")
        model = load_model(
            model_ref,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
            neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
        )
        neighbor_cell_atom_threshold = int(model.neighbor_cell_atom_threshold)

        allocation_box = None
        if forcePeriodic:
            box_vectors = topology.getPeriodicBoxVectors()
            if box_vectors is None:
                box_vectors = system.getDefaultPeriodicBoxVectors()
            allocation_box = jnp.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                dtype=jnp.float32,
            )
        neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=forcePeriodic,
        )
        energy_fn = partial(
            _energyMACE,
            model=model,
            species=species,
            pbc=forcePeriodic,
            neighbor_list=neighbor_list,
        )

        def _energy_kjmol(positions_nm, box_vectors_nm=None):
            selected_positions = positions_nm if indices is None else positions_nm[indices]
            return energy_fn((selected_positions, box_vectors_nm))

        def _energy_and_forces_kjmol(positions_nm, box_vectors_nm=None):
            energy, minus_forces = jax.value_and_grad(_energy_kjmol)(positions_nm, box_vectors_nm)
            return energy, -minus_forces

        def _forces_kjmol(positions_nm, box_vectors_nm=None):
            _energy, forces = _energy_and_forces_kjmol(positions_nm, box_vectors_nm)
            return forces

        force_mlir, energy_mlir, energy_and_forces_mlir, compile_options_base64 = export_jax_model(
            num_system_atoms=numSystemAtoms,
            force_function=_forces_kjmol,
            energy_function=_energy_kjmol,
            energy_and_forces_function=_energy_and_forces_kjmol,
            periodic=forcePeriodic,
        )
        force = openmmjax.JaxForce(
            force_mlir,
            energy_mlir,
            energy_and_forces_mlir,
            compile_options_base64,
        )
        configure_pjrt_plugin(force)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(forcePeriodic)
        force.setOutputsForces(True)
        force.addToSystem(system)


for model_name in MACE_MODEL_NAMES:
    MLPotential.registerImplFactory(model_name, MACEPotentialImplFactory())
MLPotential.registerImplFactory("mace-jax", MACEPotentialImplFactory())

__all__ = [
    "MLPotential",
    "MACEPotentialImplFactory",
    "MACEPotentialImpl",
]


def allocate_neighbor_list(
    num_atoms: int,
    allocation_box=None,
    *,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
    extra_capacity: int = 16,
    periodic: bool,
):
    allocation_positions = neighbor_allocation_positions(
        num_atoms,
        dtype=jnp.float32,
        periodic=periodic,
    )
    return get_neighbors(
        allocation_positions,
        allocation_box,
        cutoff=float(cutoff),
        cell_atom_threshold=cell_atom_threshold,
        cell_capacity_multiplier=cell_capacity_multiplier,
        periodic=periodic,
        extra_capacity=extra_capacity,
    )


def fractional_coordinates(positions, box_vectors):
    jax_box = jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2)
    return space.transform(_restricted_box_inverse(jax_box), positions)


def _restricted_box_inverse(box):
    """Invert OpenMM's restricted upper-triangular box without a solver op."""
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


def neighbor_allocation_positions(
    num_atoms: int,
    *,
    dtype=jnp.float32,
    periodic: bool,
):
    if num_atoms <= 0:
        return jnp.zeros((0, 3), dtype=dtype)
    if not periodic:
        return jnp.zeros((num_atoms, 3), dtype=dtype)

    grid_size = max(1, math.ceil(num_atoms ** (1.0 / 3.0)))
    atom_ids = jnp.arange(num_atoms, dtype=jnp.int32)
    x = atom_ids % grid_size
    y = (atom_ids // grid_size) % grid_size
    z = atom_ids // (grid_size * grid_size)
    return (jnp.stack((x, y, z), axis=-1).astype(dtype) + 0.5) / grid_size


def _energyMACE(
    state,
    model,
    species,
    pbc: bool,
    neighbor_list,
):
    """Evaluate MACE energy in kJ/mol from OpenMM positions in nm."""
    positions_nm, box_vectors_nm = state
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if pbc and box_vectors_nm is not None:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        positions = fractional_coordinates(positions, box_vectors)
        positions = positions - jnp.floor(positions)
    neighbors = get_neighbors(
        positions,
        box_vectors,
        cutoff=float(model.cutoff),
        cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
        cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        periodic=pbc,
        neighbors=neighbor_list,
        extra_capacity=16,
    )
    return (
        model(
            positions,
            species,
            box_vectors=box_vectors,
            neighbors=neighbors,
            periodic=pbc,
        )
        * HARTREE_TO_KJMOL
    )
