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

from .orb import (
    ORB_MODEL_NAMES,
    get_neighbors,
    load_orb_model,
)

jax.config.update("jax_default_matmul_precision", "highest")


class OrbPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(
        self,
        name,
        modelPath=None,
        charge: float = 0.0,
        total_charge: Optional[float] = None,
        multiplicity: int = 1,
        **_args,
    ):
        return OrbPotentialImpl(
            name,
            modelPath=modelPath,
            charge=charge,
            total_charge=total_charge,
            multiplicity=multiplicity,
        )


class OrbPotentialImpl(MLPotentialImpl):
    def __init__(
        self,
        name,
        modelPath=None,
        charge: float = 0.0,
        total_charge: Optional[float] = None,
        multiplicity: int = 1,
    ):
        self.name = name
        self.modelPath = modelPath
        self.charge = charge
        self.total_charge = total_charge
        self.multiplicity = multiplicity

    def addForces(
        self,
        topology: app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        modelPath: Optional[str] = None,
        charge: Optional[float] = None,
        total_charge: Optional[float] = None,
        multiplicity: Optional[int] = None,
        neighbor_cell_atom_threshold: Optional[int] = None,
        neighbor_cell_capacity_multiplier: Optional[float] = None,
        periodic_neighborlist: bool = True,
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        **_args,
    ):
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        atomic_numbers = [atom.element.atomic_number for atom in includedAtoms]
        species = jnp.array(atomic_numbers, dtype=jnp.int32)
        indices = None if atoms is None else jnp.array(atoms, dtype=jnp.int32)
        numSystemAtoms = system.getNumParticles() or topology.getNumAtoms()

        model_ref = modelPath if modelPath is not None else self.modelPath
        if model_ref is None:
            if self.name in ORB_MODEL_NAMES:
                model_ref = self.name
            else:
                raise ValueError("modelPath must be provided for custom ORB models")
        model = load_orb_model(
            model_ref,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
            neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
        )

        unsupported = sorted(
            set(z for z in atomic_numbers if z >= model.num_species_embeddings)
        )
        if unsupported:
            raise ValueError(f"ORB does not support atomic numbers {unsupported}.")

        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        forcePeriodic = periodic and periodic_neighborlist
        allocation_box = None
        if forcePeriodic:
            box_vectors = topology.getPeriodicBoxVectors()
            if box_vectors is None:
                box_vectors = system.getDefaultPeriodicBoxVectors()
            allocation_box = jnp.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                dtype=jnp.float32,
            )

        model_charge = self.charge if charge is None else charge
        if total_charge is not None:
            model_charge = total_charge
        elif self.total_charge is not None:
            model_charge = self.total_charge
        model_multiplicity = self.multiplicity if multiplicity is None else multiplicity

        allocation_positions = preprocessing_allocation_positions(
            preprocessing_positions,
            atoms,
            preprocessing_positions_unit,
        )
        neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            allocation_positions=allocation_positions,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=forcePeriodic,
        )

        total_charge_jax = jnp.asarray([model_charge], dtype=jnp.float32)
        total_spin_jax = jnp.asarray([model_multiplicity], dtype=jnp.float32)
        energy_fn = partial(
            _energyORB,
            model=model,
            species=species,
            total_charge=total_charge_jax,
            total_spin=total_spin_jax,
            pbc=forcePeriodic,
            neighbor_list=neighbor_list,
        )

        def _energy_kjmol(positions_nm, box_vectors_nm=None):
            selected_positions = positions_nm if indices is None else positions_nm[indices]
            return energy_fn((selected_positions, box_vectors_nm))

        def _energy_and_forces_kjmol(positions_nm, box_vectors_nm=None):
            energy, minus_forces = jax.value_and_grad(_energy_kjmol)(
                positions_nm,
                box_vectors_nm,
            )
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


for model_name in ORB_MODEL_NAMES:
    MLPotential.registerImplFactory(model_name, OrbPotentialImplFactory())

def _energyORB(
    state,
    model,
    species,
    total_charge,
    total_spin,
    pbc: bool,
    neighbor_list,
):
    """Evaluate ORB energy in kJ/mol from OpenMM positions in nm."""

    positions_nm, box_vectors_nm = state
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if pbc and box_vectors_nm is not None:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        positions = fractional_coordinates(positions, box_vectors)
        positions = positions - jnp.floor(positions)
    return (
        model(
            positions,
            species,
            total_charge,
            total_spin,
            box_vectors=box_vectors,
            neighbors=neighbor_list,
            periodic=pbc,
        )
        * model.ev_to_kjmol
    )


def allocate_neighbor_list(
    num_atoms: int,
    allocation_box=None,
    *,
    allocation_positions=None,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
    periodic: bool,
):
    if allocation_positions is None:
        raise ValueError("ORB JAX requires preprocessing_positions.")
    if periodic:
        if allocation_box is None:
            raise ValueError("periodic neighbor-list allocation requires a box.")
        allocation_positions = fractional_coordinates(allocation_positions, allocation_box)
    return get_neighbors(
        allocation_positions,
        allocation_box,
        cutoff=float(cutoff),
        cell_atom_threshold=cell_atom_threshold,
        cell_capacity_multiplier=cell_capacity_multiplier,
        periodic=periodic,
    )


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


def preprocessing_allocation_positions(
    preprocessing_positions,
    atoms,
    preprocessing_positions_unit,
):
    if preprocessing_positions is None:
        raise ValueError("ORB JAX requires preprocessing_positions.")
    if hasattr(preprocessing_positions, "value_in_unit"):
        positions = preprocessing_positions.value_in_unit(unit.angstrom)
    else:
        scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
        positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
    positions = jnp.asarray(positions, dtype=jnp.float32)
    if atoms is not None:
        positions = positions[jnp.asarray(atoms, dtype=jnp.int32)]
    return positions
