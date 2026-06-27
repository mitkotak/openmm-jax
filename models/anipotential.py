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

from .ani import (
    ANI2X_MODEL_NAMES,
    HARTREE_TO_KJMOL,
    get_neighbors,
    load_ani2x_model,
)


class ANI2xPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **args):
        return ANI2xPotentialImpl(name, modelPath=modelPath)


class ANI2xPotentialImpl(MLPotentialImpl):
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
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        **args,
    ):
        # Prepare inputs to the model

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        model_ref = modelPath if modelPath is not None else self.modelPath
        if model_ref is None:
            if self.name in ANI2X_MODEL_NAMES:
                model_ref = self.name
            else:
                raise ValueError("modelPath must be provided for custom ANI2x models")
        atomic_numbers = [atom.element.atomic_number for atom in includedAtoms]
        model = load_ani2x_model(
            model_ref,
            atomic_numbers=jnp.asarray(atomic_numbers, dtype=jnp.int32),
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
            neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
        )
        unsupported = [
            z
            for z in sorted(set(atomic_numbers))
            if z >= len(model.species_to_index) or model.species_to_index[z] == 0 and z != 1
        ]
        if unsupported:
            raise ValueError(f"ANI2x does not support atomic numbers {unsupported}.")
        species = jnp.array(atomic_numbers, dtype=jnp.int32)
        indices = None if atoms is None else jnp.array(atoms, dtype=jnp.int32)
        numSystemAtoms = system.getNumParticles() or topology.getNumAtoms()

        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        forcePeriodic = periodic and periodic_neighborlist
        neighbor_cell_atom_threshold = int(model.neighbor_cell_atom_threshold)
        model_species = model.species_indices(species)
        allocation_box = None
        if forcePeriodic:
            box_vectors = topology.getPeriodicBoxVectors()
            if box_vectors is None:
                box_vectors = system.getDefaultPeriodicBoxVectors()
            allocation_box = jnp.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                dtype=jnp.float32,
            )
        if preprocessing_positions is None:
            raise ValueError("ANI2x JAX requires preprocessing_positions.")
        if hasattr(preprocessing_positions, "value_in_unit"):
            allocation_positions = preprocessing_positions.value_in_unit(unit.angstrom)
        else:
            scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
            allocation_positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
        allocation_positions = jnp.asarray(allocation_positions, dtype=jnp.float32)
        if atoms is not None:
            allocation_positions = allocation_positions[jnp.asarray(atoms, dtype=jnp.int32)]

        def _allocate_neighbor_lists(box_vectors_angstrom, positions_angstrom):
            radial_neighbor_list = allocate_neighbor_list(
                box_vectors_angstrom,
                positions_angstrom,
                cell_atom_threshold=neighbor_cell_atom_threshold,
                cutoff=float(model.radial_cutoff),
                cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                periodic=forcePeriodic,
            )
            angular_neighbor_list = allocate_neighbor_list(
                box_vectors_angstrom,
                positions_angstrom,
                cell_atom_threshold=neighbor_cell_atom_threshold,
                cutoff=float(model.angular_cutoff),
                cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
                periodic=forcePeriodic,
            )
            return radial_neighbor_list, angular_neighbor_list

        radial_neighbor_list, angular_neighbor_list = _allocate_neighbor_lists(
            allocation_box,
            allocation_positions,
        )
        energy_fn = partial(
            _energyANI,
            model=model,
            species=model_species,
            pbc=forcePeriodic,
            radial_neighbor_list=radial_neighbor_list,
            angular_neighbor_list=angular_neighbor_list,
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
            return -jax.grad(_energy_kjmol)(positions_nm, box_vectors_nm)

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


for model_name in ANI2X_MODEL_NAMES:
    MLPotential.registerImplFactory(model_name, ANI2xPotentialImplFactory())

__all__ = [
    "MLPotential",
    "ANI2xPotentialImplFactory",
    "ANI2xPotentialImpl",
]


def allocate_neighbor_list(
    box_vectors_angstrom,
    positions_angstrom,
    *,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
    periodic: bool,
):
    if periodic:
        if box_vectors_angstrom is None:
            raise ValueError("periodic neighbor-list allocation requires a box.")
        positions_angstrom = fractional_coordinates(positions_angstrom, box_vectors_angstrom)
    return get_neighbors(
        positions_angstrom,
        box_vectors_angstrom,
        cell_atom_threshold=cell_atom_threshold,
        cutoff=float(cutoff),
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


def _energyANI(
    state,
    model,
    species,
    pbc: bool,
    radial_neighbor_list,
    angular_neighbor_list,
):
    """Evaluate ANI2x energy in kJ/mol from OpenMM positions in nm."""
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
        radial_neighbors=radial_neighbor_list,
        angular_neighbors=angular_neighbor_list,
        periodic=pbc,
    )
    if isinstance(energy, tuple):
        energy = energy[0]
    return energy * HARTREE_TO_KJMOL
