"""OpenMM integration for the bundled ANI2x-JAX model."""

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

from .ani import (
    ANI2X_MODEL_NAMES,
    HARTREE_TO_KJMOL,
    get_neighbors,
    load_ani2x_model,
)


class ANI2xPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **_args):
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
        periodic_neighborlist: bool = True,
        **_args,
    ):
        # Prepare inputs to the model

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        species = jnp.array(
            [atom.element.atomic_number for atom in includedAtoms], dtype=jnp.int32
        )
        indices = None if atoms is None else jnp.array(atoms, dtype=jnp.int32)
        numSystemAtoms = system.getNumParticles()

        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        forcePeriodic = periodic and periodic_neighborlist
        model_ref = modelPath if modelPath is not None else self.modelPath
        if model_ref is None:
            if self.name in ANI2X_MODEL_NAMES:
                model_ref = self.name
            else:
                raise ValueError("modelPath must be provided for custom ANI2x models")
        model = load_ani2x_model(
            model_ref,
            atomic_numbers=species,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
        )
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
        radial_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.radial_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=forcePeriodic,
        )
        angular_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.angular_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=forcePeriodic,
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


for model_name in ANI2X_MODEL_NAMES:
    MLPotential.registerImplFactory(model_name, ANI2xPotentialImplFactory())

__all__ = [
    "MLPotential",
    "ANI2xPotentialImplFactory",
    "ANI2xPotentialImpl",
]


def allocate_neighbor_list(
    num_atoms: int,
    allocation_box=None,
    *,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
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
        cell_atom_threshold=cell_atom_threshold,
        cutoff=float(cutoff),
        cell_capacity_multiplier=cell_capacity_multiplier,
        periodic=periodic,
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
    radial_neighbors = get_neighbors(
        positions,
        box_vectors,
        cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
        cutoff=float(model.radial_cutoff),
        cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        periodic=pbc,
        neighbors=radial_neighbor_list,
    )
    angular_neighbors = get_neighbors(
        positions,
        box_vectors,
        cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
        cutoff=float(model.angular_cutoff),
        cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        periodic=pbc,
        neighbors=angular_neighbor_list,
    )
    return (
        jnp.sum(
            model.node_energies(
                positions,
                species,
                radial_neighbor_idx=radial_neighbors.idx,
                angular_neighbor_idx=angular_neighbors.idx,
                box_vectors=box_vectors if pbc else None,
            )
        )
        * HARTREE_TO_KJMOL
    )
