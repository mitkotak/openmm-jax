"""OpenMM PythonForce integration for the bundled ANI2x-JAX model."""

from __future__ import annotations

import math
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.app as app
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .ani import (
    HARTREE_TO_KJMOL,
    get_neighbors,
    load_ani2x_model,
)


class ANI2xPythonForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **_args):
        return ANI2xPythonForcePotentialImpl(name, modelPath=modelPath)


class ANI2xPythonForcePotentialImpl(MLPotentialImpl):
    def __init__(self, _name, modelPath=None):
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
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        species = jnp.array(
            [atom.element.atomic_number for atom in includedAtoms], dtype=jnp.int32
        )
        indices = None if atoms is None else np.asarray(atoms, dtype=np.int32)

        periodic = (
            topology.getPeriodicBoxVectors() is not None
            or system.usesPeriodicBoundaryConditions()
        )
        forcePeriodic = periodic and periodic_neighborlist
        model_path = self.modelPath if modelPath is None else modelPath
        model = load_ani2x_model(
            model_path,
            atomic_numbers=species,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
        )
        neighbor_cell_atom_threshold = int(model.neighbor_cell_atom_threshold)
        model_species = jnp.asarray(model.species_to_index, dtype=jnp.int32)[species]
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

        force = openmm.PythonForce(
            _ComputeANI2xPythonForce(
                model=model,
                species=model_species,
                pbc=forcePeriodic,
                radial_neighbor_list=radial_neighbor_list,
                angular_neighbor_list=angular_neighbor_list,
                indices=indices,
            )
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(forcePeriodic)
        system.addForce(force)


MLPotential.registerImplFactory(
    "ani2x-jax-pythonforce", ANI2xPythonForcePotentialImplFactory()
)
MLPotential.registerImplFactory("ani2x-jax-python", ANI2xPythonForcePotentialImplFactory())

__all__ = [
    "MLPotential",
    "ANI2xPythonForcePotentialImplFactory",
    "ANI2xPythonForcePotentialImpl",
]


def fractional_coordinates(positions, box_vectors):
    z = positions[..., 2] / box_vectors[2, 2]
    y = (positions[..., 1] - z * box_vectors[2, 1]) / box_vectors[1, 1]
    x = (positions[..., 0] - y * box_vectors[1, 0] - z * box_vectors[2, 0]) / box_vectors[0, 0]
    return jnp.stack((x, y, z), axis=-1)


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


def allocate_neighbor_list(
    num_atoms: int,
    allocation_box,
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
        model(
            positions,
            species,
            box_vectors=box_vectors,
            radial_neighbors=radial_neighbors,
            angular_neighbors=angular_neighbors,
            periodic=pbc,
        )
        * HARTREE_TO_KJMOL
    )


class _ComputeANI2xPythonForce:
    def __init__(
        self,
        *,
        model,
        species,
        pbc: bool,
        radial_neighbor_list,
        angular_neighbor_list,
        indices,
    ):
        self.model = model
        self.species = species
        self.pbc = pbc
        self.radial_neighbor_list = radial_neighbor_list
        self.angular_neighbor_list = angular_neighbor_list
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int32)
        self._jax_indices = (
            None if self.indices is None else jnp.asarray(self.indices, dtype=jnp.int32)
        )
        self._energy_and_grad = None

    def _energy_kjmol(self, positions_nm, box_vectors_nm):
        selected_positions = (
            positions_nm if self._jax_indices is None else positions_nm[self._jax_indices]
        )
        return _energyANI(
            (selected_positions, box_vectors_nm),
            model=self.model,
            species=self.species,
            pbc=self.pbc,
            radial_neighbor_list=self.radial_neighbor_list,
            angular_neighbor_list=self.angular_neighbor_list,
        )

    def _compiled_energy_and_grad(self):
        if self._energy_and_grad is None:
            self._energy_and_grad = jax.jit(jax.value_and_grad(self._energy_kjmol))
        return self._energy_and_grad

    def __call__(self, state):
        positions_nm = jnp.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=jnp.float32,
        )
        box_vectors_nm = None
        if self.pbc:
            box_vectors_nm = jnp.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=jnp.float32,
            )

        energy, energy_grad = self._compiled_energy_and_grad()(
            positions_nm, box_vectors_nm
        )
        forces = -energy_grad
        return float(jax.device_get(energy)), np.asarray(jax.device_get(forces))
