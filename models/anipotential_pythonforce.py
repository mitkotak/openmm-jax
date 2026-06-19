"""OpenMM PythonForce integration for the bundled ANI2x-JAX model."""

from __future__ import annotations

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.app as app
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .ani import (
    ANI2X_MODEL_NAMES,
    HARTREE_TO_KJMOL,
    get_neighbors,
    load_ani2x_model,
)


class ANI2xPythonForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **_args):
        return ANI2xPythonForcePotentialImpl(name, modelPath=modelPath)


class ANI2xPythonForcePotentialImpl(MLPotentialImpl):
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
        periodic_neighborlist: bool = False,
        **_args,
    ):
        if periodic_neighborlist:
            raise ValueError(
                "ANI2x-JAX PythonForce periodic neighbor lists are currently disabled."
            )

        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        species = jnp.array(
            [atom.element.atomic_number for atom in includedAtoms], dtype=jnp.int32
        )
        indices = None if atoms is None else np.asarray(atoms, dtype=np.int32)

        model_ref = self.modelPath if modelPath is None else modelPath
        if model_ref is None:
            if self.name in ANI2X_MODEL_NAMES:
                model_ref = self.name
            else:
                model_ref = "ani2x-jax-model0"
        model = load_ani2x_model(
            model_ref,
            atomic_numbers=species,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
        )
        neighbor_cell_atom_threshold = int(model.neighbor_cell_atom_threshold)
        model_species = model.species_indices(species)
        radial_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.radial_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        )
        angular_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.angular_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        )

        force = openmm.PythonForce(
            _ComputeANI2xPythonForce(
                model=model,
                species=model_species,
                radial_neighbor_list=radial_neighbor_list,
                angular_neighbor_list=angular_neighbor_list,
                indices=indices,
            )
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(False)
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


def allocate_neighbor_list(
    num_atoms: int,
    *,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
):
    allocation_positions = jnp.zeros((max(0, num_atoms), 3), dtype=jnp.float32)
    return get_neighbors(
        allocation_positions,
        cell_atom_threshold=cell_atom_threshold,
        cutoff=float(cutoff),
        cell_capacity_multiplier=cell_capacity_multiplier,
        periodic=False,
    )


def _energyANI(
    positions_nm,
    model,
    species,
    radial_neighbor_list,
    angular_neighbor_list,
):
    """Evaluate ANI2x energy in kJ/mol from OpenMM positions in nm."""
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    radial_neighbors = get_neighbors(
        positions,
        cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
        cutoff=float(model.radial_cutoff),
        cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        periodic=False,
        neighbors=radial_neighbor_list,
    )
    angular_neighbors = get_neighbors(
        positions,
        cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
        cutoff=float(model.angular_cutoff),
        cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        periodic=False,
        neighbors=angular_neighbor_list,
    )
    return (
        jnp.sum(
            model.node_energies(
                positions,
                species,
                radial_neighbor_idx=radial_neighbors.idx,
                angular_neighbor_idx=angular_neighbors.idx,
                box_vectors=None,
            )
        )
        * HARTREE_TO_KJMOL
    )


class _ComputeANI2xPythonForce:
    def __init__(
        self,
        *,
        model,
        species,
        radial_neighbor_list,
        angular_neighbor_list,
        indices,
    ):
        self.model = model
        self.species = species
        self.radial_neighbor_list = radial_neighbor_list
        self.angular_neighbor_list = angular_neighbor_list
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int32)
        self._jax_indices = (
            None if self.indices is None else jnp.asarray(self.indices, dtype=jnp.int32)
        )
        self._energy_and_grad = None

    def _energy_kjmol(self, positions_nm):
        selected_positions = (
            positions_nm if self._jax_indices is None else positions_nm[self._jax_indices]
        )
        return _energyANI(
            selected_positions,
            model=self.model,
            species=self.species,
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
        energy, energy_grad = self._compiled_energy_and_grad()(positions_nm)
        forces = -energy_grad
        return float(jax.device_get(energy)), np.asarray(jax.device_get(forces))
