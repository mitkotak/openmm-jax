from __future__ import annotations

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.app as app
from jax_md import space
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
            neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
        )
        neighbor_cell_atom_threshold = int(model.neighbor_cell_atom_threshold)
        model_species = model.species_indices(species)
        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        use_periodic_neighbors = periodic and periodic_neighborlist
        allocation_box = _initial_box_vectors_angstrom(
            topology,
            system,
            use_periodic_neighbors,
        )
        allocation_positions = preprocessing_allocation_positions(
            preprocessing_positions,
            atoms,
            preprocessing_positions_unit,
        )
        radial_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            allocation_positions=allocation_positions,
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.radial_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )
        angular_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            allocation_positions=allocation_positions,
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.angular_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )

        force = openmm.PythonForce(
            _ComputeANI2xPythonForce(
                model=model,
                species=model_species,
                radial_neighbor_list=radial_neighbor_list,
                angular_neighbor_list=angular_neighbor_list,
                indices=indices,
                periodic=use_periodic_neighbors,
            )
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(use_periodic_neighbors)
        system.addForce(force)


MLPotential.registerImplFactory("ani2x-jax-python", ANI2xPythonForcePotentialImplFactory())

__all__ = [
    "MLPotential",
    "ANI2xPythonForcePotentialImplFactory",
    "ANI2xPythonForcePotentialImpl",
]


def allocate_neighbor_list(
    num_atoms: int,
    box_vectors_angstrom,
    *,
    allocation_positions=None,
    cell_atom_threshold: int,
    cutoff: float,
    cell_capacity_multiplier: float,
    periodic: bool,
):
    positions = allocation_positions
    if positions is None:
        raise ValueError("ANI2x PythonForce requires preprocessing_positions.")
    if periodic:
        if box_vectors_angstrom is None:
            raise ValueError("periodic neighbor-list allocation requires a box.")
        positions = fractional_coordinates(positions, box_vectors_angstrom)
    return get_neighbors(
        positions,
        box_vectors_angstrom,
        cell_atom_threshold=cell_atom_threshold,
        cutoff=float(cutoff),
        cell_capacity_multiplier=cell_capacity_multiplier,
        periodic=periodic,
    )


def _initial_box_vectors_angstrom(topology, system, periodic: bool):
    if not periodic:
        return None
    box_vectors = topology.getPeriodicBoxVectors()
    if box_vectors is None:
        box_vectors = system.getDefaultPeriodicBoxVectors()
    return jnp.asarray(
        [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
        dtype=jnp.float32,
    )


def preprocessing_allocation_positions(
    preprocessing_positions,
    atoms,
    preprocessing_positions_unit,
):
    if preprocessing_positions is None:
        raise ValueError("ANI2x PythonForce requires preprocessing_positions.")
    if hasattr(preprocessing_positions, "value_in_unit"):
        positions = preprocessing_positions.value_in_unit(unit.angstrom)
    else:
        scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
        positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
    positions = jnp.asarray(positions, dtype=jnp.float32)
    if atoms is not None:
        positions = positions[jnp.asarray(atoms, dtype=jnp.int32)]
    return positions


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
        (
            (1.0 / a, -b / (a * d), (b * e - c * d) / (a * d * f)),
            (0.0, 1.0 / d, -e / (d * f)),
            (0.0, 0.0, 1.0 / f),
        ),
        dtype=box.dtype,
    )


def _energyANI(
    positions_nm,
    box_vectors_nm,
    model,
    species,
    radial_neighbor_list,
    angular_neighbor_list,
    periodic: bool,
):
    """Evaluate ANI2x energy in kJ/mol from OpenMM positions in nm."""
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if periodic:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        positions = fractional_coordinates(positions, box_vectors)
        positions = positions - jnp.floor(positions)
    radial_neighbors = get_neighbors(
        positions,
        box_vectors,
        cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
        cutoff=float(model.radial_cutoff),
        cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        periodic=periodic,
        neighbors=radial_neighbor_list,
    )
    angular_neighbors = get_neighbors(
        positions,
        box_vectors,
        cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
        cutoff=float(model.angular_cutoff),
        cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
        periodic=periodic,
        neighbors=angular_neighbor_list,
    )
    return (
        jnp.sum(
            model.node_energies(
                positions,
                species,
                radial_neighbor_idx=radial_neighbors.idx,
                angular_neighbor_idx=angular_neighbors.idx,
                box_vectors=box_vectors if periodic else None,
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
        periodic: bool,
    ):
        self.model = model
        self.species = species
        self.radial_neighbor_list = radial_neighbor_list
        self.angular_neighbor_list = angular_neighbor_list
        self.periodic = bool(periodic)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int32)
        self._jax_indices = (
            None if self.indices is None else jnp.asarray(self.indices, dtype=jnp.int32)
        )
        self._energy_and_grad = None

    def _energy_kjmol(self, positions_nm, box_vectors_nm=None):
        selected_positions = (
            positions_nm if self._jax_indices is None else positions_nm[self._jax_indices]
        )
        return _energyANI(
            selected_positions,
            box_vectors_nm,
            model=self.model,
            species=self.species,
            radial_neighbor_list=self.radial_neighbor_list,
            angular_neighbor_list=self.angular_neighbor_list,
            periodic=self.periodic,
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
        if self.periodic:
            box_vectors_nm = jnp.asarray(
                state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=jnp.float32,
            )
        energy, energy_grad = self._compiled_energy_and_grad()(positions_nm, box_vectors_nm)
        forces = -energy_grad
        return float(jax.device_get(energy)), np.asarray(jax.device_get(forces))
