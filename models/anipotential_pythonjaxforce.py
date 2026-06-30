from __future__ import annotations

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import openmm
import openmm.app as app
from openmm import unit
from openmmjax import PythonJaxForce
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .ani import (
    ANI2X_MODEL_NAMES,
    HARTREE_TO_KJMOL,
    get_neighbors,
    load_model,
)


class ANI2xPythonJaxForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **args):
        if name.endswith("-pythonjaxforce"):
            name = name[: -len("-pythonjaxforce")]
        return ANI2xPythonJaxForcePotentialImpl(name, modelPath=modelPath)


class ANI2xPythonJaxForcePotentialImpl(MLPotentialImpl):
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
        if atoms is not None:
            raise ValueError("ANI2x PythonJaxForce does not support atom subsets")
        includedAtoms = list(topology.atoms())
        species = jnp.array(
            [atom.element.atomic_number for atom in includedAtoms], dtype=jnp.int32
        )

        model_ref = self.modelPath if modelPath is None else modelPath
        if model_ref is None:
            if self.name in ANI2X_MODEL_NAMES:
                model_ref = self.name
            else:
                raise ValueError(
                    "modelPath must be provided for custom ANI2x PythonJaxForce models"
                )
        model = load_model(
            model_ref,
            atomic_numbers=species,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
            neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
        )
        unsupported = [
            z
            for z in sorted(set(species.tolist()))
            if z >= len(model.species_to_index) or model.species_to_index[z] == 0 and z != 1
        ]
        if unsupported:
            raise ValueError(f"ANI2x does not support atomic numbers {unsupported}.")
        neighbor_cell_atom_threshold = int(model.neighbor_cell_atom_threshold)
        model_species = model.species_indices(species)
        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        use_periodic_neighbors = periodic and periodic_neighborlist
        allocation_box = None
        if use_periodic_neighbors:
            box_vectors = topology.getPeriodicBoxVectors()
            if box_vectors is None:
                box_vectors = system.getDefaultPeriodicBoxVectors()
            allocation_box = jnp.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                dtype=jnp.float32,
            )
        if preprocessing_positions is None:
            raise ValueError("ANI2x PythonJaxForce requires preprocessing_positions.")
        if hasattr(preprocessing_positions, "value_in_unit"):
            allocation_positions = preprocessing_positions.value_in_unit(unit.angstrom)
        else:
            scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
            allocation_positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
        allocation_positions = jnp.asarray(allocation_positions, dtype=jnp.float32)
        radial_neighbor_list = allocate_neighbor_list(
            allocation_box,
            allocation_positions,
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.radial_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )
        angular_neighbor_list = allocate_neighbor_list(
            allocation_box,
            allocation_positions,
            cell_atom_threshold=neighbor_cell_atom_threshold,
            cutoff=float(model.angular_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )

        compute = _ComputeANI2xPythonJaxForce(
            model=model,
            species=model_species,
            radial_neighbor_list=radial_neighbor_list,
            angular_neighbor_list=angular_neighbor_list,
            periodic=use_periodic_neighbors,
        )
        force = PythonJaxForce(
            compute.energy,
            compute.forces,
            compute.energy_and_forces,
            {},
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(use_periodic_neighbors)
        system.addForce(force)


for model_name in ANI2X_MODEL_NAMES:
    MLPotential.registerImplFactory(
        f"{model_name}-pythonjaxforce",
        ANI2xPythonJaxForcePotentialImplFactory(),
    )

__all__ = [
    "MLPotential",
    "ANI2xPythonJaxForcePotentialImplFactory",
    "ANI2xPythonJaxForcePotentialImpl",
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
    if periodic and box_vectors_angstrom is None:
        raise ValueError("periodic neighbor-list allocation requires a box.")
    return get_neighbors(
        positions_angstrom,
        box_vectors_angstrom,
        cell_atom_threshold=cell_atom_threshold,
        cutoff=float(cutoff),
        cell_capacity_multiplier=float(cell_capacity_multiplier),
        periodic=periodic,
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
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if periodic:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    energy = model(
        positions,
        species,
        box_vectors=box_vectors,
        radial_neighbors=radial_neighbor_list,
        angular_neighbors=angular_neighbor_list,
        periodic=periodic,
    )
    return energy * HARTREE_TO_KJMOL


class _ComputeANI2xPythonJaxForce:
    def __init__(
        self,
        *,
        model,
        species,
        radial_neighbor_list,
        angular_neighbor_list,
        periodic: bool,
    ):
        self.model = model
        self.species = species
        self.radial_neighbor_list = radial_neighbor_list
        self.angular_neighbor_list = angular_neighbor_list
        self.periodic = bool(periodic)
        self._energy = None
        self._forces = None
        self._energy_and_grad = None

    def _energy_kjmol(self, positions_nm, box_vectors_nm=None):
        return _energyANI(
            positions_nm,
            box_vectors_nm,
            model=self.model,
            species=self.species,
            radial_neighbor_list=self.radial_neighbor_list,
            angular_neighbor_list=self.angular_neighbor_list,
            periodic=self.periodic,
        )

    def energy(self, positions_nm, box_vectors_nm=None):
        if self._energy is None:
            self._energy = jax.jit(self._energy_kjmol)
        return self._energy(positions_nm, box_vectors_nm)

    def forces(self, positions_nm, box_vectors_nm=None):
        if self._forces is None:
            self._forces = jax.jit(jax.grad(self._energy_kjmol))
        return -self._forces(positions_nm, box_vectors_nm)

    def energy_and_forces(self, positions_nm, box_vectors_nm=None):
        if self._energy_and_grad is None:
            self._energy_and_grad = jax.jit(jax.value_and_grad(self._energy_kjmol))
        energy, energy_grad = self._energy_and_grad(
            positions_nm,
            box_vectors_nm,
        )
        return energy, -energy_grad
