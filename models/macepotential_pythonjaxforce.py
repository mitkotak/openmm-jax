from __future__ import annotations

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import openmm
import openmm.app as app
from jax_md import space
from openmm import unit
from openmmjax import PythonJaxForce
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .mace import (
    HARTREE_TO_KJMOL,
    MACE_MODEL_NAMES,
    get_neighbors,
    load_model,
)

jax.config.update("jax_default_matmul_precision", "highest")


class MACEPythonJaxForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **args):
        if name.endswith("-pythonjaxforce"):
            name = name[: -len("-pythonjaxforce")]
        return MACEPythonJaxForcePotentialImpl(name, modelPath=modelPath)


class MACEPythonJaxForcePotentialImpl(MLPotentialImpl):
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
            raise ValueError("MACE PythonJaxForce does not support atom subsets")
        includedAtoms = list(topology.atoms())
        species = jnp.array(
            [atom.element.atomic_number for atom in includedAtoms],
            dtype=jnp.int32,
        )

        model_ref = self.modelPath if modelPath is None else modelPath
        if model_ref is None:
            if self.name not in MACE_MODEL_NAMES:
                raise ValueError(
                    "modelPath must be provided for custom MACE PythonJaxForce models"
                )
            model_ref = self.name
        model = load_model(
            model_ref,
            neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
            neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
        )

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
            raise ValueError("MACE PythonJaxForce requires preprocessing_positions.")
        if hasattr(preprocessing_positions, "value_in_unit"):
            allocation_positions = preprocessing_positions.value_in_unit(unit.angstrom)
        else:
            scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
            allocation_positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
        allocation_positions = jnp.asarray(allocation_positions, dtype=jnp.float32)
        neighbor_list = allocate_neighbor_list(
            allocation_box,
            allocation_positions,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )

        compute = _ComputeMACEPythonJaxForce(
            model=model,
            species=species,
            neighbor_list=neighbor_list,
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


for model_name in MACE_MODEL_NAMES:
    MLPotential.registerImplFactory(
        f"{model_name}-pythonjaxforce", MACEPythonJaxForcePotentialImplFactory()
    )

__all__ = [
    "MLPotential",
    "MACEPythonJaxForcePotentialImplFactory",
    "MACEPythonJaxForcePotentialImpl",
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
        cutoff=float(cutoff),
        cell_atom_threshold=int(cell_atom_threshold),
        cell_capacity_multiplier=float(cell_capacity_multiplier),
        periodic=periodic,
    )


def _energyMACE(
    positions_nm,
    box_vectors_nm,
    model,
    species,
    neighbor_list,
    periodic: bool,
):
    """Evaluate MACE energy in kJ/mol from OpenMM positions in nm."""
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if periodic:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        positions = _fractional_positions(positions, box_vectors)
    energy = model(
        positions,
        species,
        box_vectors=box_vectors,
        neighbors=neighbor_list,
        periodic=periodic,
    )
    return energy * HARTREE_TO_KJMOL


def _fractional_positions(positions, box_vectors):
    fractional = fractional_coordinates(positions, box_vectors)
    return fractional - jnp.floor(fractional)


def fractional_coordinates(positions, box_vectors):
    openmm_box = jnp.swapaxes(jnp.asarray(box_vectors, dtype=positions.dtype), -1, -2)
    return space.transform(_restricted_box_inverse(openmm_box), positions)


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


class _ComputeMACEPythonJaxForce:
    def __init__(
        self,
        *,
        model,
        species,
        neighbor_list,
        periodic: bool,
    ):
        self.model = model
        self.species = species
        self.neighbor_list = neighbor_list
        self.periodic = bool(periodic)
        self._energy = None
        self._forces = None
        self._energy_and_grad = None

    def _energy_kjmol(self, positions_nm, box_vectors_nm=None):
        return _energyMACE(
            positions_nm,
            box_vectors_nm,
            model=self.model,
            species=self.species,
            neighbor_list=self.neighbor_list,
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
