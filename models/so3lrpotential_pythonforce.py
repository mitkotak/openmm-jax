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

from .so3lr import SO3LR_MODEL_NAMES, get_sparse_neighbors, load_model

jax.config.update("jax_default_matmul_precision", "highest")


class SO3LRPythonForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, charge: float = 0.0, **args):
        if name.endswith("-python"):
            name = name[: -len("-python")]
        return SO3LRPythonForcePotentialImpl(name, modelPath=modelPath, charge=charge)


class SO3LRPythonForcePotentialImpl(MLPotentialImpl):
    def __init__(self, name, modelPath=None, charge: float = 0.0):
        self.name = name
        self.modelPath = modelPath
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
        indices = None if atoms is None else np.asarray(atoms, dtype=np.int32)

        model_ref = self.modelPath if modelPath is None else modelPath
        if model_ref is None:
            if self.name not in SO3LR_MODEL_NAMES:
                raise ValueError("modelPath must be provided for custom SO3LR PythonForce models")
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
            raise ValueError("SO3LR PythonForce requires preprocessing_positions.")
        if hasattr(preprocessing_positions, "value_in_unit"):
            allocation_positions = preprocessing_positions.value_in_unit(unit.angstrom)
        else:
            scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
            allocation_positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
        allocation_positions = jnp.asarray(allocation_positions, dtype=jnp.float32)
        if atoms is not None:
            allocation_positions = allocation_positions[jnp.asarray(atoms, dtype=jnp.int32)]
        neighbor_list = allocate_neighbor_list(
            allocation_box,
            allocation_positions,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )
        lr_neighbor_list = allocate_neighbor_list(
            allocation_box,
            allocation_positions,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.long_range_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )
        model_charge = self.charge if charge is None else charge
        if total_charge is not None:
            model_charge = total_charge

        force = openmm.PythonForce(
            _ComputeSO3LRPythonForce(
                model=model,
                species=species,
                total_charge=jnp.asarray(model_charge, dtype=jnp.float32),
                neighbor_list=neighbor_list,
                lr_neighbor_list=lr_neighbor_list,
                indices=indices,
                periodic=use_periodic_neighbors,
            )
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(use_periodic_neighbors)
        system.addForce(force)


for model_name in SO3LR_MODEL_NAMES:
    MLPotential.registerImplFactory(
        f"{model_name}-python",
        SO3LRPythonForcePotentialImplFactory(),
    )

__all__ = [
    "MLPotential",
    "SO3LRPythonForcePotentialImplFactory",
    "SO3LRPythonForcePotentialImpl",
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
    return get_sparse_neighbors(
        positions_angstrom,
        box_vectors_angstrom,
        cutoff=float(cutoff),
        cell_atom_threshold=int(cell_atom_threshold),
        cell_capacity_multiplier=float(cell_capacity_multiplier),
        periodic=periodic,
    )


def _energySO3LR(
    positions_nm,
    box_vectors_nm,
    model,
    species,
    total_charge,
    neighbor_list,
    lr_neighbor_list,
    periodic: bool,
):
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
        neighbors_lr=lr_neighbor_list,
        periodic=periodic,
        total_charge=total_charge,
    )
    return energy * model.ev_to_kjmol


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


class _ComputeSO3LRPythonForce:
    def __init__(
        self,
        *,
        model,
        species,
        total_charge,
        neighbor_list,
        lr_neighbor_list,
        indices,
        periodic: bool,
    ):
        self.model = model
        self.species = species
        self.total_charge = total_charge
        self.neighbor_list = neighbor_list
        self.lr_neighbor_list = lr_neighbor_list
        self.periodic = bool(periodic)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int32)
        self.jax_indices = (
            None if self.indices is None else jnp.asarray(self.indices, dtype=jnp.int32)
        )
        self._energy_and_grad = None

    def _energy_kjmol(self, positions_nm, box_vectors_nm=None):
        selected_positions = (
            positions_nm if self.jax_indices is None else positions_nm[self.jax_indices]
        )
        return _energySO3LR(
            selected_positions,
            box_vectors_nm,
            model=self.model,
            species=self.species,
            total_charge=self.total_charge,
            neighbor_list=self.neighbor_list,
            lr_neighbor_list=self.lr_neighbor_list,
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
        energy, energy_grad = self._compiled_energy_and_grad()(
            positions_nm,
            box_vectors_nm,
        )
        forces = -energy_grad
        return float(jax.device_get(energy)), np.asarray(jax.device_get(forces))
