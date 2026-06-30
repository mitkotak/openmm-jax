from __future__ import annotations

from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import openmm
import openmm.app as app
from openmm import unit
from openmmjax import PythonJaxForce
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .so3lr import SO3LR_MODEL_NAMES, get_sparse_neighbors, load_model

jax.config.update("jax_default_matmul_precision", "highest")


class SO3LRPythonJaxForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, charge: float = 0.0, **args):
        if name.endswith("-pythonjaxforce"):
            name = name[: -len("-pythonjaxforce")]
        return SO3LRPythonJaxForcePotentialImpl(name, modelPath=modelPath, charge=charge)


class SO3LRPythonJaxForcePotentialImpl(MLPotentialImpl):
    def __init__(self, name, modelPath=None, charge: float = 0.0):
        if modelPath is not None:
            raise ValueError("SO3LR only supports the bundled checkpoint.")
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
        if modelPath is not None:
            raise ValueError("SO3LR only supports the bundled checkpoint.")
        if atoms is not None:
            raise ValueError("SO3LR PythonJaxForce does not support atom subsets")
        included_atoms = list(topology.atoms())
        species = jnp.asarray(
            [atom.element.atomic_number for atom in included_atoms],
            dtype=jnp.int32,
        )

        model_ref = self.modelPath if modelPath is None else modelPath
        if model_ref is None:
            if self.name not in SO3LR_MODEL_NAMES:
                raise ValueError(
                    "modelPath must be provided for custom SO3LR PythonJaxForce models"
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
            raise ValueError("SO3LR PythonJaxForce requires preprocessing_positions.")
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

        compute = _ComputeSO3LRPythonJaxForce(
            model=model,
            species=species,
            total_charge=jnp.asarray(model_charge, dtype=jnp.float32),
            neighbor_list=neighbor_list,
            lr_neighbor_list=lr_neighbor_list,
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


for model_name in SO3LR_MODEL_NAMES:
    MLPotential.registerImplFactory(
        f"{model_name}-pythonjaxforce",
        SO3LRPythonJaxForcePotentialImplFactory(),
    )

__all__ = [
    "MLPotential",
    "SO3LRPythonJaxForcePotentialImplFactory",
    "SO3LRPythonJaxForcePotentialImpl",
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



class _ComputeSO3LRPythonJaxForce:
    def __init__(
        self,
        *,
        model,
        species,
        total_charge,
        neighbor_list,
        lr_neighbor_list,
        periodic: bool,
    ):
        self.model = model
        self.species = species
        self.total_charge = total_charge
        self.neighbor_list = neighbor_list
        self.lr_neighbor_list = lr_neighbor_list
        self.periodic = bool(periodic)
        self._energy = None
        self._forces = None
        self._energy_and_grad = None

    def _energy_kjmol(self, positions_nm, box_vectors_nm=None):
        return _energySO3LR(
            positions_nm,
            box_vectors_nm,
            model=self.model,
            species=self.species,
            total_charge=self.total_charge,
            neighbor_list=self.neighbor_list,
            lr_neighbor_list=self.lr_neighbor_list,
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
