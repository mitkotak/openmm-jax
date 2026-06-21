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

from .aimnet2 import (
    AIMNET2_MODEL_NAMES,
    get_neighbors,
    load_aimnet2_model,
)


class AIMNet2PythonForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(
        self,
        name,
        modelPath=None,
        charge: float = 0.0,
        total_charge: Optional[float] = None,
        multiplicity: int = 1,
        **args,
    ):
        return AIMNet2PythonForcePotentialImpl(
            name,
            modelPath=modelPath,
            charge=charge,
            total_charge=total_charge,
            multiplicity=multiplicity,
        )


class AIMNet2PythonForcePotentialImpl(MLPotentialImpl):
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
        neighbor_cell_atom_threshold: Optional[int] = None,
        neighbor_cell_capacity_multiplier: Optional[float] = None,
        periodic_neighborlist: bool = True,
        multiplicity: Optional[int] = None,
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        **args,
    ):
        multiplicity = self.multiplicity if multiplicity is None else multiplicity
        if multiplicity != 1:
            raise ValueError("AIMNet2 JAX only supports multiplicity=1")
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        species_np = np.asarray(
            [atom.element.atomic_number for atom in includedAtoms],
            dtype=np.int32,
        )
        species = jnp.asarray(species_np, dtype=jnp.int32)
        indices = None if atoms is None else np.asarray(atoms, dtype=np.int32)

        model_ref = self.modelPath if modelPath is None else modelPath
        if model_ref is None:
            if self.name in AIMNET2_MODEL_NAMES:
                model = load_aimnet2_model(
                    self.name,
                    neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
                    neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
                )
            else:
                model = load_aimnet2_model(
                    _builtin_model_name(self.name),
                    neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
                    neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
                )
        else:
            model = load_aimnet2_model(
                self.name,
                model_path=model_ref,
                neighbor_cell_atom_threshold=neighbor_cell_atom_threshold,
                neighbor_cell_capacity_multiplier=neighbor_cell_capacity_multiplier,
            )

        unsupported = sorted(set(species_np.tolist()) - set(model.implemented_species))
        if unsupported:
            supported = ", ".join(str(z) for z in model.implemented_species)
            raise ValueError(
                f"AIMNet2 does not support atomic numbers {unsupported}. "
                f"Supported atomic numbers: {supported}."
            )
        if model.d3_c6ab is None or model.d3_rcov is None or model.d3_r2r4 is None:
            raise ValueError("AIMNet2 checkpoint does not contain D3 parameters.")

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
        neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            allocation_positions=allocation_positions,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )
        lr_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            allocation_positions=allocation_positions,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.lr_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=use_periodic_neighbors,
        )
        model_charge = self.charge if charge is None else charge
        if total_charge is not None:
            model_charge = total_charge
        elif self.total_charge is not None:
            model_charge = self.total_charge

        force = openmm.PythonForce(
            _ComputeAIMNet2PythonForce(
                model=model,
                species=species,
                d3_data=model.prepare_d3_data(species_np),
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


for model_name in AIMNET2_MODEL_NAMES:
    MLPotential.registerImplFactory(
        f"{model_name}-python", AIMNet2PythonForcePotentialImplFactory()
    )

__all__ = [
    "MLPotential",
    "AIMNet2PythonForcePotentialImplFactory",
    "AIMNet2PythonForcePotentialImpl",
]


def _builtin_model_name(name: str):
    if name in AIMNET2_MODEL_NAMES:
        return name
    for suffix in ("-python",):
        if name.endswith(suffix):
            model_name = name[: -len(suffix)]
            if model_name in AIMNET2_MODEL_NAMES:
                return model_name
    raise ValueError("modelPath must be provided for custom AIMNet2 PythonForce models")


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
        raise ValueError("AIMNet2 PythonForce requires preprocessing_positions.")
    if periodic:
        if box_vectors_angstrom is None:
            raise ValueError("periodic neighbor-list allocation requires a box.")
        positions = fractional_coordinates(positions, box_vectors_angstrom)
    return get_neighbors(
        positions,
        box_vectors_angstrom,
        cutoff=float(cutoff),
        cell_atom_threshold=int(cell_atom_threshold),
        cell_capacity_multiplier=float(cell_capacity_multiplier),
        periodic=periodic,
    )


def preprocessing_allocation_positions(
    preprocessing_positions,
    atoms,
    preprocessing_positions_unit,
):
    if preprocessing_positions is None:
        raise ValueError("AIMNet2 PythonForce requires preprocessing_positions.")
    if hasattr(preprocessing_positions, "value_in_unit"):
        positions = preprocessing_positions.value_in_unit(unit.angstrom)
    else:
        scale = preprocessing_positions_unit.conversion_factor_to(unit.angstrom)
        positions = jnp.asarray(preprocessing_positions, dtype=jnp.float32) * scale
    positions = jnp.asarray(positions, dtype=jnp.float32)
    if atoms is not None:
        positions = positions[jnp.asarray(atoms, dtype=jnp.int32)]
    return positions


def _energyAIMNet2(
    positions_nm,
    box_vectors_nm,
    model,
    species,
    total_charge,
    d3_data,
    neighbor_list,
    lr_neighbor_list,
    periodic: bool,
):
    """Evaluate AIMNet2 energy in kJ/mol from OpenMM positions in nm."""
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if periodic:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        positions = _fractional_positions(positions, box_vectors)
    return (
        model(
            positions,
            species,
            d3_data=d3_data,
            box_vectors=box_vectors,
            neighbors=neighbor_list,
            lr_neighbors=lr_neighbor_list,
            periodic=periodic,
            total_charge=total_charge,
        )
        * model.ev_to_kjmol
    )


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


class _ComputeAIMNet2PythonForce:
    def __init__(
        self,
        *,
        model,
        species,
        d3_data,
        total_charge,
        neighbor_list,
        lr_neighbor_list,
        indices,
        periodic: bool,
    ):
        self.model = model
        self.species = species
        self.d3_data = d3_data
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
        return _energyAIMNet2(
            selected_positions,
            box_vectors_nm,
            model=self.model,
            species=self.species,
            total_charge=self.total_charge,
            d3_data=self.d3_data,
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
        energy, energy_grad = self._compiled_energy_and_grad()(positions_nm, box_vectors_nm)
        forces = -energy_grad
        return float(jax.device_get(energy)), np.asarray(jax.device_get(forces))
