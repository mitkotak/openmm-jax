"""OpenMM integration for the bundled AIMNet2 JAX model."""

from __future__ import annotations

import math
from functools import partial
from typing import Iterable, Optional

import jax
import jax.numpy as jnp
import numpy as np
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

from .aimnet2 import (
    AIMNET2_MODEL_NAMES,
    get_neighbors,
    load_aimnet2_model,
)


class AIMNet2PotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name, modelPath=None, **_args):
        return AIMNet2PotentialImpl(name, modelPath=modelPath)


class AIMNet2PotentialImpl(MLPotentialImpl):
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
        charge: float = 0.0,
        total_charge: Optional[float] = None,
        periodic_neighborlist: bool = True,
        **_args,
    ):
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        species_np = np.asarray(
            [atom.element.atomic_number for atom in includedAtoms],
            dtype=np.int32,
        )
        model_ref = modelPath if modelPath is not None else self.modelPath
        if model_ref is None:
            if self.name in AIMNET2_MODEL_NAMES:
                model = load_aimnet2_model(self.name)
            else:
                raise ValueError("modelPath must be provided for custom AIMNet2 models")
        else:
            model = load_aimnet2_model(self.name, model_path=model_ref)
        unsupported = sorted(set(species_np.tolist()) - set(model.implemented_species))
        if unsupported:
            supported = ", ".join(str(z) for z in model.implemented_species)
            raise ValueError(
                f"AIMNet2 does not support atomic numbers {unsupported}. "
                f"Supported atomic numbers: {supported}."
            )
        species = jnp.asarray(species_np, dtype=jnp.int32)
        indices = None if atoms is None else jnp.array(atoms, dtype=jnp.int32)
        numSystemAtoms = system.getNumParticles()

        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        forcePeriodic = periodic and periodic_neighborlist
        allocation_box = None
        if forcePeriodic:
            box_vectors = topology.getPeriodicBoxVectors()
            if box_vectors is None:
                box_vectors = system.getDefaultPeriodicBoxVectors()
            allocation_box = jnp.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                dtype=jnp.float32,
            )
        neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=forcePeriodic,
        )
        lr_neighbor_list = allocate_neighbor_list(
            len(includedAtoms),
            allocation_box,
            cell_atom_threshold=int(model.neighbor_cell_atom_threshold),
            cutoff=float(model.lr_cutoff),
            cell_capacity_multiplier=float(model.neighbor_cell_capacity_multiplier),
            periodic=forcePeriodic,
        )
        if model.d3_c6ab is None or model.d3_rcov is None or model.d3_r2r4 is None:
            raise ValueError("AIMNet2 checkpoint does not contain D3 parameters.")
        unique_z = np.unique(species_np)
        z_to_idx = np.zeros(int(unique_z.max()) + 1, dtype=np.int32)
        for i, z in enumerate(unique_z):
            z_to_idx[int(z)] = i
        species_idx = z_to_idx[species_np]
        unique_z_jax = jnp.asarray(unique_z, dtype=jnp.int32)
        d3_data = {
            "c6ab": model.d3_c6ab[unique_z_jax[:, None], unique_z_jax[None, :]],
            "rcov": model.d3_rcov[unique_z_jax],
            "r2r4": model.d3_r2r4[unique_z_jax],
            "species_idx": jnp.asarray(species_idx, dtype=jnp.int32),
            "d3_s6": float(model.d3_s6),
            "d3_s8": float(model.d3_s8),
            "d3_a1": float(model.d3_a1),
            "d3_a2": float(model.d3_a2),
            "d3_k1": float(model.d3_k1),
            "d3_k3": float(model.d3_k3),
            "bohr_a": float(model.bohr_a),
            "hartree_ev": float(model.hartree_ev),
        }
        model_charge = charge if total_charge is None else total_charge

        energy_fn = partial(
            _energyAIMNet2,
            model=model,
            species=species,
            total_charge=jnp.asarray(model_charge, dtype=jnp.float32),
            d3_data=d3_data,
            pbc=forcePeriodic,
            neighbor_list=neighbor_list,
            lr_neighbor_list=lr_neighbor_list,
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


for model_name in AIMNET2_MODEL_NAMES:
    MLPotential.registerImplFactory(model_name, AIMNet2PotentialImplFactory())

__all__ = [
    "MLPotential",
    "AIMNet2PotentialImplFactory",
    "AIMNet2PotentialImpl",
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


def _energyAIMNet2(
    state,
    model,
    species,
    total_charge,
    d3_data,
    pbc: bool,
    neighbor_list,
    lr_neighbor_list,
):
    """Evaluate AIMNet2 energy in kJ/mol from OpenMM positions in nm."""
    positions_nm, box_vectors_nm = state
    positions = positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
    box_vectors = None
    if pbc and box_vectors_nm is not None:
        box_vectors = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        positions = fractional_coordinates(positions, box_vectors)
        positions = positions - jnp.floor(positions)
    return (
        model(
            positions,
            species,
            d3_data=d3_data,
            box_vectors=box_vectors,
            neighbors=neighbor_list,
            lr_neighbors=lr_neighbor_list,
            periodic=pbc,
            total_charge=total_charge,
        )
        * model.ev_to_kjmol
    )
