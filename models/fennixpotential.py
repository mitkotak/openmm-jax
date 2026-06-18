"""OpenMM integration for the FeNNix JAX model."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from functools import partial
from typing import Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import openmm
import openmm.app as app
import openmmjax
from openmm import unit
from openmmjax_export import (
    configure_pjrt_plugin,
    export_jax_model,
)
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory


class FeNNixPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(
        self,
        name: str,
        modelPath: str | None = None,
        **_args,
    ) -> MLPotentialImpl:
        return FeNNixPotentialImpl(name, modelPath)


class FeNNixPotentialImpl(MLPotentialImpl):
    KNOWN_MODELS = {
        "fennix-bio1-small-jax": (
            "https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/"
            "FENNIX-BIO1/v1.0/fennix-bio1S.fnx"
        ),
        "fennix-bio1-medium-jax": (
            "https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/"
            "FENNIX-BIO1/v1.0/fennix-bio1M.fnx"
        ),
        "fennix-bio1-small-finetune-ions-jax": (
            "https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/"
            "FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1S-finetuneIons.fnx"
        ),
        "fennix-bio1-medium-finetune-ions-jax": (
            "https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/"
            "FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1M-finetuneIons.fnx"
        ),
    }

    def __init__(self, name: str, modelPath: str | None = None) -> None:
        self.name = name
        self.modelPath = modelPath

    def addForces(
        self,
        topology: app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        charge: int = 0,
        gpu_preprocessing: bool = True,
        use_float64: bool = False,
        matmul_prec: Optional[str] = "highest",
        energy_terms: Optional[Sequence[str]] = None,
        periodic_neighborlist: bool = True,
        **args,
    ):
        import fennol
        import numpy as np

        if matmul_prec is not None:
            jax.config.update("jax_default_matmul_precision", matmul_prec)

        # Load the model.
        downloaded_model_path = None
        if self.modelPath is not None:
            modelPath = self.modelPath
        elif self.name in FeNNixPotentialImpl.KNOWN_MODELS:
            url = FeNNixPotentialImpl.KNOWN_MODELS[self.name]
            tmp_file = tempfile.NamedTemporaryFile(suffix=".fnx", delete=False)
            downloaded_model_path = tmp_file.name
            tmp_file.close()
            urllib.request.urlretrieve(url, downloaded_model_path)
            modelPath = downloaded_model_path
        else:
            raise ValueError("modelPath must be provided for custom FeNNix models")

        try:
            model = fennol.FENNIX.load(modelPath, **args)
        finally:
            if downloaded_model_path is not None:
                os.unlink(downloaded_model_path)
        if energy_terms is not None:
            model.set_energy_terms(energy_terms)
        energyScale = (
            unit.hartree / model.Ha_to_model_energy * unit.AVOGADRO_CONSTANT_NA
        ).value_in_unit(unit.kilojoule_per_mole)
        forceScale = (energyScale / unit.angstrom).value_in_unit(unit.nanometer**-1)

        # Get the atoms that should be included.
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            atoms = list(atoms)
            includedAtoms = [includedAtoms[i] for i in atoms]
        species = jnp.array(
            [atom.element.atomic_number for atom in includedAtoms], dtype=jnp.int32
        )
        indices = None if atoms is None else jnp.array(atoms, dtype=jnp.int32)
        numSystemAtoms = system.getNumParticles()

        inputs = dict(
            species=species,
            natoms=jnp.array([species.size], dtype=jnp.int32),
            batch_index=jnp.zeros(species.size, dtype=jnp.int32),
            total_charge=charge,
        )

        periodic = (
            topology.getPeriodicBoxVectors() is not None or system.usesPeriodicBoundaryConditions()
        )
        forcePeriodic = periodic and periodic_neighborlist
        dtype = np.float32

        # Prepare static inputs and initialize preprocessing state on CPU
        staticInputs = {
            key: np.asarray(value) if isinstance(value, jax.Array) else value
            for key, value in inputs.items()
        }
        preprocInputs = {
            **staticInputs,
            "coordinates": np.zeros((species.size, 3), dtype=dtype),
        }
        if forcePeriodic:
            box_vectors = topology.getPeriodicBoxVectors()
            if box_vectors is None:
                box_vectors = system.getDefaultPeriodicBoxVectors()
            cells_ang = np.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                dtype=dtype,
            ).reshape(1, 3, 3)
            preprocInputs["cells"] = cells_ang
            preprocInputs["reciprocal_cells"] = np.linalg.inv(cells_ang)
        preprocState, _ = model.preprocessing.init_with_output(preprocInputs)

        energy_fn = partial(
            _energyFeNNix,
            model=model,
            static_inputs=staticInputs,
            preproc_state=preprocState,
            pbc=forcePeriodic,
            energy_scale=energyScale,
        )
        energy_and_forces_fn = partial(
            _energyAndForcesFeNNix,
            model=model,
            static_inputs=staticInputs,
            preproc_state=preprocState,
            pbc=forcePeriodic,
            energy_scale=energyScale,
            force_scale=forceScale,
        )

        def _energy_kjmol(positions_nm, box_vectors_nm=None):
            selected_positions = positions_nm if indices is None else positions_nm[indices]
            return energy_fn((selected_positions, box_vectors_nm))

        def _energy_and_forces_kjmol(positions_nm, box_vectors_nm=None):
            selected_positions = positions_nm if indices is None else positions_nm[indices]
            energy, forces = energy_and_forces_fn((selected_positions, box_vectors_nm))
            if indices is not None:
                forces = jnp.zeros_like(positions_nm).at[indices].set(forces)
            return energy, forces

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


for model_name in FeNNixPotentialImpl.KNOWN_MODELS:
    MLPotential.registerImplFactory(model_name, FeNNixPotentialImplFactory())
MLPotential.registerImplFactory("fennix-jax", FeNNixPotentialImplFactory())

__all__ = [
    "MLPotential",
    "FeNNixPotentialImplFactory",
    "FeNNixPotentialImpl",
]


def _preprocessFeNNix(
    state,
    model,
    static_inputs,
    preproc_state,
    pbc: bool,
):
    positions_nm, box_vectors_nm = state
    preproc_in = {
        **static_inputs,
        "coordinates": positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom),
    }
    if pbc and box_vectors_nm is not None:
        cells_ang = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
        cells_ang = cells_ang.reshape(1, 3, 3)
        preproc_in["cells"] = cells_ang
        preproc_in["reciprocal_cells"] = _inverse_3x3(cells_ang)
    return model.preprocessing.process(preproc_state, preproc_in)


def _energyFeNNix(
    state,
    model,
    static_inputs,
    preproc_state,
    pbc: bool,
    energy_scale: float,
):
    """Evaluate FeNNix energy in kJ/mol from OpenMM positions in nm."""
    processed = _preprocessFeNNix(
        state,
        model=model,
        static_inputs=static_inputs,
        preproc_state=preproc_state,
        pbc=pbc,
    )
    energy, _ = model._total_energy(model.variables, processed)
    return energy.squeeze() * energy_scale


def _energyAndForcesFeNNix(
    state,
    model,
    static_inputs,
    preproc_state,
    pbc: bool,
    energy_scale: float,
    force_scale: float,
):
    """Evaluate FeNNix energy and forces in OpenMM units from positions in nm."""
    processed = _preprocessFeNNix(
        state,
        model=model,
        static_inputs=static_inputs,
        preproc_state=preproc_state,
        pbc=pbc,
    )
    energy, forces, _ = model._energy_and_forces(model.variables, processed)
    return energy.squeeze() * energy_scale, forces * force_scale


def _inverse_3x3(matrix):
    """Invert one or more 3x3 matrices without lowering to a solver FFI call."""
    a = matrix[..., 0, 0]
    b = matrix[..., 0, 1]
    c = matrix[..., 0, 2]
    d = matrix[..., 1, 0]
    e = matrix[..., 1, 1]
    f = matrix[..., 1, 2]
    g = matrix[..., 2, 0]
    h = matrix[..., 2, 1]
    i = matrix[..., 2, 2]

    cofactors = jnp.stack(
        [
            jnp.stack([e * i - f * h, c * h - b * i, b * f - c * e], axis=-1),
            jnp.stack([f * g - d * i, a * i - c * g, c * d - a * f], axis=-1),
            jnp.stack([d * h - e * g, b * g - a * h, a * e - b * d], axis=-1),
        ],
        axis=-2,
    )
    determinant = a * cofactors[..., 0, 0] + b * cofactors[..., 1, 0] + c * cofactors[..., 2, 0]
    return cofactors / determinant[..., None, None]
