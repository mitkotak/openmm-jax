"""
OpenMM PythonForce integration for the FeNNix JAX model.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Evan Pretti
Contributors:

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from typing import Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import openmm
import openmm.app as app
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .fennixpotential import (
    FeNNixPotentialImpl as JaxFeNNixPotentialImpl,
)


class FeNNixPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(
        self,
        name: str,
        modelPath: str | None = None,
        **args,
    ) -> MLPotentialImpl:
        return FeNNixPotentialImpl(_base_model_name(name), modelPath)


class FeNNixPotentialImpl(MLPotentialImpl):
    KNOWN_MODELS = {
        name: url
        for name, url in JaxFeNNixPotentialImpl.KNOWN_MODELS.items()
        if not name.endswith("-jax")
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
        precision: str | None = None,
        gpu_preprocessing: bool = True,
        use_float64: bool = False,
        matmul_prec: Optional[str] = "highest",
        energy_terms: Optional[Sequence[str]] = None,
        periodic_neighborlist: bool = True,
        minimum_image: bool = True,
        preprocessing_positions=None,
        preprocessing_positions_unit=unit.nanometer,
        **args,
    ) -> None:
        import fennol

        del gpu_preprocessing
        if precision is not None:
            if precision == "single":
                use_float64 = False
            elif precision == "double":
                use_float64 = True
            else:
                raise ValueError(
                    f"Invalid precision {precision!r} (expected 'single' or 'double')"
                )

        with jax.enable_x64(use_float64):
            if matmul_prec is not None:
                jax.config.update("jax_default_matmul_precision", matmul_prec)

            downloaded_model_path = None
            if self.modelPath is not None:
                model_path = self.modelPath
            elif self.name in self.KNOWN_MODELS:
                tmp_file = tempfile.NamedTemporaryFile(suffix=".fnx", delete=False)
                downloaded_model_path = tmp_file.name
                tmp_file.close()
                urllib.request.urlretrieve(self.KNOWN_MODELS[self.name], downloaded_model_path)
                model_path = downloaded_model_path
            else:
                raise ValueError("modelPath must be provided for custom FeNNix models")

            try:
                model = fennol.FENNIX.load(model_path, **args)
            finally:
                if downloaded_model_path is not None:
                    os.unlink(downloaded_model_path)
            if energy_terms is not None:
                model.set_energy_terms(energy_terms)

            energy_scale = (
                unit.hartree / model.Ha_to_model_energy * unit.AVOGADRO_CONSTANT_NA
            ).value_in_unit(unit.kilojoules_per_mole)
            force_scale = (energy_scale / unit.angstrom).value_in_unit(unit.nanometer**-1)

            included_atoms = list(topology.atoms())
            atom_indices_np = None
            if atoms is not None:
                atoms = list(atoms)
                included_atoms = [included_atoms[i] for i in atoms]
                atom_indices_np = np.asarray(atoms, dtype=np.int32)

            species = jnp.array(
                [atom.element.atomic_number for atom in included_atoms],
                dtype=jnp.int32,
            )
            static_inputs = {
                "species": species,
                "natoms": jnp.array([species.size], dtype=jnp.int32),
                "batch_index": jnp.zeros(species.size, dtype=jnp.int32),
                "total_charge": charge,
            }

            periodic = (
                topology.getPeriodicBoxVectors() is not None
                or system.usesPeriodicBoundaryConditions()
            )
            force_periodic = periodic and periodic_neighborlist
            if force_periodic and minimum_image:
                static_inputs["flags"] = {"minimum_image": None}

            force = openmm.PythonForce(
                _ComputeFeNNixPythonForce(
                    model=model,
                    static_inputs=static_inputs,
                    energy_scale=energy_scale,
                    force_scale=force_scale,
                    indices=atom_indices_np,
                    periodic=force_periodic,
                    use_float64=use_float64,
                )
            )
            force.setForceGroup(forceGroup)
            force.setUsesPeriodicBoundaryConditions(force_periodic)
            system.addForce(force)


class _ComputeFeNNixPythonForce:
    def __init__(
        self,
        *,
        model,
        static_inputs,
        energy_scale: float,
        force_scale: float,
        indices,
        periodic: bool,
        use_float64: bool,
    ) -> None:
        self.model = model
        self.static_inputs = static_inputs
        self.energy_scale = energy_scale
        self.force_scale = force_scale
        self.periodic = bool(periodic)
        self.use_float64 = bool(use_float64)
        self.coordinate_dtype = jnp.float64 if use_float64 else jnp.float32
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int32)
        self.jax_indices = (
            None if self.indices is None else jnp.asarray(self.indices, dtype=jnp.int32)
        )

    def _energy_and_forces_kjmol(self, positions_nm, box_vectors_nm=None):
        selected_positions = (
            positions_nm if self.jax_indices is None else positions_nm[self.jax_indices]
        )
        coordinates = (
            selected_positions * unit.nanometer.conversion_factor_to(unit.angstrom)
        ).astype(self.coordinate_dtype)
        inputs = {"coordinates": coordinates, **self.static_inputs}
        if self.periodic and box_vectors_nm is not None:
            inputs["cells"] = (
                box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
            ).reshape(1, 3, 3).astype(self.coordinate_dtype)
        model_outputs = self.model.energy_and_forces(
            gpu_preprocessing=True,
            **inputs,
        )
        energy, forces = model_outputs[:2]
        energy = energy.squeeze() * self.energy_scale
        forces = forces * self.force_scale
        if self.jax_indices is not None:
            forces = jnp.zeros_like(positions_nm).at[self.jax_indices].set(forces)
        return energy, forces

    def __call__(self, state):
        with jax.enable_x64(self.use_float64):
            positions_nm = jnp.asarray(
                state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
                dtype=self.coordinate_dtype,
            )
            box_vectors_nm = None
            if self.periodic:
                box_vectors_nm = jnp.asarray(
                    state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
                    dtype=self.coordinate_dtype,
                )
            energy, forces = self._energy_and_forces_kjmol(
                positions_nm,
                box_vectors_nm,
            )
        return float(jax.device_get(energy)), np.asarray(jax.device_get(forces))


def _base_model_name(name: str) -> str:
    for suffix in ("-jax-python", "-pythonforce", "-python"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


for model_name in FeNNixPotentialImpl.KNOWN_MODELS:
    MLPotential.registerImplFactory(
        f"{model_name}-jax-python",
        FeNNixPotentialImplFactory(),
    )
    MLPotential.registerImplFactory(
        f"{model_name}-python",
        FeNNixPotentialImplFactory(),
    )
    MLPotential.registerImplFactory(
        f"{model_name}-pythonforce",
        FeNNixPotentialImplFactory(),
    )
MLPotential.registerImplFactory("fennix-python", FeNNixPotentialImplFactory())
MLPotential.registerImplFactory("fennix-pythonforce", FeNNixPotentialImplFactory())

__all__ = [
    "MLPotential",
    "FeNNixPotentialImplFactory",
    "FeNNixPotentialImpl",
]
