# This is part of the OpenMM molecular simulation toolkit originating from
# Simbios, the NIH National Center for Physics-Based Simulation of
# Biological Structures at Stanford, funded under the NIH Roadmap for
# Medical Research, grant U54 GM072970. See https://simtk.org.
#
# Portions copyright (c) 2026 Stanford University and the Authors.
# Authors: Evan Pretti
# Contributors:
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

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
from openmmjax import PythonJaxForce
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

from .fennixpotential import (
    FeNNixPotentialImpl as JaxFeNNixPotentialImpl,
)
from .fennixpotential import (
    _configured_preprocessing_state,
    _initial_preprocessing_coordinates_angstrom,
    _inverse_3x3,
)


class FeNNixPythonJaxForcePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(
        self,
        name: str,
        modelPath: str | None = None,
        **args,
    ) -> MLPotentialImpl:
        if name.endswith("-pythonjaxforce"):
            name = name[: -len("-pythonjaxforce")]
        return FeNNixPythonJaxForcePotentialImpl(name, modelPath)


class FeNNixPythonJaxForcePotentialImpl(MLPotentialImpl):
    KNOWN_MODELS = dict(JaxFeNNixPotentialImpl.KNOWN_MODELS)

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
        nblist_skin: float | None = 1.5,
        nblist_mult_size: float | None = 1.5,
        nblist_add_neigh: int | None = None,
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

            if atoms is not None:
                raise ValueError("FeNNix PythonJaxForce does not support atom subsets")
            included_atoms = list(topology.atoms())

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
            dtype = np.float64 if use_float64 else np.float32
            coordinate_dtype = jnp.float64 if use_float64 else jnp.float32
            num_system_atoms = system.getNumParticles() or topology.getNumAtoms()

            if preprocessing_positions is None:
                raise ValueError("FeNNix PythonJaxForce requires preprocessing_positions.")
            static_inputs_np = {
                key: np.asarray(value) if isinstance(value, jax.Array) else value
                for key, value in static_inputs.items()
            }
            preproc_coordinates = _initial_preprocessing_coordinates_angstrom(
                preprocessing_positions,
                dtype=dtype,
                indices=None,
                system_shape=(num_system_atoms, 3),
                fallback_shape=(species.size, 3),
                positions_unit=preprocessing_positions_unit,
            )
            preproc_inputs = {**static_inputs_np, "coordinates": preproc_coordinates}
            if force_periodic:
                box_vectors = topology.getPeriodicBoxVectors()
                if box_vectors is None:
                    box_vectors = system.getDefaultPeriodicBoxVectors()
                cells_ang = np.asarray(
                    [vector.value_in_unit(unit.angstrom) for vector in box_vectors],
                    dtype=dtype,
                ).reshape(1, 3, 3)
                preproc_inputs["cells"] = cells_ang
                preproc_inputs["reciprocal_cells"] = np.linalg.inv(cells_ang)
            preproc_state = _configured_preprocessing_state(
                model.preprocessing,
                nblist_skin=nblist_skin,
                nblist_mult_size=nblist_mult_size,
                nblist_add_neigh=nblist_add_neigh,
            )
            preproc_state, _ = model.preprocessing(
                preproc_state,
                preproc_inputs,
            )

            compute = _ComputeFeNNixPythonJaxForce(
                model=model,
                static_inputs=static_inputs_np,
                preproc_state=preproc_state,
                energy_scale=energy_scale,
                force_scale=force_scale,
                periodic=force_periodic,
                use_float64=use_float64,
                coordinate_dtype=coordinate_dtype,
            )
            force = PythonJaxForce(
                compute.energy,
                compute.forces,
                compute.energy_and_forces,
                {},
            )
            force.setForceGroup(forceGroup)
            force.setUsesPeriodicBoundaryConditions(force_periodic)
            system.addForce(force)


class _ComputeFeNNixPythonJaxForce:
    def __init__(
        self,
        *,
        model,
        static_inputs,
        preproc_state,
        energy_scale: float,
        force_scale: float,
        periodic: bool,
        use_float64: bool,
        coordinate_dtype,
    ) -> None:
        self.model = model
        self.static_inputs = static_inputs
        self.preproc_state = preproc_state
        self.energy_scale = energy_scale
        self.force_scale = force_scale
        self.periodic = bool(periodic)
        self.use_float64 = bool(use_float64)
        self.coordinate_dtype = coordinate_dtype
        self._energy = None
        self._forces = None
        self._energy_and_forces = None

    def _selected_positions_nm(self, positions_nm):
        return positions_nm

    def _preprocessing_inputs(self, selected_positions_nm, box_vectors_nm=None):
        inputs = {
            **self.static_inputs,
            "coordinates": (
                selected_positions_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
            ).astype(self.coordinate_dtype),
        }
        if self.periodic and box_vectors_nm is not None:
            cells_ang = box_vectors_nm * unit.nanometer.conversion_factor_to(unit.angstrom)
            cells_ang = cells_ang.reshape(1, 3, 3).astype(self.coordinate_dtype)
            inputs["cells"] = cells_ang
            inputs["reciprocal_cells"] = _inverse_3x3(cells_ang)
        return inputs

    def _preprocess(self, selected_positions_nm, box_vectors_nm=None):
        return self.model.preprocessing.process(
            self.preproc_state,
            self._preprocessing_inputs(selected_positions_nm, box_vectors_nm),
        )

    def _energy_and_forces_kjmol(self, positions_nm, box_vectors_nm=None):
        selected_positions = self._selected_positions_nm(positions_nm)
        processed = self._preprocess(selected_positions, box_vectors_nm)
        energy, forces, _ = self.model._energy_and_forces(self.model.variables, processed)
        energy = energy.squeeze() * self.energy_scale
        forces = forces * self.force_scale
        return energy, forces

    def _energy_kjmol(self, positions_nm, box_vectors_nm=None):
        energy, _ = self._energy_and_forces_kjmol(positions_nm, box_vectors_nm)
        return energy

    def _forces_kjmol(self, positions_nm, box_vectors_nm=None):
        _, forces = self._energy_and_forces_kjmol(positions_nm, box_vectors_nm)
        return forces

    def _typed_inputs(self, positions_nm, box_vectors_nm=None):
        positions_nm = jnp.asarray(positions_nm, dtype=self.coordinate_dtype)
        if box_vectors_nm is not None:
            box_vectors_nm = jnp.asarray(box_vectors_nm, dtype=self.coordinate_dtype)
        return positions_nm, box_vectors_nm

    def energy(self, positions_nm, box_vectors_nm=None):
        with jax.enable_x64(self.use_float64):
            positions_nm, box_vectors_nm = self._typed_inputs(positions_nm, box_vectors_nm)
            if self._energy is None:
                self._energy = jax.jit(self._energy_kjmol)
            return self._energy(positions_nm, box_vectors_nm)

    def forces(self, positions_nm, box_vectors_nm=None):
        with jax.enable_x64(self.use_float64):
            positions_nm, box_vectors_nm = self._typed_inputs(positions_nm, box_vectors_nm)
            if self._forces is None:
                self._forces = jax.jit(self._forces_kjmol)
            return self._forces(positions_nm, box_vectors_nm)

    def energy_and_forces(self, positions_nm, box_vectors_nm=None):
        with jax.enable_x64(self.use_float64):
            positions_nm, box_vectors_nm = self._typed_inputs(positions_nm, box_vectors_nm)
            if self._energy_and_forces is None:
                self._energy_and_forces = jax.jit(self._energy_and_forces_kjmol)
            energy, forces = self._energy_and_forces(
                positions_nm,
                box_vectors_nm,
            )
        return energy, forces

for model_name in FeNNixPythonJaxForcePotentialImpl.KNOWN_MODELS:
    MLPotential.registerImplFactory(
        f"{model_name}-pythonjaxforce",
        FeNNixPythonJaxForcePotentialImplFactory(),
    )

__all__ = [
    "MLPotential",
    "FeNNixPythonJaxForcePotentialImplFactory",
    "FeNNixPythonJaxForcePotentialImpl",
]
