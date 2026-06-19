"""
fennixpotential.py: Support for FeNNix potentials.

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

from typing import Iterable
import openmm
from openmm import unit
from openmmml.mlpotential import MLPotential, MLPotentialImpl, MLPotentialImplFactory

class FeNNixPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates FeNNixPotentialImpl objects."""

    def createImpl(self, name: str, modelPath: str | None = None, **args) -> MLPotentialImpl:
        return FeNNixPotentialImpl(name, modelPath)

class FeNNixPotentialImpl(MLPotentialImpl):
    """
    Implementation of FeNNix potentials for OpenMM.

    The FeNNol library is used to load FeNNix models and evaluate energies and
    forces.  This implementation can use local files for models or automatically
    download them from the FeNNol-PMC repository.

    To use one of the pre-trained FeNNix models, specify it by name.  For example:

    >>> potential = MLPotential('fennix-bio1-small')

    Other available models include 'fennix-bio1-medium', 'fennix-bio1-small-finetune-ions', and 'fennix-bio1-medium-finetune-ions'.

    To use a local `.fnx` file, specify 'fennix' as the model name, and supply the `modelPath` argument, *e.g.*,

    >>> potential = MLPotential('fennix', modelPath='custom_fennix_model.fnx')
    """

    KNOWN_MODELS = {
        "fennix-bio1-small": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0/fennix-bio1S.fnx", True),
        "fennix-bio1-medium": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0/fennix-bio1M.fnx", True),
        "fennix-bio1-small-finetune-ions": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1S-finetuneIons.fnx", True),
        "fennix-bio1-medium-finetune-ions": ("https://github.com/FeNNol-tools/FeNNol-PMC/raw/refs/heads/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1M-finetuneIons.fnx", True),
    }

    def __init__(self, name: str, modelPath: str | None) -> None:
        """
        Initialize the `FeNNixPotentialImpl`.

        Parameters
        ----------
        name : str
            The name of the model.  Options include the pre-trained models
            'fennix-bio1-small', 'fennix-bio1-medium', 'fennix-bio1-small-finetune-ions', and
            'fennix-bio1-medium-finetune-ions', or 'fennix' to load a local model file.
        modelPath : str, optional
            A path to the model file to load.
        """

        self.name = name
        self.modelPath = modelPath

    def addForces(self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        charge: int = 0,
        precision: str = "single",
        **args
    ) -> None:

        try:
            import fennol
        except ImportError:
            raise ImportError("Failed to import FeNNol: for installation instructions, visit https://github.com/FeNNol-tools/FeNNol")
        import jax
        import jax.numpy as jnp
        import numpy as np

        # Check precision argument.
        if precision == "single":
            useDouble = False
        elif precision == "double":
            useDouble = True
        else:
            raise ValueError(f"Invalid precision {precision} (expected single or double)")

        # Download or look up the model file to use.
        baseName = _base_model_name(self.name)
        if baseName in FeNNixPotentialImpl.KNOWN_MODELS:
            url, warn = FeNNixPotentialImpl.KNOWN_MODELS[baseName]
            if warn:
                import logging
                logging.warning(f"The model {baseName} is distributed under the restrictive ASL license.  Commercial use is not permitted.")
            modelPath = self._downloadOrFindFile(f"{baseName}.fnx", url)
        elif baseName == "fennix":
            if self.modelPath is None:
                raise ValueError("No modelPath provided for local FeNNix model.")
            modelPath = self.modelPath
        else:
            supported_options = ", ".join(list(FeNNixPotentialImpl.KNOWN_MODELS) + ["fennix"])
            raise ValueError(f"Unsupported FeNNix model: {baseName} (options are {supported_options})")

        # Load the model.
        model = fennol.FENNIX.load(modelPath, **args)
        energyScale = (unit.hartree / model.Ha_to_model_energy * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)
        forceScale = (energyScale / unit.angstrom).value_in_unit(unit.nanometer ** -1)

        # Get the atoms that should be included.
        includedAtoms = list(topology.atoms())
        indices = None
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = np.array(atoms, dtype=int)

        # Prepare inputs to the model that remain constant from step to step.
        species = jnp.array([atom.element.atomic_number for atom in includedAtoms], dtype=jnp.int32)
        inputs = dict(
            species=species,
            natoms=jnp.array([species.size], dtype=jnp.int32),
            batch_index=jnp.zeros(species.size, dtype=jnp.int32),
            total_charge=charge,
        )

        periodic = (topology.getPeriodicBoxVectors() is not None) or system.usesPeriodicBoundaryConditions()
        dtype = np.float64 if useDouble else np.float32
        staticInputs = {
            key: np.asarray(value) if isinstance(value, jax.Array) else value
            for key, value in inputs.items()
        }
        preprocInputs = {
            **staticInputs,
            "coordinates": np.zeros((species.size, 3), dtype=dtype),
        }
        if periodic:
            boxVectors = topology.getPeriodicBoxVectors()
            if boxVectors is None:
                boxVectors = system.getDefaultPeriodicBoxVectors()
            cells = np.asarray(
                [vector.value_in_unit(unit.angstrom) for vector in boxVectors],
                dtype=dtype,
            ).reshape(1, 3, 3)
            preprocInputs["cells"] = cells
            preprocInputs["reciprocal_cells"] = np.linalg.inv(cells)
        preprocState, _ = model.preprocessing.init_with_output(preprocInputs)

        # Create the PythonForce and add it to the System.
        force = openmm.PythonForce(
            _ComputeFeNNix(
                model,
                energyScale,
                forceScale,
                indices,
                staticInputs,
                preprocState,
                periodic,
                useDouble,
            )
        )
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)

class _ComputeFeNNix:
    def __init__(
        self,
        model,
        energyScale,
        forceScale,
        indices,
        staticInputs,
        preprocState,
        periodic,
        useDouble,
    ):
        self.model = model
        self.energyScale = energyScale
        self.forceScale = forceScale
        self.indices = indices
        self.staticInputs = staticInputs
        self.preprocState = preprocState
        self.periodic = periodic
        self.useDouble = useDouble
        self._buildCompute()

    def _buildCompute(self):
        import jax
        import jax.numpy as jnp

        if self.periodic:
            def energyAndForces(positionsNm, boxVectorsNm):
                cellsAng = boxVectorsNm * unit.nanometer.conversion_factor_to(unit.angstrom)
                cellsAng = cellsAng.reshape(1, 3, 3)
                preprocIn = {
                    **self.staticInputs,
                    "coordinates": positionsNm * unit.nanometer.conversion_factor_to(unit.angstrom),
                    "cells": cellsAng,
                    "reciprocal_cells": jnp.linalg.inv(cellsAng),
                }
                processed = self.model.preprocessing.process(self.preprocState, preprocIn)
                energy, forces, _ = self.model._energy_and_forces(self.model.variables, processed)
                return energy.squeeze() * self.energyScale, forces * self.forceScale
        else:
            def energyAndForces(positionsNm):
                preprocIn = {
                    **self.staticInputs,
                    "coordinates": positionsNm * unit.nanometer.conversion_factor_to(unit.angstrom),
                }
                processed = self.model.preprocessing.process(self.preprocState, preprocIn)
                energy, forces, _ = self.model._energy_and_forces(self.model.variables, processed)
                return energy.squeeze() * self.energyScale, forces * self.forceScale

        self._compute = jax.jit(energyAndForces)

    def __call__(self, state):
        import jax
        import numpy as np

        # Load coordinates and box vectors from the state.
        dtype = np.float64 if self.useDouble else np.float32
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer).astype(dtype)
        numAtoms = positions.shape[0]
        if self.indices is not None:
            positions = positions[self.indices]
        if self.periodic:
            boxVectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer).astype(dtype)

        # Invoke the model to get the energy and forces.
        with jax.enable_x64(self.useDouble):
            if self.periodic:
                jaxEnergy, jaxForces = self._compute(positions, boxVectors)
            else:
                jaxEnergy, jaxForces = self._compute(positions)
            energy = jaxEnergy.item()
            if self.indices is None:
                forces = np.asarray(jaxForces)
            else:
                forces = np.zeros((numAtoms, 3), dtype=jaxForces.dtype)
                forces[self.indices] = jaxForces

        return energy, forces

    def __getstate__(self):
        return (
            self.model.to_dict(),
            self.energyScale,
            self.forceScale,
            self.indices,
            self.staticInputs,
            self.preprocState,
            self.periodic,
            self.useDouble,
        )

    def __setstate__(self, pickle_state):
        import fennol
        (
            model_dict,
            self.energyScale,
            self.forceScale,
            self.indices,
            self.staticInputs,
            self.preprocState,
            self.periodic,
            self.useDouble,
        ) = pickle_state
        self.model = fennol.FENNIX(**model_dict)
        self._buildCompute()


def _base_model_name(name: str) -> str:
    for suffix in ("-pythonforce", "-python"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


for modelName in FeNNixPotentialImpl.KNOWN_MODELS:
    MLPotential.registerImplFactory(f"{modelName}-pythonforce", FeNNixPotentialImplFactory())
    MLPotential.registerImplFactory(f"{modelName}-python", FeNNixPotentialImplFactory())
MLPotential.registerImplFactory("fennix-pythonforce", FeNNixPotentialImplFactory())
MLPotential.registerImplFactory("fennix-python", FeNNixPotentialImplFactory())
