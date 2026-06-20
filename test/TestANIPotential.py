import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

equinox = pytest.importorskip("equinox", reason="equinox is not installed")

import models.anipotential  # noqa: E402,F401
from models.ani import ANI2X_MODEL_PATHS  # noqa: E402


cuda_platform = mm.Platform.getPlatformByName("CUDA")
pytestmark = pytest.mark.skipif(cuda_platform is None, reason="CUDA platform is not available")

test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ani_model_names = ("ani2x-model-0", "ani2x-jax-ensemble")
available_models = [
    model
    for model in ani_model_names
    if ANI2X_MODEL_PATHS[model].is_file()
]


def skip_if_model_unavailable(model):
    if model not in available_models:
        pytest.skip(f"{model} checkpoint is not available")


@pytest.mark.parametrize("model", ani_model_names)
class TestANIPotential:
    def testSimulate(self, model):
        skip_if_model_unavailable(model)
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        system = potential.createSystem(
            pdb.topology,
            preprocessing_positions=pdb.positions,
        )
        integrator = mm.LangevinIntegrator(300.0, 1.0, 0.001)
        context = mm.Context(
            system, integrator, cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"}
        )
        context.setPositions(pdb.positions)
        integrator.step(10)
        positions = (
            context.getState(positions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.nanometer)
        )
        assert np.all(np.isfinite(positions))

    def testCreateMixedSystem(self, model):
        skip_if_model_unavailable(model)
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))  # noqa: E501
        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        mmSystem = ff.createSystem(pdb.topology, nonbondedMethod=app.PME)
        potential = MLPotential(model)
        mlAtoms = [atom.index for atom in next(pdb.topology.chains()).atoms()]
        mixedSystem = potential.createMixedSystem(
            pdb.topology,
            mmSystem,
            mlAtoms,
            interpolate=False,
            preprocessing_positions=pdb.positions,
        )
        interpSystem = potential.createMixedSystem(
            pdb.topology,
            mmSystem,
            mlAtoms,
            interpolate=True,
            preprocessing_positions=pdb.positions,
        )
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        mmContext.setPositions(pdb.positions)
        mixedContext.setPositions(pdb.positions)
        interpContext.setPositions(pdb.positions)
        mmState = mmContext.getState(energy=True, forces=True)
        mixedState = mixedContext.getState(energy=True, forces=True)
        interpState1 = interpContext.getState(energy=True, forces=True)
        interpContext.setParameter("lambda_interpolate", 0)
        interpState2 = interpContext.getState(energy=True, forces=True)
        assert np.isclose(
            mixedState.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            interpState1.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            rtol=1e-5,
        )
        assert np.isclose(
            mmState.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            interpState2.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole),
            rtol=1e-5,
        )
        assert np.allclose(
            mixedState.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            interpState1.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            rtol=1e-3,
            atol=1e-4,
        )
        assert np.allclose(
            mmState.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            interpState2.getForces().value_in_unit(unit.kilojoules_per_mole / unit.nanometer),
            rtol=1e-3,
            atol=1e-4,
        )
