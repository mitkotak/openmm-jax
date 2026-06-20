import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

equinox = pytest.importorskip("equinox", reason="equinox is not installed")

import models.aceffpotential  # noqa: E402,F401
from models.aceff import ACEFF_MODEL_PATHS  # noqa: E402

cuda_platform = mm.Platform.getPlatformByName("CUDA")
pytestmark = pytest.mark.skipif(cuda_platform is None, reason="CUDA platform is not available")

ENERGIES = {
    "toluene": {
        "aceff-jax-1.1": -6976.571458405198,
        "aceff-jax-2.0": -362.4613670809051,
    },
    "alanine-dipeptide-explicit": {
        "aceff-jax-1.1": -732686.0090431497,
        "aceff-jax-2.0": -72195.52665709531,
    },
}
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
models = [
    model
    for model in ["aceff-jax-1.1", "aceff-jax-2.0"]
    if ACEFF_MODEL_PATHS[model].is_file()
]


class TestAceFF:
    @pytest.mark.parametrize("model", models)
    def testCreatePureMLSystem(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        positions_original = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(
            pdb.topology,
            returnEnergyType="energy",
            preprocessing_positions=positions_original,
        )
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        energy_ref = ENERGIES["toluene"][model]
        for i in range(10):
            context.setPositions(positions_original + i * 0.5 * unit.nanometers)
            energy_ml = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
            assert np.isclose(energy_ref, energy_ml, rtol=1e-4)

    def testPeriodicSystem(self):
        model = "aceff-jax-2.0"
        if not ACEFF_MODEL_PATHS[model].is_file():
            pytest.skip(f"{model} model file is not available")
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))  # noqa: E501
        potential = MLPotential(model)
        positions_original = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(
            pdb.topology,
            returnEnergyType="energy",
            preprocessing_positions=positions_original,
        )
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        energy_ref = ENERGIES["alanine-dipeptide-explicit"][model]
        for i in range(3):
            positions = positions_original + i * 0.9 * unit.nanometers
            context.setPositions(positions)
            energy_ml = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
            assert np.isclose(energy_ref, energy_ml, rtol=1e-4)

    @pytest.mark.parametrize("model", models)
    def testCreateMixedSystem(self, model):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))  # noqa: E501
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))  # noqa: E501
        ml_atoms = list(range(15))
        mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential(model)
        mixed_system = potential.createMixedSystem(
            prmtop.topology,
            mm_system,
            ml_atoms,
            interpolate=False,
            preprocessing_positions=inpcrd.positions,
        )
        interp_system = potential.createMixedSystem(
            prmtop.topology,
            mm_system,
            ml_atoms,
            interpolate=True,
            preprocessing_positions=inpcrd.positions,
        )
        mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        mixed_context = mm.Context(mixed_system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        interp_context = mm.Context(interp_system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        mm_context.setPositions(inpcrd.positions)
        mixed_context.setPositions(inpcrd.positions)
        interp_context.setPositions(inpcrd.positions)
        mm_energy = mm_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        mixed_energy = mixed_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        interp_energy_1 = interp_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        interp_context.setParameter("lambda_interpolate", 0)
        interp_energy_2 = interp_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        assert np.isclose(mixed_energy, interp_energy_1, rtol=1e-5)
        assert np.isclose(mm_energy, interp_energy_2, rtol=1e-5)
