import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

equinox = pytest.importorskip("equinox", reason="equinox is not installed")

import models.macepotential  # noqa: E402,F401
from models.mace import MACE_MODEL_PATHS  # noqa: E402


cuda_platform = mm.Platform.getPlatformByName("CUDA")
pytestmark = pytest.mark.skipif(cuda_platform is None, reason="CUDA platform is not available")

ENERGIES = {
    "toluene": {
        "mace-jax-off-s-23": -713468.6327560507,
        "mace-jax-off-m-24": -713467.9394350434,
    },
    "alanine-dipeptide-explicit": {
        "mace-jax-off-s-23": -151723354.26015,
    },
}
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
models = [
    model
    for model in ["mace-jax-off-s-23", "mace-jax-off-m-24"]
    if MACE_MODEL_PATHS[model].is_file()
]

class TestMACE:
    @pytest.mark.parametrize("model", models)
    def testCreatePureMLSystem(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        positions = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(
            pdb.topology,
            returnEnergyType="energy",
            preprocessing_positions=positions,
        )
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        context.setPositions(positions)
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        # Reference energies are calculated with MACECalculator
        refEnergy = ENERGIES["toluene"]
        assert np.isclose(refEnergy[model], energyML, rtol=1e-6)

    def testPeriodicSystem(self):
        model = "mace-jax-off-s-23"
        if model not in models:
            pytest.skip(f"{model} checkpoint is not available")
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))  # noqa: E501
        potential = MLPotential(model)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(
            pdb.topology,
            returnEnergyType="energy",
            preprocessing_positions=positionsOriginal,
        )
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        energyRef = ENERGIES["alanine-dipeptide-explicit"][model]  # Calculated with MACECalculator
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers
            context.setPositions(positions)
            energyML = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
            assert np.isclose(energyRef, energyML, rtol=1e-5)

    def testCreateMixedSystem(self):
        model = "mace-jax-off-s-23"
        if model not in models:
            pytest.skip(f"{model} checkpoint is not available")
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))  # noqa: E501
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))  # noqa: E501
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential(model)
        mixedSystem = potential.createMixedSystem(
            prmtop.topology,
            mmSystem,
            mlAtoms,
            interpolate=False,
            preprocessing_positions=inpcrd.positions,
        )
        interpSystem = potential.createMixedSystem(
            prmtop.topology,
            mmSystem,
            mlAtoms,
            interpolate=True,
            preprocessing_positions=inpcrd.positions,
        )
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})  # noqa: E501
        mmContext.setPositions(inpcrd.positions)
        mixedContext.setPositions(inpcrd.positions)
        interpContext.setPositions(inpcrd.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        interpContext.setParameter("lambda_interpolate", 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=1e-5)
        assert np.isclose(mmEnergy, interpEnergy2, rtol=1e-5)
