import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

fennol = pytest.importorskip("fennol", reason="FeNNol is not installed")

import models.fennixpotential  # noqa: E402,F401


cuda_platform = mm.Platform.getPlatformByName("CUDA")
pytestmark = pytest.mark.skipif(cuda_platform is None, reason="CUDA platform is not available")

ENERGIES = {
    "toluene": {
        "fennix-bio1-small": -5.200859421605564,
        "fennix-bio1-medium": -2.3028696986989523,
    },
    "methanol-ions": {
        "fennix-bio1-small": -599.6015619222414,
        "fennix-bio1-medium": -1109.2088074881058,
        "fennix-bio1-small-finetune-ions": -560.4959154537397,
        "fennix-bio1-medium-finetune-ions": -1068.5316655421075,
    },
    "alanine-dipeptide-explicit": {
        "fennix-bio1-small": -68462.99055925063,
    },
}
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

class TestFeNNix:
    @pytest.mark.parametrize("model", ["fennix-bio1-small", "fennix-bio1-medium"])
    def testCreatePureMLSystem(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model)
        positions = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(
            pdb.topology,
            precision="double",
            preprocessing_positions=positions,
        )
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "double"})  # noqa: E501
        context.setPositions(positions)
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        # Reference energies are calculated with FENNIXCalculator
        refEnergy = ENERGIES["toluene"]
        assert np.isclose(refEnergy[model], energyML, rtol=1e-5)

    @pytest.mark.parametrize(
        "model",
        [
            "fennix-bio1-small",
            "fennix-bio1-medium",
            "fennix-bio1-small-finetune-ions",
            "fennix-bio1-medium-finetune-ions",
        ],
    )
    def testChargedSystem(self, model):
        pdb = app.PDBFile(os.path.join(test_data_dir, "methanol-ions", "methanol-ions.pdb"))
        potential = MLPotential(model)
        positions = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(
            pdb.topology,
            charge=1,
            precision="double",
            preprocessing_positions=positions,
        )
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "double"})  # noqa: E501
        context.setPositions(positions)
        energyML = context.getState(energy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
        # Reference energies are calculated with FENNIXCalculator
        refEnergy = ENERGIES["methanol-ions"]
        assert np.isclose(refEnergy[model], energyML, rtol=1e-5)

    def testPeriodicSystem(self):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))  # noqa: E501
        potential = MLPotential("fennix-bio1-small")
        positionsOriginal = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(
            pdb.topology,
            precision="double",
            preprocessing_positions=positionsOriginal,
        )
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "double"})  # noqa: E501
        energyRef = ENERGIES["alanine-dipeptide-explicit"]["fennix-bio1-small"]  # Calculated with FENNIXCalculator  # noqa: E501
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers
            context.setPositions(positions)
            energyML = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)  # noqa: E501
            assert np.isclose(energyRef, energyML, rtol=1e-5)

    def testCreateMixedSystem(self):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))  # noqa: E501
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))  # noqa: E501
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential("fennix-bio1-small")
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
