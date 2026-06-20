import os

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
from openmmml import MLPotential

equinox = pytest.importorskip("equinox", reason="equinox is not installed")

import models.orbpotential  # noqa: E402,F401
from models.orb import ORB_MODEL_PATHS  # noqa: E402

cuda_platform = mm.Platform.getPlatformByName("CUDA")
model = "orb-jax-v3-conservative-omol"
ENERGIES = {
    "toluene": {
        model: -712903.547903221,
        f"{model}/override-charge-spin": -712892.4765882556,
    },
    "alanine-dipeptide-explicit": {
        model: -151632910.67503712,
    },
}
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
pytestmark = [
    pytest.mark.skipif(cuda_platform is None, reason="CUDA platform is not available"),
    pytest.mark.skipif(not ORB_MODEL_PATHS[model].is_file(), reason=f"{model} checkpoint is not available"),
]


class TestORB:
    def testCreatePureMLSystem(self):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model, charge=0, multiplicity=1)
        positions_original = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(pdb.topology, preprocessing_positions=positions_original)
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        context.setPositions(positions_original)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(ENERGIES["toluene"][model], energy, rtol=5e-5)

    def testOverrideChargeSpin(self):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential(model, charge=-1, multiplicity=3)
        positions = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(pdb.topology, preprocessing_positions=positions)
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        energy_ref = ENERGIES["toluene"][f"{model}/override-charge-spin"]
        assert np.isclose(energy_ref, energy, rtol=5e-5)

    @pytest.mark.skip(reason="ORB periodic OpenMM export exceeds available CUDA memory")
    def testPeriodicSystem(self):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential(model, charge=0, multiplicity=1)
        positions_original = pdb.getPositions(asNumpy=True)
        system = potential.createSystem(pdb.topology, preprocessing_positions=positions_original)
        context = mm.Context(system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        energy_ref = ENERGIES["alanine-dipeptide-explicit"][model]
        for i in range(3):
            context.setPositions(positions_original + i * 0.9 * unit.nanometers)
            energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            assert np.isclose(energy_ref, energy, rtol=5e-5)

    def testCreateMixedSystem(self):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        ml_atoms = list(range(15))
        mm_system = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential(model, charge=0, multiplicity=1)
        mixed_system = potential.createMixedSystem(prmtop.topology, mm_system, ml_atoms, interpolate=False, preprocessing_positions=inpcrd.positions)
        interp_system = potential.createMixedSystem(prmtop.topology, mm_system, ml_atoms, interpolate=True, preprocessing_positions=inpcrd.positions)
        mixed_context = mm.Context(mixed_system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        interp_context = mm.Context(interp_system, mm.VerletIntegrator(0.001), cuda_platform, {"DeviceIndex": "0", "Precision": "mixed"})
        mixed_context.setPositions(inpcrd.positions)
        interp_context.setPositions(inpcrd.positions)
        mixed_energy = mixed_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interp_energy = interp_context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(mixed_energy, interp_energy, rtol=1e-5)
