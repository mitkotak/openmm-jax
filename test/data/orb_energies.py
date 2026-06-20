# /// script
# dependencies = ["ase", "numpy", "openmm", "orb-models"]
# ///
# This script computes reference energies for the ORB conservative OMOL model.

from pathlib import Path

import ase.io
from openmm import unit
from orb_models.forcefield import pretrained as orb

try:
    from orb_models.forcefield.inference.calculator import ORBCalculator
except ModuleNotFoundError:
    from orb_models.forcefield.calculator import ORBCalculator

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
MODEL = "orb-v3-conservative-omol"


def make_calculator():
    pretrained = orb.ORB_PRETRAINED_MODELS[MODEL](precision="float32-highest")
    if isinstance(pretrained, tuple):
        orbff, atoms_adapter = pretrained
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter)
    return ORBCalculator(pretrained)


results = {}

atoms = ase.io.read(DATA_DIR / "toluene" / "toluene.pdb")
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.calc = make_calculator()
results[f"toluene/{MODEL}"] = atoms.get_potential_energy()

atoms.info["charge"] = -1
atoms.info["spin"] = 3
atoms.calc = make_calculator()
results[f"toluene/{MODEL}/override-charge-spin"] = atoms.get_potential_energy()

atoms = ase.io.read(DATA_DIR / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb")
atoms.info["charge"] = 0
atoms.info["spin"] = 1
atoms.calc = make_calculator()
results[f"alanine-dipeptide-explicit/{MODEL}"] = atoms.get_potential_energy()

for key in results:
    print(f"{key}: {results[key] * EV_TO_KJMOL}")
