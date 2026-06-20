# /// script
# dependencies = ["aimnet", "ase", "numpy", "openmm"]
# ///
# This script computes reference energies for the AIMNet2 model.

from pathlib import Path

import ase.io
from aimnet.calculators import AIMNet2ASE
from openmm import unit

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)

results = {}

atoms = ase.io.read(DATA_DIR / "toluene" / "toluene.pdb")
atoms.calc = AIMNet2ASE("aimnet2", charge=0)
results["toluene/aimnet2-jax"] = atoms.get_potential_energy()

atoms = ase.io.read(DATA_DIR / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb")
atoms.calc = AIMNet2ASE("aimnet2", charge=0)
results["alanine-dipeptide-explicit/aimnet2-jax"] = atoms.get_potential_energy()

for key in results:
    print(f"{key}: {results[key] * EV_TO_KJMOL}")
