# /// script
# dependencies = ["ase", "mace-torch", "numpy", "openmm"]
# ///
# This script computes reference energies for the MACE JAX foundation models.

from pathlib import Path

import ase.io
from mace.calculators.foundations_models import mace_off
from openmm import unit

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)

results = {}

atoms = ase.io.read(DATA_DIR / "toluene" / "toluene.pdb")
for model_name, checkpoint in [
    ("mace-jax-off-s-23", "small"),
    (
        "mace-jax-off-m-24",
        "https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true",
    ),
]:
    atoms.calc = mace_off(checkpoint, device="cpu")
    results[f"toluene/{model_name}"] = atoms.get_potential_energy()

atoms = ase.io.read(DATA_DIR / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb")
atoms.calc = mace_off("small", device="cpu")
results["alanine-dipeptide-explicit/mace-jax-off-s-23"] = atoms.get_potential_energy()

for key in results:
    print(f"{key}: {results[key] * EV_TO_KJMOL}")
