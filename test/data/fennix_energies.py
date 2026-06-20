# /// script
# dependencies = ["ase", "fennol", "numpy", "openmm"]
# ///
# This script computes reference energies for the FeNNix models.

import os
import urllib.request
from pathlib import Path

import ase.io
from fennol.ase import FENNIXCalculator
from openmm import unit

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
MODEL_URLS = {
    "fennix-bio1-small": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0/fennix-bio1S.fnx",
    "fennix-bio1-medium": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0/fennix-bio1M.fnx",
    "fennix-bio1-small-finetune-ions": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1S-finetuneIons.fnx",
    "fennix-bio1-medium-finetune-ions": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1M-finetuneIons.fnx",
}


def model_path(model):
    cache_dir = Path(os.environ.get("FENNIX_MODEL_DIR", "/tmp/openmm-jax-fennix-models"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(MODEL_URLS[model]).name
    if not path.is_file():
        urllib.request.urlretrieve(MODEL_URLS[model], path)
    return str(path)


results = {}

atoms = ase.io.read(DATA_DIR / "toluene" / "toluene.pdb")
for model_name in ["fennix-bio1-small", "fennix-bio1-medium"]:
    atoms.calc = FENNIXCalculator(model_path(model_name), use_float64=True)
    results[f"toluene/{model_name}"] = atoms.get_potential_energy()

atoms = ase.io.read(DATA_DIR / "methanol-ions" / "methanol-ions.pdb")
atoms.set_initial_charges([0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1])
ion_models = [
    "fennix-bio1-small",
    "fennix-bio1-medium",
    "fennix-bio1-small-finetune-ions",
    "fennix-bio1-medium-finetune-ions",
]
for model_name in ion_models:
    atoms.calc = FENNIXCalculator(model_path(model_name), use_float64=True)
    results[f"methanol-ions/{model_name}"] = atoms.get_potential_energy()

atoms = ase.io.read(DATA_DIR / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb")
atoms.calc = FENNIXCalculator(model_path("fennix-bio1-small"), use_float64=True)
results["alanine-dipeptide-explicit/fennix-bio1-small"] = atoms.get_potential_energy()

for key in results:
    print(f"{key}: {results[key] * EV_TO_KJMOL}")
