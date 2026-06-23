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
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "methanol-ions": DATA_DIR / "methanol-ions" / "methanol-ions.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODEL_URLS = {
    "fennix-bio1-small": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0/fennix-bio1S.fnx",
    "fennix-bio1-medium": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0/fennix-bio1M.fnx",
    "fennix-bio1-small-finetune-ions": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1S-finetuneIons.fnx",
    "fennix-bio1-medium-finetune-ions": "https://raw.githubusercontent.com/FeNNol-tools/FeNNol-PMC/main/FENNIX-BIO1/v1.0-finetuneIons/fennix-bio1M-finetuneIons.fnx",
}
TOLUENE_MODELS = ["fennix-bio1-small", "fennix-bio1-medium"]
ION_MODELS = [
    "fennix-bio1-small",
    "fennix-bio1-medium",
    "fennix-bio1-small-finetune-ions",
    "fennix-bio1-medium-finetune-ions",
]
ALANINE_DIPEPTIDE_MODELS = ["fennix-bio1-small"]
ION_CHARGES = [0, 0, 0, 0, 0, 0, 1, 1, 1, -1, -1]


def model_path(model: str) -> str:
    cache_dir = Path(os.environ.get("FENNIX_MODEL_DIR", "/tmp/openmm-jax-fennix-models"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / Path(MODEL_URLS[model]).name
    if not path.is_file():
        urllib.request.urlretrieve(MODEL_URLS[model], path)
    return str(path)


def calculate_energy(path: Path, model_name: str, charges: list[int] | None = None) -> float:
    atoms = ase.io.read(path)
    if charges is not None:
        atoms.set_initial_charges(charges)
    atoms.calc = FENNIXCalculator(model_path(model_name), use_float64=True)
    return atoms.get_potential_energy() * EV_TO_KJMOL


def calculate_results() -> dict[str, float]:
    results = {}

    for model_name in TOLUENE_MODELS:
        results[f"toluene/{model_name}"] = calculate_energy(SYSTEMS["toluene"], model_name)

    for model_name in ION_MODELS:
        results[f"methanol-ions/{model_name}"] = calculate_energy(
            SYSTEMS["methanol-ions"],
            model_name,
            charges=ION_CHARGES,
        )

    for model_name in ALANINE_DIPEPTIDE_MODELS:
        results[f"alanine-dipeptide-explicit/{model_name}"] = calculate_energy(
            SYSTEMS["alanine-dipeptide-explicit"],
            model_name,
        )

    return results


def print_results(results: dict[str, float]) -> None:
    for key, value in results.items():
        print(f"{key}: {value!r}")


def main() -> None:
    print_results(calculate_results())


if __name__ == "__main__":
    main()
