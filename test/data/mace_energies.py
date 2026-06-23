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
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODELS = {
    "mace-jax-off-s-23": "small",
    "mace-jax-off-m-24": (
        "https://github.com/ACEsuit/mace-off/blob/main/mace_off24/MACE-OFF24_medium.model?raw=true"
    ),
}


def calculate_energy(path: Path, checkpoint: str) -> float:
    atoms = ase.io.read(path)
    atoms.calc = mace_off(checkpoint, device="cpu")
    return atoms.get_potential_energy() * EV_TO_KJMOL


def calculate_results() -> dict[str, float]:
    results = {}

    for model_name, checkpoint in MODELS.items():
        results[f"toluene/{model_name}"] = calculate_energy(SYSTEMS["toluene"], checkpoint)

    results["alanine-dipeptide-explicit/mace-jax-off-s-23"] = calculate_energy(
        SYSTEMS["alanine-dipeptide-explicit"],
        MODELS["mace-jax-off-s-23"],
    )
    return results


def print_results(results: dict[str, float]) -> None:
    for key, value in results.items():
        print(f"{key}: {value!r}")


def main() -> None:
    print_results(calculate_results())


if __name__ == "__main__":
    main()
