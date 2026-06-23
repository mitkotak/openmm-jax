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
MODEL = "aimnet2-jax"
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}


def calculate_energy(path: Path) -> float:
    atoms = ase.io.read(path)
    atoms.calc = AIMNet2ASE("aimnet2", charge=0)
    return atoms.get_potential_energy() * EV_TO_KJMOL


def calculate_results() -> dict[str, float]:
    return {f"{system}/{MODEL}": calculate_energy(path) for system, path in SYSTEMS.items()}


def print_results(results: dict[str, float]) -> None:
    for key, value in results.items():
        print(f"{key}: {value!r}")


def main() -> None:
    print_results(calculate_results())


if __name__ == "__main__":
    main()
