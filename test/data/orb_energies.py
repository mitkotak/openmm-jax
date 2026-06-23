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
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODEL = "orb-v3-conservative-omol"


def make_calculator():
    pretrained = orb.ORB_PRETRAINED_MODELS[MODEL](precision="float32-highest")
    if isinstance(pretrained, tuple):
        orbff, atoms_adapter = pretrained
        return ORBCalculator(orbff, atoms_adapter=atoms_adapter)
    return ORBCalculator(pretrained)


def calculate_energy(path: Path, charge: int, spin: int) -> float:
    atoms = ase.io.read(path)
    atoms.info["charge"] = charge
    atoms.info["spin"] = spin
    atoms.calc = make_calculator()
    return atoms.get_potential_energy() * EV_TO_KJMOL


def calculate_results() -> dict[str, float]:
    results = {}
    results[f"toluene/{MODEL}"] = calculate_energy(SYSTEMS["toluene"], charge=0, spin=1)
    results[f"toluene/{MODEL}/override-charge-spin"] = calculate_energy(
        SYSTEMS["toluene"],
        charge=-1,
        spin=3,
    )
    results[f"alanine-dipeptide-explicit/{MODEL}"] = calculate_energy(
        SYSTEMS["alanine-dipeptide-explicit"],
        charge=0,
        spin=1,
    )
    return results


def print_results(results: dict[str, float]) -> None:
    for key, value in results.items():
        print(f"{key}: {value!r}")


def main() -> None:
    print_results(calculate_results())


if __name__ == "__main__":
    main()
