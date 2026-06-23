# /// script
# dependencies = ["ase", "equinox", "jax", "jax-md", "numpy", "openmm"]
# ///
# This script computes reference energies for the ANI2x JAX models.

from pathlib import Path

import ase.io
import jax
import jax.numpy as jnp
import numpy as np
from openmm import unit

from models.ani import load_ani2x_model

DATA_DIR = Path(__file__).resolve().parent
HARTREE_TO_KJMOL = (unit.hartree * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
SYSTEMS = {
    "toluene": DATA_DIR / "toluene" / "toluene.pdb",
    "alanine-dipeptide-explicit": DATA_DIR
    / "alanine-dipeptide"
    / "alanine-dipeptide-explicit.pdb",
}
MODELS = ["ani2x-jax-model0", "ani2x-jax-ensemble"]


def calculate_energy(path: Path, model_name: str) -> float:
    atoms = ase.io.read(path)
    species = jnp.asarray(np.asarray(atoms.numbers, dtype=np.int32))
    model = load_ani2x_model(model_name, atomic_numbers=species)
    energy = model(
        jnp.asarray(atoms.positions, dtype=jnp.float32),
        model.species_indices(species),
        periodic=False,
    )
    return float(jax.device_get(energy * HARTREE_TO_KJMOL))


def calculate_results() -> dict[str, float]:
    results = {}

    for system, path in SYSTEMS.items():
        for model_name in MODELS:
            results[f"{system}/{model_name}"] = calculate_energy(path, model_name)

    return results


def print_results(results: dict[str, float]) -> None:
    for key, value in results.items():
        print(f"{key}: {value!r}")


def main() -> None:
    print_results(calculate_results())


if __name__ == "__main__":
    main()
