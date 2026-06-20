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


def compute_energy(atoms, model_name):
    species = jnp.asarray(np.asarray(atoms.numbers, dtype=np.int32))
    model = load_ani2x_model(model_name, atomic_numbers=species)
    energy = model(
        jnp.asarray(atoms.positions, dtype=jnp.float32),
        model.species_indices(species),
        periodic=False,
    )
    return float(jax.device_get(energy * HARTREE_TO_KJMOL))


results = {}

for system, path in [
    ("toluene", DATA_DIR / "toluene" / "toluene.pdb"),
    (
        "alanine-dipeptide-explicit",
        DATA_DIR / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb",
    ),
]:
    atoms = ase.io.read(path)
    for model_name in ["ani2x-model-0", "ani2x-jax-ensemble"]:
        results[f"{system}/{model_name}"] = compute_energy(atoms, model_name)

for key in results:
    print(f"{key}: {results[key]}")
