# /// script
# dependencies = [
#   "ase",
#   "numpy",
#   "openmm",
#   "so3lr @ git+https://github.com/general-molecular-simulations/so3lr.git@5c6f36914bd2424563c1fd80bc21610960d947a4",
#   "mlff @ git+https://github.com/kabylda/mlff.git@aeb80dcd208a4607c01dbac6c9574b7b32bcf93e",
#   "glp @ git+https://github.com/kabylda/glp.git@f6955d50b34b352b2ea27b0ac1264909f4cde278",
#   "e3x",
#   "jax-pme",
#   "flax",
#   "ml-collections",
#   "orbax-checkpoint",
#   "pyyaml",
# ]
# ///
# This script computes SO3LR reference energies from the upstream SO3LR calculator.
#
# Run with:
#   uv run --script test/data/so3lr_energies.py

from pathlib import Path

import ase.io
import numpy as np
from openmm import unit
from so3lr import So3lrCalculator

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)


def calculate(path: Path) -> float:
    atoms = ase.io.read(path)
    atoms.info["charge"] = 0.0
    atoms.calc = So3lrCalculator(
        calculate_stress=False,
        lr_cutoff=12.0,
        dtype=np.float32,
    )
    return atoms.get_potential_energy() * EV_TO_KJMOL


results = {
    "toluene/so3lr": calculate(DATA_DIR / "toluene" / "toluene.pdb"),
    "alanine-dipeptide-explicit/so3lr": calculate(
        DATA_DIR / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb"
    ),
}

for key, value in results.items():
    print(f"{key}: {value!r}")
