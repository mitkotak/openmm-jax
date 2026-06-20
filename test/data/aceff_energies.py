# /// script
# dependencies = ["ase", "huggingface-hub", "openmm", "torchmd-net"]
# ///
# This script computes AceFF reference energies from upstream TorchMD-Net checkpoints.

from pathlib import Path

import ase.io
from huggingface_hub import hf_hub_download
from openmm import unit
from torchmdnet.calculators import TMDNETCalculator

DATA_DIR = Path(__file__).resolve().parent
EV_TO_KJMOL = (unit.elementary_charge * unit.volt * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
    unit.kilojoules_per_mole
)
MODELS = {
    "aceff-jax-1.1": (
        "Acellera/AceFF-1.1",
        "aceff_v1.1.ckpt",
        {},
    ),
    "aceff-jax-2.0": (
        "Acellera/AceFF-2.0",
        "aceff_v2.0.ckpt",
        {"coulomb_cutoff": 12.0},
    ),
}

results = {}

for system_name, pdb_path in [
    ("toluene", DATA_DIR / "toluene" / "toluene.pdb"),
    (
        "alanine-dipeptide-explicit",
        DATA_DIR / "alanine-dipeptide" / "alanine-dipeptide-explicit.pdb",
    ),
]:
    atoms = ase.io.read(pdb_path)
    atoms.info["charge"] = 0
    for model_name, (repo_id, filename, kwargs) in MODELS.items():
        model_file = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        atoms.calc = TMDNETCalculator(
            model_file,
            device="cpu",
            remove_ref_energy=True,
            max_num_neighbors=min(64, len(atoms)),
            **kwargs,
        )
        results[f"{system_name}/{model_name}"] = atoms.get_potential_energy() * EV_TO_KJMOL

for key in results:
    print(f"{key}: {results[key]!r}")
