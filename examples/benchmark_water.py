from __future__ import annotations

import importlib
import time
from pathlib import Path

from openmm import (
    LangevinMiddleIntegrator,
    Platform,
    unit,
)
from openmm.app import PDBFile, Simulation
from openmmml.mlpotential import MLPotential

WATER_DIR = Path(__file__).with_name("water")
# SIZES = [3, 12, 24, 33, 93, 777, 2661, 6288, 12261, 21384, 98880, 999978]
SIZES = [93, 777, 2661]
# CASES = (
#     "mace-jax-off-s-23",
#     "mace-jax-off-s-23-python",
#     "mace-jax-off-m-24",
#     "mace-jax-off-m-24-python",
# )
# CASES = ("fennix-bio1-small-python", "fennix-bio1-small")
# CASES = ("ani2x-jax-model0", "ani2x-jax-python")
# CASES = ("aimnet2-jax", "aimnet2-jax-python")
# CASES = ("aceff-jax-1.1-python", "aceff-jax-1.1")
CASES = ("so3lr", "fennix-bio1-small")

CASE_LABELS = {
    "fennix-bio1-small": "FeNNix-S (JaxForce)",
    "fennix-bio1-small-python": "FeNNiX-S (PythonForce)",
    "ani2x-jax-model0": "ANI2x-JAX model0 (JaxForce)",
    "ani2x-jax-ensemble": "ANI2x-JAX ensemble (JaxForce)",
    "ani2x-jax-python": "ANI2x-JAX model0 (PythonForce)",
    "aimnet2-jax": "AIMNet2-JAX (JaxForce)",
    "aimnet2-jax-python": "AIMNet2-JAX (PythonForce)",
    "mace-jax-off-s-23": "MACE-JAX-OFF-S(23) (JaxForce)",
    "mace-jax-off-m-24": "MACE-JAX-OFF-M(24) (JaxForce)",
    "mace-jax-off-s-23-python": "MACE-JAX-OFF-S(23) (PythonForce)",
    "mace-jax-off-m-24-python": "MACE-JAX-OFF-M(24) (PythonForce)",
    "aceff-jax-1.1": "AceFF-JAX-1.1 (JaxForce)",
    "aceff-jax-1.1-python": "AceFF-JAX-1.1 (PythonForce)",
    "aceff-jax-2.0": "AceFF-JAX-2.0 (JaxForce)",
    "aceff-jax-2.0-python": "AceFF-JAX-2.0 (PythonForce)",
    "so3lr": "SO3LR (JaxForce)"
}
TEMP_K = 400.0
FRICTION_PER_PS = 1.0
TIMESTEP_PS = 0.001

# Need to skip minimization since it triggers energy+force call which goes OOM on RTX
MINIMIZE_STEPS = 0
# MINIMIZE_STEPS = 50

EQUILIBRATION_STEPS = 100
WARMUP_STEPS = 10
PRODUCTION_STEPS = 100


def setup_simulation(model_name: str, size: int) -> tuple[Simulation, dict[str, object]]:
    pdb = PDBFile(str(WATER_DIR / f"water_atoms_{size}.pdb"))
    topology = pdb.topology
    if model_name == "fennix-bio1-small":
        importlib.import_module("openmmjax_models.fennixpotential")
    elif model_name == "fennix-bio1-small-python":
        importlib.import_module("openmmjax_models.fennixpotential_pythonforce")
    elif model_name == "ani2x-jax-python":
        importlib.import_module("openmmjax_models.anipotential_pythonforce")
    elif model_name.startswith("ani2x-jax"):
        importlib.import_module("openmmjax_models.anipotential")
    elif model_name == "aimnet2-jax":
        importlib.import_module("openmmjax_models.aimnet2potential")
    elif model_name == "aimnet2-jax-python":
        importlib.import_module("openmmjax_models.aimnet2potential_pythonforce")
    elif model_name.startswith("mace-jax-off-") and not model_name.endswith("-python"):
        importlib.import_module("openmmjax_models.macepotential")
    elif model_name.startswith("mace-jax-off-") and model_name.endswith("-python"):
        importlib.import_module("openmmjax_models.macepotential_pythonforce")
    elif model_name.startswith("aceff-") and model_name.endswith("-python"):
        importlib.import_module("openmmjax_models.aceffpotential_pythonforce")
    elif model_name.startswith("aceff-"):
        importlib.import_module("openmmjax_models.aceffpotential")
    elif model_name == "so3lr":
        importlib.import_module("openmmjax_models.so3lrpotential")
    else:
        raise ValueError(f"unknown benchmark case: {model_name}")
    system = MLPotential(model_name).createSystem(
            topology,
            removeCMMotion=False,
            periodic_neighborlist=False,
            preprocessing_positions=pdb.positions,
        )
    integrator = LangevinMiddleIntegrator(
        TEMP_K * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        TIMESTEP_PS * unit.picoseconds,
    )
    integrator.setConstraintTolerance(1.0e-5)
    simulation = Simulation(
        topology,
        system,
        integrator,
        Platform.getPlatformByName("CUDA"),
        {"Precision": "mixed"},
    )

    simulation.context.setPositions(pdb.positions)
    if MINIMIZE_STEPS > 0:
        simulation.minimizeEnergy(maxIterations=MINIMIZE_STEPS)
    simulation.context.setVelocitiesToTemperature(TEMP_K * unit.kelvin)
    simulation.step(EQUILIBRATION_STEPS)
    simulation.context.getState(energy=True)

    return simulation


def run_simulation(
    model_name: str,
    size: int,
    simulation: Simulation,
) -> dict[str, object]:
    integrator = simulation.context.getIntegrator()
    atom_count = simulation.topology.getNumAtoms()
    label = CASE_LABELS.get(model_name, model_name)

    simulation.context.getState(energy=True)
    start = time.perf_counter()
    simulation.step(PRODUCTION_STEPS)
    simulation.context.getState(energy=True)
    elapsed = time.perf_counter() - start
    time_per_step = elapsed * unit.seconds / PRODUCTION_STEPS
    ns_per_day = (integrator.getStepSize() / time_per_step) / (unit.nanoseconds / unit.day)
    print(
        f"  {label:28s} {atom_count:>6} atoms: {float(ns_per_day):10.3f} ns/day ",
        flush=True,
    )
    return {
        "case": model_name,
        "label": label,
        "fixture_size": size,
        "atoms": atom_count,
        "equilibration_steps": EQUILIBRATION_STEPS,
        "production_steps": PRODUCTION_STEPS,
        "ns_per_day": float(ns_per_day),
    }


def run_case(model_name: str, size: int) -> None:
    simulation = setup_simulation(model_name, size)
    run_simulation(model_name, size, simulation)


def main() -> int:
    print(
        f"timestep={TIMESTEP_PS * 1000:.1f} fs "
        f"temperature={TEMP_K:.1f} K "
        f"minimize={MINIMIZE_STEPS} steps "
        f"equilibration={EQUILIBRATION_STEPS} steps "
        f"production={PRODUCTION_STEPS} steps"
    )
    for size in SIZES:
        for case in CASES:
            run_case(case, size)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
