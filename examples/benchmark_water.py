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
SIZES = [3, 9, 21, 30, 96, 774, 2661, 6282, 12255, 21384, 98880, 999999]
CASES = ("mace-off-s(23)", "mace-off-m(24)")
# CASES = ("fennix-bio1-small-jax", "fennix-bio1-small-python")
# CASES = ("ani2x-jax-model0", "ani2x-jax-ensemble")

CASE_LABELS = {
    "fennix-bio1-small-jax": "FeNNix-S (JaxForce)",
    "fennix-bio1-small-python": "FeNNiX-S (PythonForce)",
    "ani2x-jax": "ANI2x-JAX (JaxForce)",
    "ani2x-jax-model0": "ANI2x-JAX model0 (JaxForce)",
    "ani2x-jax-ensemble": "ANI2x-JAX ensemble (JaxForce)",
    "ani2x-jax-python": "ANI2x-JAX model 0(PythonForce)",
    "mace-off-s(23)": "MACE-OFF-S(23) (JaxForce)",
    "mace-off-m(24)": "MACE-OFF-M(24) (JaxForce)",
}
TEMP_K = 400.0
FRICTION_PER_PS = 1.0
TIMESTEP_PS = 0.001

# Need to skip minimization since it triggers energy+force call which goes OOM on RTX
MINIMIZE_STEPS = 0
# MINIMIZE_STEPS = 50

EQUILIBRATION_STEPS = 100
PRODUCTION_STEPS = 100

def setup_simulation(model_name: str, size: int) -> tuple[Simulation, dict[str, object]]:
    pdb = PDBFile(str(WATER_DIR / f"water_atoms_{size}.pdb"))
    topology = pdb.topology
    if model_name == "fennix-bio1-small-jax":
        importlib.import_module("openmmjax_models.fennixpotential")
        system = MLPotential("fennix-bio1-small-jax").createSystem(
            topology,
            removeCMMotion=False,
        )
    elif model_name == "fennix-bio1-small-python":
        importlib.import_module("openmmml.models.fennixpotential")
        system = MLPotential("fennix-bio1-small").createSystem(
            topology,
            removeCMMotion=False,
        )
    elif model_name.startswith("ani2x-jax"):
        importlib.import_module("openmmjax_models.anipotential")
        system = MLPotential(model_name).createSystem(
            topology,
            removeCMMotion=False,
            periodic_neighborlist=False,
        )
    elif model_name == "ani2x-jax-python":
        importlib.import_module("openmmjax_models.anipotential_pythonforce")
        system = MLPotential("ani2x-jax-python").createSystem(
            topology,
            removeCMMotion=False,
        ) 
    elif model_name.startswith("mace-off-"):
        importlib.import_module("openmmjax_models.macepotential")
        system = MLPotential(model_name).createSystem(
            topology,
            removeCMMotion=False,
            periodic_neighborlist=False,
        )
    else:
        raise ValueError(f"unknown benchmark case: {model_name}")

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
