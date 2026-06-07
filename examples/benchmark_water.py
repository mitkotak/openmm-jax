from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

from openmm import (
    LangevinMiddleIntegrator,
    LocalEnergyMinimizer,
    Platform,
    unit,
)
from openmm.app import PDBFile, Simulation
from openmmml.mlpotential import MLPotential

WATER_DIR = Path(__file__).with_name("water")
SIZES = [3, 9, 21, 30, 96, 774, 2661, 6282, 12255, 21384, 98880, 999999]
CASES = ("ani2x-jax", "ani2x-nnpops")
CASE_LABELS = {
    "ani2x-jax": "ANI2x-JAX (JaxForce)",
    "ani2x-nnpops": "ANI2x-NNPOps (PythonForce)",
}
TEMP_K = 400.0
FRICTION_PER_PS = 1.0
TIMESTEP_PS = 0.001
MINIMIZE_STEPS = 50
EQUILIBRATION_STEPS = 100
PRODUCTION_STEPS = 1000


def setup_simulation(model_name: str, size: int) -> Simulation:
    pdb = PDBFile(str(WATER_DIR / f"water_atoms_{size}.pdb"))
    topology = pdb.topology
    if model_name == "ani2x-jax":
        importlib.import_module("openmmjax_models.anixpotential")
        system = MLPotential("ani2x-jax").createSystem(
            topology,
            removeCMMotion=False,
        )
    elif model_name == "ani2x-nnpops":
        importlib.import_module("openmmml.models.anipotential")
        system = MLPotential("ani2x").createSystem(
            topology,
            removeCMMotion=False,
            modelIndex=0,
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
    LocalEnergyMinimizer.minimize(simulation.context, maxIterations=MINIMIZE_STEPS)
    simulation.context.setVelocitiesToTemperature(TEMP_K * unit.kelvin)
    simulation.step(EQUILIBRATION_STEPS)

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
    ns_per_day = (integrator.getStepSize() / time_per_step) / (
        unit.nanoseconds / unit.day
    )
    print(
        f"  {label:28s} {atom_count:>6} atoms: {float(ns_per_day):10.3f} ns/day "
        f"({PRODUCTION_STEPS} steps in {elapsed:.2f} s)",
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-case", choices=CASES)
    parser.add_argument("--size", type=int)
    args = parser.parse_args(argv)
    if (args.single_case is None) != (args.size is None):
        parser.error("--single-case and --size must be passed together")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.single_case is not None:
        run_case(args.single_case, args.size)
        return 0

    print(
        f"timestep={TIMESTEP_PS * 1000:.1f} fs "
        f"temperature={TEMP_K:.1f} K "
        f"minimize={MINIMIZE_STEPS} steps "
        f"equilibration={EQUILIBRATION_STEPS} steps "
        f"production={PRODUCTION_STEPS} steps"
    )
    for case in CASES:
        for size in SIZES:
            run_case(case, size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
