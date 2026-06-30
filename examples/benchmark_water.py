from __future__ import annotations

import importlib
import math
import time
from pathlib import Path

from openmm import (
    LangevinMiddleIntegrator,
    Platform,
    unit,
)
from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, PME, Simulation
from openmmml.mlpotential import MLPotential

WATER_DIR = Path(__file__).with_name("water")
# SIZES = [12, 24, 33, 93, 777, 2661, 6288, 12261, 21384, 98880]
SIZES = [12, 24, 33, 93, 255, 777, 2661]
WATER_PDBS = {
    12: "water_0.4933nm_atoms_12.pdb",
    24: "water_0.6215nm_atoms_24.pdb",
    33: "water_0.6911nm_atoms_33.pdb",
    93: "water_0.9762nm_atoms_93.pdb",
    255: "water_1.3663nm_atoms_255.pdb",
    777: "water_1.9808nm_atoms_777.pdb",
    2661: "water_2.9857nm_atoms_2661.pdb",
    6288: "water_3.9768nm_atoms_6288.pdb",
    12261: "water_4.9682nm_atoms_12261.pdb",
    21384: "water_5.9803nm_atoms_21384.pdb",
    98880: "water_9.9631nm_atoms_98880.pdb",
}
CASES = (
    "amber14-tip3p-pme",
    "ani2x-jax-model0",
    "aimnet2-jax",
    "aceff-jax-1.1",
    "aceff-jax-2.0",
    "mace-jax-off-s-23",
    "so3lr",
    "orb-jax-v3-conservative-omol",
)

CASE_LABELS = {
    "amber14-tip3p-pme": "AMBER14 TIP3P PME",
    "fennix-bio1-small": "FeNNix-S (JaxForce)",
    "ani2x-jax-model0": "ANI2x-JAX model0 (JaxForce)",
    "ani2x-jax-ensemble": "ANI2x-JAX ensemble (JaxForce)",
    "aimnet2-jax": "AIMNet2-JAX (JaxForce)",
    "mace-jax-off-s-23": "MACE-JAX-OFF-S(23) (JaxForce)",
    "mace-jax-off-m-24": "MACE-JAX-OFF-M(24) (JaxForce)",
    "aceff-jax-1.1": "AceFF-JAX-1.1 (JaxForce)",
    "aceff-jax-2.0": "AceFF-JAX-2.0 (JaxForce)",
    "so3lr": "SO3LR (JaxForce)",
    "orb-jax-v3-conservative-omol": "ORB-v3 Conservative OMOL (JaxForce)",
}
TEMP_K = 400.0
FRICTION_PER_PS = 1.0
TIMESTEP_PS = 0.001
NONBONDED_CUTOFF_NM = 1.0
EWALD_TOL = 5.0e-4

# Need to skip minimization since it triggers energy+force call which goes OOM on RTX
MINIMIZE_STEPS = 0
# MINIMIZE_STEPS = 50

EQUILIBRATION_STEPS = 100
WARMUP_STEPS = 10
PRODUCTION_STEPS = 100


def import_model_module(model_name: str) -> None:
    if model_name == "fennix-bio1-small":
        importlib.import_module("openmmjax_models.fennixpotential")
    elif model_name.startswith("ani2x-jax"):
        importlib.import_module("openmmjax_models.anipotential")
    elif model_name == "aimnet2-jax":
        importlib.import_module("openmmjax_models.aimnet2potential")
    elif model_name.startswith("mace-jax-off-"):
        importlib.import_module("openmmjax_models.macepotential")
    elif model_name.startswith("aceff-"):
        importlib.import_module("openmmjax_models.aceffpotential")
    elif model_name == "so3lr":
        importlib.import_module("openmmjax_models.so3lrpotential")
    elif model_name == "orb-jax-v3-conservative-omol":
        importlib.import_module("openmmjax_models.orbpotential")
    else:
        raise ValueError(f"unknown benchmark case: {model_name}")


def create_amber14_tip3p_system(topology):
    forcefield = ForceField("amber14/tip3p.xml")
    forcefield_kwargs = {
        "constraints": HBonds,
        "rigidWater": True,
        "ewaldErrorTolerance": EWALD_TOL,
        "removeCMMotion": False,
    }
    if topology.getPeriodicBoxVectors() is None:
        forcefield_kwargs["nonbondedMethod"] = NoCutoff
    else:
        forcefield_kwargs["nonbondedMethod"] = PME
        box_lengths = []
        for vector in topology.getPeriodicBoxVectors():
            vec = vector.value_in_unit(unit.nanometer)
            box_lengths.append(math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z))
        forcefield_kwargs["nonbondedCutoff"] = (
            min(NONBONDED_CUTOFF_NM, 0.49 * min(box_lengths)) * unit.nanometer
        )
    return forcefield.createSystem(topology, **forcefield_kwargs)


def setup_simulation(model_name: str, size: int) -> Simulation:
    pdb = PDBFile(str(WATER_DIR / WATER_PDBS[size]))
    topology = pdb.topology
    if model_name == "amber14-tip3p-pme":
        system = create_amber14_tip3p_system(topology)
    else:
        import_model_module(model_name)
    if model_name != "amber14-tip3p-pme":
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
        f"  {label:28s} {atom_count:>6} atoms: {elapsed:.4f} s {float(ns_per_day):10.3f} ns/day ",
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
