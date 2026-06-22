#!/usr/bin/env python3
"""Phenol-water RDF following arXiv:2604.21441 Figure S2."""

from __future__ import annotations

import importlib
import time
from datetime import datetime, timezone
from pathlib import Path

import mdtraj as md
import numpy as np
import openmm
import openmm.app as app
import openmm.unit as unit
from openff.toolkit import Molecule
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmml.mlpotential import MLPotential

INPUT_PDB = Path(__file__).with_name("phenol.pdb")
PHENOL_SMILES = "c1ccccc1O"
SMALL_MOLECULE_FORCEFIELD = "gaff-2.2.20"
CASES = ("aceff-jax-1.1-python", "aceff-jax-1.1", "mm",)
CASE_LABELS = {
    "mm": "GAFF/TIP3P",
    "ani2x-jax-model0": "ANI2x-JAX model0",
    "ani2x-jax-ensemble": "ANI2x-JAX ensemble",
    "fennix-bio1-small": "FeNNix-S (JaxForce)",
    "fennix-bio1-small-python": "FeNNix-S (PythonForce)",
    "mace-jax-off-s-23": "MACE-OFF-S(23) (JaxForce)",
    "mace-jax-off-m-24": "MACE-OFF-M(24) (JaxForce)",
    "aimnet2-jax": "AIMNet2-JAX (JaxForce)",
    "aimnet2-jax-python": "AIMNet2-JAX (PythonForce)",
    "aceff-jax-1.1": "AceFF-JAX-1.1 (JaxForce)",
    "aceff-jax-1.1-python": "AceFF-JAX-1.1 (PythonForce)",
    "aceff-jax-2.0": "AceFF-JAX-2.0 (JaxForce)",
    "aceff-jax-2.0-python": "AceFF-JAX-2.0 (PythonForce)",
 
}
CASE_COLORS = {
    "GAFF/TIP3P": "#3d64c8",
    "ANI2x-JAX model0 (JaxForce)": "#c44e52",
    "ANI2x-JAX ensemble (JaxForce)": "#55a868",
    "FeNNix-S (JaxForce)": "#f05a9d",
    "FeNNix-S (PythonForce)": "#f5a623",
    "MACE-OFF-S(23) (JaxForce)": "#7a52cc",
    "MACE-OFF-M(24) (JaxForce)": "#00bfa5",
    "AIMNet2-JAX (JaxForce)": "#7a52cc",
    "AIMNet2-JAX (PythonForce)": "#00bfa5",
    "AceFF-JAX-1.1 (JaxForce)": "#c44e52",
    "AceFF-JAX-1.1 (PythonForce)": "#55a868",
    "AceFF-JAX-2.0 (JaxForce)": "#7a52cc",
    "AceFF-JAX-2.0 (PythonForce)": "#00bfa5",
}
PLATFORM = "CUDA"
TEMP_K = 300.0
PRESSURE_BAR = 1.0
FRICTION_PER_PS = 1.0
NONBONDED_CUTOFF_NM = 1.0
EWALD_TOL = 5.0e-4
WATERS = 3991
MINIMIZE_STEPS = 10_000
EQUIL_PS = 100.0
EQUIL_TIMESTEP_FS = 2.0
PRODUCTION_PS = 200.0
PRODUCTION_TIMESTEP_FS = 0.5
RDF_DISCARD_PS = 5.0
RDF_INTERVAL_STEPS = 100
RDF_RMIN_NM = 0.2
RDF_RMAX_NM = 0.9
RDF_BIN_WIDTH_NM = 0.002
SEED = 2026042141
EQUIL_STEPS = int(round(EQUIL_PS * 1000.0 / EQUIL_TIMESTEP_FS))
PRODUCTION_STEPS = int(round(PRODUCTION_PS * 1000.0 / PRODUCTION_TIMESTEP_FS))
RDF_DISCARD_STEPS = int(round(RDF_DISCARD_PS * 1000.0 / PRODUCTION_TIMESTEP_FS))


def create_forcefield(cache_path: Path) -> app.ForceField:
    phenol = Molecule.from_smiles(PHENOL_SMILES)
    phenol.name = "phenol"
    gaff = GAFFTemplateGenerator(
        molecules=phenol,
        forcefield=SMALL_MOLECULE_FORCEFIELD,
        cache=str(cache_path),
    )

    forcefield = app.ForceField("amber/tip3p_standard.xml")
    forcefield.registerTemplateGenerator(gaff.generator)
    return forcefield


def create_mm_system(topology: app.Topology, forcefield: app.ForceField) -> openmm.System:
    forcefield_kwargs = {
        "constraints": app.HBonds,
        "rigidWater": True,
        "ewaldErrorTolerance": EWALD_TOL,
        "removeCMMotion": False,
    }
    if topology.getPeriodicBoxVectors() is None:
        forcefield_kwargs["nonbondedMethod"] = app.NoCutoff
    else:
        forcefield_kwargs["nonbondedMethod"] = app.PME
        forcefield_kwargs["nonbondedCutoff"] = NONBONDED_CUTOFF_NM * unit.nanometer
    return forcefield.createSystem(topology, **forcefield_kwargs)


def setup_simulation(output_dir: Path) -> dict[str, object]:
    pdb = app.PDBFile(str(INPUT_PDB))
    forcefield = create_forcefield(output_dir / "phenol-gaff-cache.json")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model="tip3p", numAdded=WATERS)

    platform = openmm.Platform.getPlatformByName(PLATFORM)
    props = {"Precision": "mixed"} if PLATFORM == "CUDA" else {}

    system = create_mm_system(modeller.topology, forcefield)
    integrator = openmm.LangevinMiddleIntegrator(
        TEMP_K * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        EQUIL_TIMESTEP_FS * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(SEED)
    simulation = app.Simulation(modeller.topology, system, integrator, platform, props)
    simulation.context.setPositions(modeller.positions)

    print(f"Minimizing {modeller.topology.getNumAtoms()} atoms...", flush=True)
    simulation.minimizeEnergy(maxIterations=MINIMIZE_STEPS)
    state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
    topology = simulation.topology
    topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
    positions = state.getPositions()

    for label, seed, use_barostat in (
        ("NVT", SEED + 1, False),
        ("NPT", SEED + 2, True),
    ):
        print(f"Equilibrating {label} for {EQUIL_PS:g} ps...", flush=True)
        system = create_mm_system(topology, forcefield)
        if use_barostat:
            system.addForce(
                openmm.MonteCarloBarostat(PRESSURE_BAR * unit.bar, TEMP_K * unit.kelvin)
            )
        integrator = openmm.LangevinMiddleIntegrator(
            TEMP_K * unit.kelvin,
            FRICTION_PER_PS / unit.picosecond,
            EQUIL_TIMESTEP_FS * unit.femtoseconds,
        )
        integrator.setRandomNumberSeed(seed)
        simulation = app.Simulation(topology, system, integrator, platform, props)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(TEMP_K * unit.kelvin, seed)
        simulation.step(EQUIL_STEPS)
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        topology = simulation.topology
        topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
        positions = state.getPositions()

    return {
        "topology": topology,
        "positions": positions,
        "forcefield": forcefield,
    }


def run_simulation(
    model_name: str,
    prepared: dict[str, object],
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, md.Topology]:
    topology = prepared["topology"]
    forcefield = prepared["forcefield"]
    mm_system = create_mm_system(topology, forcefield)
    md_topology = md.Topology.from_openmm(topology)
    ml_atoms = md_topology.select("resname PHN").astype(int).tolist()
    if not ml_atoms:
        raise ValueError("no atoms found in residue PHN")

    if model_name == "mm":
        system = mm_system
    elif model_name == "ani2x-jax-python":
        importlib.import_module("openmmjax_models.anipotential_pythonforce")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name.startswith("ani2x-jax-"):
        importlib.import_module("openmmjax_models.anipotential")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name == "fennix-bio1-small":
        importlib.import_module("openmmjax_models.fennixpotential")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential("fennix-bio1-small").createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name == "fennix-bio1-small-python":
        importlib.import_module("openmmjax_models.fennixpotential_pythonforce")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential("fennix-bio1-small-python").createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name.startswith("mace-jax-off-") and model_name.endswith("-python"):
        importlib.import_module("openmmjax_models.macepotential_pythonforce")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name.startswith("mace-jax-off-"):
        importlib.import_module("openmmjax_models.macepotential")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name == "aimnet2-jax":
        importlib.import_module("openmmjax_models.aimnet2potential")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name == "aimnet2-jax-python":
        importlib.import_module("openmmjax_models.aimnet2potential_pythonforce")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        ) 
    elif model_name.startswith("aceff-") and model_name.endswith("-python"):
        importlib.import_module("openmmjax_models.aceffpotential_pythonforce")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    elif model_name.startswith("aceff-") and not model_name.endswith("-python"):
        importlib.import_module("openmmjax_models.aceffpotential")
        cloned = openmm.XmlSerializer.deserialize(openmm.XmlSerializer.serialize(mm_system))
        system = MLPotential(model_name).createMixedSystem(
            topology,
            cloned,
            ml_atoms,
            removeConstraints=True,
            periodic_neighborlist=False,
            preprocessing_positions=prepared["positions"],
        )
    else:
        raise ValueError(f"unknown RDF case: {model_name}")

    platform = openmm.Platform.getPlatformByName(PLATFORM)
    props = {"Precision": "mixed"} if PLATFORM == "CUDA" else {}
    integrator = openmm.LangevinMiddleIntegrator(
        TEMP_K * unit.kelvin,
        FRICTION_PER_PS / unit.picosecond,
        PRODUCTION_TIMESTEP_FS * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(SEED + 10)
    simulation = app.Simulation(topology, system, integrator, platform, props)
    simulation.context.setPositions(prepared["positions"])
    simulation.context.setVelocitiesToTemperature(TEMP_K * unit.kelvin, SEED + 10)

    phenol_o = md_topology.select("resname PHN and element O")
    water_o = md_topology.select("water and element O")
    if len(phenol_o) != 1 or len(water_o) == 0:
        raise ValueError("expected one phenol oxygen and at least one water oxygen")

    rdf_atom_indices = np.concatenate((phenol_o, water_o)).astype(int)
    rdf_pairs = np.array([[0, index] for index in range(1, len(rdf_atom_indices))], dtype=int)
    rdf_topology = md_topology.subset(rdf_atom_indices.tolist())
    rdf_positions = []
    rdf_box_vectors = []

    start = time.perf_counter()
    completed_steps = 0
    while completed_steps < PRODUCTION_STEPS:
        steps = min(RDF_INTERVAL_STEPS, PRODUCTION_STEPS - completed_steps)
        simulation.step(steps)
        completed_steps += steps
        if completed_steps <= RDF_DISCARD_STEPS:
            continue
        state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        box_vectors = np.asarray(
            [vector.value_in_unit(unit.nanometer) for vector in state.getPeriodicBoxVectors()]
        )
        rdf_positions.append(positions[rdf_atom_indices])
        rdf_box_vectors.append(box_vectors)
    elapsed = time.perf_counter() - start

    ns_per_day = PRODUCTION_STEPS * PRODUCTION_TIMESTEP_FS * 1.0e-6 * 86400.0 / elapsed
    print(
        f"  {len(rdf_positions)} RDF samples, {ns_per_day:.3f} ns/day",
        flush=True,
    )
    return rdf_positions, rdf_box_vectors, rdf_pairs, rdf_topology


def compute_rdf(
    positions_by_sample: list[np.ndarray],
    box_vectors_by_sample: list[np.ndarray],
    atom_pairs: np.ndarray,
    topology: md.Topology,
) -> tuple[np.ndarray, np.ndarray]:
    if not positions_by_sample:
        raise ValueError("no RDF samples were collected")
    trajectory = md.Trajectory(np.asarray(positions_by_sample, dtype=np.float32), topology)
    trajectory.unitcell_vectors = np.asarray(box_vectors_by_sample, dtype=np.float32)
    return md.compute_rdf(
        trajectory,
        atom_pairs,
        r_range=(RDF_RMIN_NM, RDF_RMAX_NM),
        bin_width=RDF_BIN_WIDTH_NM,
        periodic=True,
    )


def plot_rdfs(
    rdf_series: list[tuple[str, np.ndarray, np.ndarray]],
    output_dir: Path,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(4.8, 3.35), constrained_layout=True)
    for label, radii, rdf in rdf_series:
        ax.plot(
            radii,
            rdf,
            linewidth=1.15,
            color=CASE_COLORS.get(label),
            label=label,
        )
    ax.set_xlabel(r"$r_{O-O_W}$")
    ax.set_ylabel("g(r)")
    ax.set_xlim(RDF_RMIN_NM, RDF_RMAX_NM)
    ax.set_ylim(0.0, 1.5)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.legend(loc="lower right", frameon=False)

    plot_path = output_dir / "phenol_o_water_o_rdf.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    return plot_path


def main() -> None:
    output_dir = Path.cwd() / f"phenol_rdf_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = setup_simulation(output_dir)

    rdf_series = []
    for case in CASES:
        print(f"Running {case} production...", flush=True)
        rdf_positions, rdf_box_vectors, rdf_pairs, rdf_topology = run_simulation(
            case,
            prepared,
        )
        radii, rdf = compute_rdf(rdf_positions, rdf_box_vectors, rdf_pairs, rdf_topology)
        rdf_series.append((CASE_LABELS.get(case, case), radii, rdf))
    plot_path = plot_rdfs(rdf_series, output_dir)
    print(f"RDF plot: {plot_path}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
