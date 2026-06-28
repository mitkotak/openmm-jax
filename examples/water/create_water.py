from __future__ import annotations

from pathlib import Path

from openmm import Vec3, VerletIntegrator, unit
from openmm.app import PME, ForceField, Modeller, PDBFile, Simulation, Topology

ATOM_COUNTS = (12, 24, 33, 93, 255, 777, 2661, 6288, 12261, 21384, 98880)
OUTPUT_DIR = Path(__file__).resolve().parent
TARGET_DENSITY = 0.997 * unit.gram / unit.milliliter
WATER_MOLAR_MASS = 18.01528 * unit.gram / unit.mole
PME_CUTOFF = 1.0 * unit.nanometer
SMALL_BOX_CUTOFF_FRACTION = 0.45
MINIMIZATION_MAX_ITERATIONS = 100


def create_water_box(water_count: int, box_length: float) -> Modeller:
    forcefield = ForceField("tip3pfb.xml")
    modeller = Modeller(Topology(), [])
    modeller.addSolvent(forcefield, model="tip3p", numAdded=water_count)

    old_length = modeller.topology.getUnitCellDimensions().value_in_unit(unit.nanometer)[0]
    scale = box_length / old_length
    positions = modeller.positions.value_in_unit(unit.nanometer)
    scaled_positions = [None] * len(positions)

    for residue in modeller.topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        center = Vec3(0.0, 0.0, 0.0)
        for atom_index in atom_indices:
            center += positions[atom_index]
        center /= len(atom_indices)
        scaled_center = center * scale
        for atom_index in atom_indices:
            scaled_positions[atom_index] = scaled_center + positions[atom_index] - center

    modeller.positions = scaled_positions * unit.nanometer
    modeller.topology.setPeriodicBoxVectors(
        (
            Vec3(box_length, 0.0, 0.0),
            Vec3(0.0, box_length, 0.0),
            Vec3(0.0, 0.0, box_length),
        )
        * unit.nanometer
    )
    return modeller


def minimize_water_box(
    modeller: Modeller,
    box_length: float,
    max_iterations: int,
) -> None:
    forcefield = ForceField("tip3pfb.xml")
    cutoff = min(PME_CUTOFF, SMALL_BOX_CUTOFF_FRACTION * box_length * unit.nanometer)

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=cutoff,
    )
    integrator = VerletIntegrator(0.001 * unit.picoseconds)
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy(maxIterations=max_iterations)
    modeller.positions = simulation.context.getState(positions=True).getPositions()


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for atom_count in ATOM_COUNTS:
        if atom_count % 3 != 0:
            raise ValueError(f"water atom count must be divisible by 3: {atom_count}")
        water_count = atom_count // 3
        volume = water_count * WATER_MOLAR_MASS / (
            unit.AVOGADRO_CONSTANT_NA * TARGET_DENSITY
        )
        box_length = volume.value_in_unit(unit.nanometer**3) ** (1.0 / 3.0)
        modeller = create_water_box(water_count, box_length)
        minimize_water_box(modeller, box_length, MINIMIZATION_MAX_ITERATIONS)
        path = OUTPUT_DIR / f"water_{box_length:.4f}nm_atoms_{atom_count}.pdb"
        with path.open("w") as f:
            PDBFile.writeFile(modeller.topology, modeller.positions, f)
        volume = (box_length * unit.nanometer) ** 3
        actual_density = (
            water_count * WATER_MOLAR_MASS / (unit.AVOGADRO_CONSTANT_NA * volume)
        ).value_in_unit(unit.gram / unit.milliliter)
        cutoff = min(
            PME_CUTOFF,
            SMALL_BOX_CUTOFF_FRACTION * box_length * unit.nanometer,
        ).value_in_unit(unit.nanometer)
        print(
            f"{path.name}: {water_count} waters, {box_length:.4f} nm, "
            f"{actual_density:.4f} g/mL, minimization=PME, cutoff={cutoff:.4f} nm"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
