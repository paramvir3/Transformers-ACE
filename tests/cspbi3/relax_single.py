#!/usr/bin/env python3
"""Relax one periodic structure with a Transformers-ACE checkpoint."""

import argparse
from pathlib import Path
import sys

import numpy as np
from ase import Atoms
from ase.build.tools import sort
from ase.calculators.calculator import Calculator
from ase.constraints import FixSymmetry
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import LBFGS
from ase.spacegroup.symmetrize import check_symmetry

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (parent for parent in SCRIPT_PATH.parents if (parent / "transformers_ace" / "__init__.py").is_file()),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError(
        "Could not locate the Transformers-ACE repository. Keep this script inside "
        "the repository tree, or install the transformers-ace package first."
    )
sys.path.insert(0, str(REPO_ROOT))

from transformers_ace import TransformersACECalculator


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, type=Path, help="Input CIF, POSCAR, VASP, or XYZ file")
    parser.add_argument("--model", required=True, type=Path, help="Transformers-ACE .pt checkpoint")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps, or omit for automatic")
    parser.add_argument("--repeat", nargs=3, type=int, default=(1, 1, 1), metavar=("NX", "NY", "NZ"))
    parser.add_argument("--fmax", type=float, default=0.005, help="Force threshold in eV/Angstrom")
    parser.add_argument("--steps", type=int, default=100000, help="Maximum optimizer steps")
    parser.add_argument("--hydrostatic", action="store_true", help="Allow only hydrostatic cell strain")
    parser.add_argument("--fix-symmetry", action="store_true", help="Preserve the initial space-group symmetry")
    parser.add_argument("--fixed-cell", action="store_true", help="Relax positions without changing the cell")
    parser.add_argument("--phase", default=None, help="Optional phase name for output filenames")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for outputs")
    return parser.parse_args()


def print_symmetry(label: str, atoms: Atoms):
    print(f"\n{label} symmetry at precision 1e-3")
    try:
        check_symmetry(atoms, 1.0e-3, verbose=True)
    except Exception as error:
        print(f"Symmetry analysis unavailable: {error}")


def optimize_structure(
    atoms_in: Atoms,
    calculator: Calculator,
    *,
    fix_symmetry: bool,
    hydrostatic_strain: bool,
    fixed_cell: bool,
    fmax: float,
    steps: int,
    logfile: Path,
):
    atoms = atoms_in.copy()
    atoms.calc = calculator

    if fix_symmetry:
        atoms.set_constraint(FixSymmetry(atoms))

    optimization_target = atoms
    if not fixed_cell:
        optimization_target = FrechetCellFilter(
            atoms,
            hydrostatic_strain=hydrostatic_strain,
        )

    optimizer = LBFGS(optimization_target, logfile=str(logfile))
    converged = optimizer.run(fmax=fmax, steps=steps)

    cell_diff = (atoms.cell.cellpar() / atoms_in.cell.cellpar() - 1.0) * 100.0
    forces = atoms.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max())

    print(f"\nOptimization converged: {converged}")
    print(f"Optimization steps:     {optimizer.nsteps}")
    print(f"Optimized cell:         {atoms.cell.cellpar()}")
    print(f"Cell difference (%):    {cell_diff}")
    print(f"Maximum force:          {max_force:.8f} eV/Angstrom")
    print(f"Potential energy:       {atoms.get_potential_energy():.12f} eV")
    print("Scaled positions:\n", atoms.get_scaled_positions())
    return atoms


def main():
    args = parse_args()
    structure_path = args.structure.expanduser().resolve()
    model_path = args.model.expanduser().resolve()

    if not structure_path.is_file():
        raise FileNotFoundError(f"Structure not found: {structure_path}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if min(args.repeat) < 1:
        raise ValueError("All --repeat values must be positive integers")

    atoms = read(structure_path)
    atoms.pbc = True
    atoms = sort(atoms.repeat(tuple(args.repeat)))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_name = args.phase or structure_path.stem
    initial_output = output_dir / f"{phase_name}_initial.vasp"
    relaxed_output = output_dir / f"{phase_name}_relaxed.vasp"
    log_path = output_dir / f"{phase_name}.log"

    write(initial_output, atoms, format="vasp", direct=True, sort=True)

    print(f"Loaded structure: {structure_path}")
    print(f"Atoms after repeat: {len(atoms)}")
    print(f"Initial cell: {atoms.cell.cellpar()}")
    print_symmetry("Initial", atoms)

    calculator = TransformersACECalculator(model_path=str(model_path), device=args.device)
    relaxed = optimize_structure(
        atoms,
        calculator,
        fix_symmetry=args.fix_symmetry,
        hydrostatic_strain=args.hydrostatic,
        fixed_cell=args.fixed_cell,
        fmax=args.fmax,
        steps=args.steps,
        logfile=log_path,
    )

    print_symmetry("Final", relaxed)
    relaxed = sort(relaxed)
    write(relaxed_output, relaxed, format="vasp", direct=True, sort=True)
    print(f"\nSaved initial structure: {initial_output}")
    print(f"Saved relaxed structure:  {relaxed_output}")
    print(f"Saved optimization log:   {log_path}")


if __name__ == "__main__":
    main()
