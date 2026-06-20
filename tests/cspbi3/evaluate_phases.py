#!/usr/bin/env python3
"""Evaluate relative energies of periodic CsPbI3 polymorphs with Transformers-ACE."""

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers_ace import TransformersACECalculator


PHASE_FILES = {
    "edge_sharing_delta_phase": "edge_sharing_delta_phase.vasp",
    "face_sharing_delta_phase": "face_sharing_delta_phase.vasp",
    "orthorhombic_gamma_phase": "orthorhombic_gamma_phase.vasp",
    "tetragonal_beta_phase": "tetragonal_beta_phase.vasp",
    "cubic_alpha_phase": "cubic_alpha_phase.vasp",
}
EV_PER_FU_TO_KJ_PER_MOL = 96.4853321233


def parse_args():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Transformers-ACE .pt checkpoint")
    parser.add_argument("--device", default=None, help="cpu, cuda, mps, or omit for automatic")
    parser.add_argument("--structures", type=Path, default=base / "structures")
    parser.add_argument("--output", type=Path, default=base / "results")
    parser.add_argument(
        "--reference",
        default="edge_sharing_delta_phase",
        choices=[*PHASE_FILES, "minimum"],
        help="Zero of relative energy",
    )
    parser.add_argument("--relax", action="store_true", help="Relax atomic positions")
    parser.add_argument(
        "--relax-cell",
        action="store_true",
        help="Relax positions and cell (implies --relax)",
    )
    parser.add_argument("--fmax", type=float, default=0.03, help="Force convergence in eV/Angstrom")
    parser.add_argument("--steps", type=int, default=500, help="Maximum relaxation steps")
    return parser.parse_args()


def formula_units(atoms):
    symbols = atoms.get_chemical_symbols()
    counts = {symbol: symbols.count(symbol) for symbol in set(symbols)}
    expected = {"Cs", "Pb", "I"}
    if set(counts) != expected or counts["Cs"] != counts["Pb"] or counts["I"] != 3 * counts["Cs"]:
        raise ValueError(f"Expected stoichiometric CsPbI3, found {counts}")
    return counts["Cs"]


def relax_structure(atoms, phase, args):
    relaxed_dir = args.output / "relaxed"
    relaxed_dir.mkdir(parents=True, exist_ok=True)
    target = FrechetCellFilter(atoms) if args.relax_cell else atoms
    optimizer = FIRE(target, logfile=str(relaxed_dir / f"{phase}.log"))
    converged = optimizer.run(fmax=args.fmax, steps=args.steps)
    write(relaxed_dir / f"{phase}.vasp", atoms, format="vasp", direct=True, sort=True)
    return bool(converged), optimizer.nsteps


def main():
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    args.structures = args.structures.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if not args.model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.model}")
    if args.relax_cell:
        args.relax = True

    args.output.mkdir(parents=True, exist_ok=True)
    calculator = TransformersACECalculator(model_path=str(args.model), device=args.device)
    rows = []

    for phase, filename in PHASE_FILES.items():
        structure_path = args.structures / filename
        try:
            structure_label = structure_path.relative_to(REPO_ROOT)
        except ValueError:
            structure_label = structure_path
        atoms = read(structure_path, format="vasp")
        atoms.pbc = True
        n_fu = formula_units(atoms)
        atoms.calc = calculator

        converged = None
        relaxation_steps = 0
        if args.relax:
            converged, relaxation_steps = relax_structure(atoms, phase, args)

        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max())
        rows.append(
            {
                "phase": phase,
                "structure": str(structure_label),
                "n_atoms": len(atoms),
                "n_formula_units": n_fu,
                "total_energy_eV": energy,
                "energy_eV_per_fu": energy / n_fu,
                "relative_eV_per_fu": 0.0,
                "relative_meV_per_fu": 0.0,
                "relative_kJ_per_mol": 0.0,
                "volume_A3_per_fu": atoms.get_volume() / n_fu,
                "max_force_eV_per_A": max_force,
                "relaxed": args.relax,
                "converged": converged,
                "relaxation_steps": relaxation_steps,
            }
        )

    if args.reference == "minimum":
        reference_row = min(rows, key=lambda row: row["energy_eV_per_fu"])
    else:
        reference_row = next(row for row in rows if row["phase"] == args.reference)
    reference_energy = reference_row["energy_eV_per_fu"]

    for row in rows:
        relative = row["energy_eV_per_fu"] - reference_energy
        row["relative_eV_per_fu"] = relative
        row["relative_meV_per_fu"] = 1000.0 * relative
        row["relative_kJ_per_mol"] = relative * EV_PER_FU_TO_KJ_PER_MOL

    rows.sort(key=lambda row: row["energy_eV_per_fu"])
    csv_path = args.output / "phase_energies.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReference: {reference_row['phase']}")
    print(f"{'phase':32s} {'E (eV/f.u.)':>14s} {'dE (meV/f.u.)':>16s} {'dE (kJ/mol)':>14s} {'max|F|':>10s}")
    for row in rows:
        print(
            f"{row['phase']:32s} {row['energy_eV_per_fu']:14.7f} "
            f"{row['relative_meV_per_fu']:16.3f} {row['relative_kJ_per_mol']:14.4f} "
            f"{row['max_force_eV_per_A']:10.4f}"
        )
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
