#!/usr/bin/env python3
"""Plot CsPbI3 phase energies from evaluation CSV or relaxation outputs."""

import argparse
import csv
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ase.io import read


BASE_DIR = Path(__file__).resolve().parent
EV_PER_FU_TO_KJ_PER_MOL = 96.4853321233
RELAXED_PHASES = [
    ("edge_sharing_delta_phase", r"$\delta$", "delta"),
    ("orthorhombic_gamma_phase", r"$\gamma$", "ortho"),
    ("tetragonal_beta_phase", r"$\beta$", "beta"),
    ("cubic_alpha_phase", r"$\alpha$", "cubic"),
]
LABELS = {
    "edge_sharing_delta_phase": r"$\delta$ (edge)",
    "face_sharing_delta_phase": r"$\delta$ (face)",
    "orthorhombic_gamma_phase": r"$\gamma$",
    "tetragonal_beta_phase": r"$\beta$",
    "cubic_alpha_phase": r"$\alpha$",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        help="phase_energies.csv produced by evaluate_phases.py",
    )
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR)
    return parser.parse_args()


def extract_energy(output_path: Path) -> float:
    """Read the last reported optimized energy from a relaxation output."""
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?)"
    patterns = [
        re.compile(rf"Potential energy:\s*{number}\s*eV"),
        re.compile(rf"Epot after opt:\s*{number}(?:\s*eV)?"),
    ]
    matches = []
    for line in output_path.read_text().splitlines():
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                matches.append(float(match.group(1)))
                break
    if not matches:
        raise ValueError(f"No optimized energy found in {output_path}")
    return matches[-1]


def cspbi3_formula_units(structure_path: Path) -> int:
    atoms = read(structure_path, format="vasp")
    symbols = atoms.get_chemical_symbols()
    counts = {symbol: symbols.count(symbol) for symbol in set(symbols)}
    if set(counts) != {"Cs", "Pb", "I"}:
        raise ValueError(f"Expected Cs, Pb, and I in {structure_path}; found {counts}")
    if counts["Cs"] != counts["Pb"] or counts["I"] != 3 * counts["Cs"]:
        raise ValueError(f"Expected stoichiometric CsPbI3 in {structure_path}; found {counts}")
    return counts["Cs"]


def rows_from_relaxation_outputs():
    rows = []
    for phase, label, directory in RELAXED_PHASES:
        phase_dir = BASE_DIR / directory
        output_path = phase_dir / "output"
        structure_path = phase_dir / "POSCAR_relaxed"
        if not output_path.is_file() or not structure_path.is_file():
            raise FileNotFoundError(
                "Relaxation outputs are incomplete. Run the individual relaxations "
                "or pass --csv results/phase_energies.csv."
            )
        total_energy = extract_energy(output_path)
        n_formula_units = cspbi3_formula_units(structure_path)
        rows.append(
            {
                "phase": phase,
                "label": label,
                "total_energy_eV": total_energy,
                "n_formula_units": n_formula_units,
                "energy_eV_per_fu": total_energy / n_formula_units,
            }
        )

    reference = rows[0]["energy_eV_per_fu"]
    for row in rows:
        relative = row["energy_eV_per_fu"] - reference
        row["relative_energy_eV_per_fu"] = relative
        row["relative_energy_kJ_per_mol"] = relative * EV_PER_FU_TO_KJ_PER_MOL
    return rows


def rows_from_csv(csv_path: Path):
    with csv_path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows:
        raise ValueError(f"No phase rows found in {csv_path}")

    rows = []
    for source in source_rows:
        phase = source["phase"]
        relative_kj = source.get("relative_kJ_per_mol") or source.get("relative_energy_kJ_per_mol")
        if relative_kj is None:
            relative_ev = float(source["relative_eV_per_fu"])
            relative_kj = relative_ev * EV_PER_FU_TO_KJ_PER_MOL
        rows.append(
            {
                "phase": phase,
                "label": LABELS.get(phase, phase.replace("_", " ")),
                "total_energy_eV": float(source.get("total_energy_eV", "nan")),
                "n_formula_units": int(source["n_formula_units"]),
                "energy_eV_per_fu": float(source["energy_eV_per_fu"]),
                "relative_energy_eV_per_fu": float(source["relative_eV_per_fu"]),
                "relative_energy_kJ_per_mol": float(relative_kj),
            }
        )
    return rows


def main():
    args = parse_args()
    if args.csv is None:
        rows = rows_from_relaxation_outputs()
    else:
        rows = rows_from_csv(args.csv.expanduser().resolve())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "relative_energies.csv"
    table_fields = [name for name in rows[0] if name != "label"]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({name: row[name] for name in table_fields} for row in rows)

    labels = [row["label"] for row in rows]
    relative_kj = [row["relative_energy_kJ_per_mol"] for row in rows]
    colors = ["#5DA5DA", "#F15854", "#60BD68", "#FAA43A", "#B276B2"]

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    bars = axis.bar(labels, relative_kj, color=colors[: len(rows)])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("CsPbI$_3$ polymorph")
    axis.set_ylabel("Relative energy (kJ/mol)")
    axis.set_title("Transformers-ACE relative phase energies")
    axis.grid(axis="y", linestyle=":", alpha=0.5)
    for bar, value in zip(bars, relative_kj):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )

    png_path = args.output_dir / "rE.png"
    pdf_path = args.output_dir / "rE.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nRelative energies")
    print(f"{'phase':32s} {'E (eV/f.u.)':>14s} {'dE (kJ/mol)':>14s}")
    for row in rows:
        print(
            f"{row['phase']:32s} {row['energy_eV_per_fu']:14.7f} "
            f"{row['relative_energy_kJ_per_mol']:14.4f}"
        )
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {table_path}")


if __name__ == "__main__":
    main()
