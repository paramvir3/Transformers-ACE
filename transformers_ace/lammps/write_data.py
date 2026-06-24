"""Convert ASE-readable structures to LAMMPS data files for Transformers-ACE."""

from __future__ import annotations

import argparse
from pathlib import Path

from ase.build import sort
from ase.io import read, write


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", required=True, type=Path, help="ASE-readable input structure")
    parser.add_argument("--output", required=True, type=Path, help="LAMMPS data file to write")
    parser.add_argument(
        "--type-map",
        nargs="+",
        required=True,
        help="LAMMPS atom-type order, e.g. --type-map Cs Pb I",
    )
    parser.add_argument(
        "--repeat",
        nargs=3,
        type=int,
        default=(1, 1, 1),
        metavar=("NX", "NY", "NZ"),
        help="Repeat the input cell before writing",
    )
    parser.add_argument("--no-sort", action="store_true", help="Do not sort atoms by chemical symbol")
    parser.add_argument(
        "--format",
        default=None,
        help="Optional ASE input format. Leave unset for automatic detection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    atoms = read(args.structure.expanduser(), format=args.format)
    atoms.pbc = True
    atoms = atoms.repeat(tuple(args.repeat))
    if not args.no_sort:
        atoms = sort(atoms)

    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    write(
        args.output.expanduser(),
        atoms,
        format="lammps-data",
        specorder=list(args.type_map),
        masses=True,
        atom_style="atomic",
        units="metal",
        force_skew=True,
    )

    counts = {symbol: atoms.get_chemical_symbols().count(symbol) for symbol in args.type_map}
    print(f"Wrote {args.output} with {len(atoms)} atoms")
    print(f"LAMMPS type map: {' '.join(args.type_map)}")
    print("Counts:", ", ".join(f"{symbol}={counts[symbol]}" for symbol in args.type_map))


if __name__ == "__main__":
    main()
