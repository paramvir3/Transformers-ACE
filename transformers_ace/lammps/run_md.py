"""Run LAMMPS MD with Transformers-ACE through ``fix external``."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from ase.data import atomic_masses, atomic_numbers

from flashace.checkpoint import load_checkpoint

from .external import TransformersACEExternal


def _quote_path(path: Path) -> str:
    text = str(path.expanduser().resolve())
    return f'"{text}"' if any(ch.isspace() for ch in text) else text


def _model_cutoff(model_path: Path) -> float:
    checkpoint = load_checkpoint(str(model_path.expanduser().resolve()), map_location="cpu")
    try:
        return float(checkpoint["config"]["r_max"])
    except KeyError as exc:
        raise KeyError("Checkpoint does not contain config.r_max; cannot set pair_style zero cutoff") from exc


def _commands(lmp, lines: Iterable[str]) -> None:
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            lmp.command(stripped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Transformers-ACE checkpoint")
    parser.add_argument("--data", required=True, type=Path, help="LAMMPS atomic data file")
    parser.add_argument(
        "--type-map",
        nargs="+",
        required=True,
        help="LAMMPS atom-type order, e.g. --type-map Cs Pb I",
    )
    parser.add_argument("--device", default="cpu", help="Torch device for Transformers-ACE")
    parser.add_argument("--steps", type=int, default=10000, help="Number of MD steps")
    parser.add_argument("--timestep", type=float, default=0.001, help="LAMMPS timestep in ps for metal units")
    parser.add_argument("--temperature", type=float, default=300.0, help="Target temperature in K")
    parser.add_argument("--tdamp", type=float, default=0.1, help="NVT thermostat damping in ps")
    parser.add_argument("--ensemble", choices=["nve", "nvt"], default="nvt")
    parser.add_argument("--seed", type=int, default=12345, help="Velocity seed")
    parser.add_argument("--no-create-velocity", action="store_true", help="Keep velocities from the data file")
    parser.add_argument("--plumed", type=Path, default=None, help="Optional PLUMED input file")
    parser.add_argument("--plumed-out", type=Path, default=Path("plumed.out"), help="PLUMED output file")
    parser.add_argument("--dump", type=Path, default=Path("traj.lammpstrj"), help="LAMMPS trajectory dump")
    parser.add_argument("--dump-every", type=int, default=100, help="Trajectory dump stride")
    parser.add_argument("--thermo", type=int, default=10, help="Thermo output stride")
    parser.add_argument("--log", type=Path, default=Path("log.transformers_ace_lammps"))
    parser.add_argument("--fix-id", default="tace", help="LAMMPS fix external ID")
    parser.add_argument("--cutoff", type=float, default=None, help="Pair zero cutoff. Defaults to checkpoint r_max")
    parser.add_argument("--neighbor-skin", type=float, default=2.0, help="LAMMPS neighbor skin in Angstrom")
    parser.add_argument("--no-virial", action="store_true", help="Do not pass stress-derived virial to LAMMPS")
    parser.add_argument(
        "--extra-input",
        type=Path,
        default=None,
        help="Optional LAMMPS command fragment inserted before the run command",
    )
    parser.add_argument("--check-only", action="store_true", help="Initialize and run 0 steps only")
    parser.add_argument(
        "--lammps-name",
        default="",
        help="LAMMPS shared-library suffix passed to lammps(name=...). Usually leave empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    args.data = args.data.expanduser().resolve()
    cutoff = float(args.cutoff) if args.cutoff is not None else _model_cutoff(args.model)

    try:
        from lammps import lammps
    except ImportError as exc:
        raise ImportError(
            "Could not import the LAMMPS Python module. Build/install LAMMPS as a "
            "shared library with its Python module, then run this command from the "
            "same Python environment."
        ) from exc

    lmp = lammps(name=args.lammps_name, cmdargs=["-log", str(args.log.expanduser())])
    try:
        if hasattr(lmp, "has_style"):
            if not lmp.has_style("fix", "external"):
                raise RuntimeError("This LAMMPS build does not provide fix external")
            if not lmp.has_style("pair", "zero"):
                raise RuntimeError("This LAMMPS build does not provide pair_style zero")
            if args.plumed is not None and not lmp.has_style("fix", "plumed"):
                raise RuntimeError("This LAMMPS build does not provide fix plumed")

        _commands(
            lmp,
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                f"read_data {_quote_path(args.data)}",
                f"pair_style zero {cutoff:.12g}",
                "pair_coeff * *",
                f"neighbor {args.neighbor_skin:.12g} bin",
                "neigh_modify every 1 delay 0 check yes",
            ],
        )

        for type_id, symbol in enumerate(args.type_map, start=1):
            try:
                mass = float(atomic_masses[atomic_numbers[symbol]])
            except KeyError as exc:
                raise ValueError(f"Unknown chemical symbol in type_map: {symbol}") from exc
            lmp.command(f"mass {type_id} {mass:.12g}")

        lmp.command(f"fix {args.fix_id} all external pf/callback 1 1")
        lmp.command(f"fix_modify {args.fix_id} energy yes virial {'no' if args.no_virial else 'yes'}")

        force_provider = TransformersACEExternal(
            lmp=lmp,
            model_path=args.model,
            type_map=args.type_map,
            fix_id=args.fix_id,
            device=args.device,
            compute_virial=not args.no_virial,
        )
        force_provider.attach()

        if not args.no_create_velocity:
            lmp.command(
                f"velocity all create {args.temperature:.12g} {args.seed} "
                "mom yes rot yes dist gaussian"
            )

        if args.ensemble == "nve":
            lmp.command("fix int all nve")
        else:
            lmp.command(
                f"fix int all nvt temp {args.temperature:.12g} "
                f"{args.temperature:.12g} {args.tdamp:.12g}"
            )

        if args.plumed is not None:
            lmp.command(
                f"fix plm all plumed plumedfile {_quote_path(args.plumed)} "
                f"outfile {_quote_path(args.plumed_out)}"
            )

        if args.dump_every > 0:
            lmp.command(
                f"dump trj all custom {args.dump_every} {_quote_path(args.dump)} "
                "id type x y z fx fy fz"
            )
            lmp.command("dump_modify trj sort id")

        lmp.command(f"timestep {args.timestep:.12g}")
        lmp.command(f"thermo_style custom step time temp pe etotal press vol f_{args.fix_id}")
        lmp.command(f"thermo {args.thermo}")

        if args.extra_input is not None:
            lmp.file(str(args.extra_input.expanduser().resolve()))

        if args.check_only:
            lmp.command("run 0")
        else:
            lmp.command(f"run {args.steps}")

        if force_provider.last_energy is not None:
            print(
                "Last Transformers-ACE callback: "
                f"E={force_provider.last_energy:.12g} eV, "
                f"max|F|={force_provider.last_max_force:.6g} eV/Angstrom"
            )
    finally:
        lmp.close()


if __name__ == "__main__":
    main()
