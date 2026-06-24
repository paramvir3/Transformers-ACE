"""LAMMPS ``fix external`` force provider for Transformers-ACE.

This module intentionally implements a validation bridge, not a production
MPI-parallel pair style. LAMMPS owns the MD loop and optional PLUMED bias;
Transformers-ACE supplies conservative forces, total energy, and global virial
from the same scalar energy used by the ASE calculator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from transformers_ace import TransformersACECalculator


def lammps_box_to_ase_cell(
    boxlo: Sequence[float],
    boxhi: Sequence[float],
    xy: float,
    yz: float,
    xz: float,
) -> np.ndarray:
    """Convert a LAMMPS restricted triclinic box into ASE cell vectors.

    LAMMPS stores restricted triclinic boxes as ``lx, ly, lz, xy, xz, yz``.
    ASE stores the three Cartesian cell vectors as rows. The returned cell is
    therefore ``a=(lx,0,0)``, ``b=(xy,ly,0)``, and ``c=(xz,yz,lz)``.
    """

    lo = np.asarray(boxlo, dtype=float)
    hi = np.asarray(boxhi, dtype=float)
    lengths = hi - lo
    return np.array(
        [
            [lengths[0], 0.0, 0.0],
            [float(xy), lengths[1], 0.0],
            [float(xz), float(yz), lengths[2]],
        ],
        dtype=float,
    )


def ase_stress_to_lammps_virial(stress_voigt: Sequence[float], volume: float) -> list[float]:
    """Convert ASE stress to the global virial expected by LAMMPS.

    ASE stress is intensive and uses Voigt order ``xx, yy, zz, yz, xz, xy``.
    LAMMPS ``fix external`` expects an extensive virial in order
    ``xx, yy, zz, xy, xz, yz``. LAMMPS pressure uses ``+ virial / V`` while ASE
    pressure is ``- mean(stress)``, so the sign conversion is ``virial=-V*sigma``.
    """

    s = np.asarray(stress_voigt, dtype=float)
    if s.shape != (6,):
        raise ValueError(f"Expected 6 ASE Voigt stress components, got shape {s.shape}")
    v = -float(volume) * np.array([s[0], s[1], s[2], s[5], s[4], s[3]], dtype=float)
    return v.tolist()


def _external_callback(caller, ntimestep, nlocal, tag, x, fexternal):
    caller.compute(ntimestep, nlocal, tag, x, fexternal)


@dataclass
class TransformersACEExternal:
    """Attach a Transformers-ACE model to LAMMPS through ``fix external``.

    Parameters
    ----------
    lmp
        A ``lammps.lammps`` Python object.
    model_path
        Path to a Transformers-ACE ``.pt`` checkpoint.
    type_map
        Chemical symbols in LAMMPS type-ID order. For CsPbI3 use
        ``("Cs", "Pb", "I")`` if type 1 is Cs, type 2 is Pb, and type 3 is I.
    fix_id
        ID of the LAMMPS ``fix external`` instance.
    require_single_rank
        Keep true for this validation bridge. A production pair style should
        use LAMMPS neighbor lists and ghost atoms directly.
    """

    lmp: Any
    model_path: str | Path
    type_map: Sequence[str]
    fix_id: str = "tace"
    device: Optional[str] = "cpu"
    require_single_rank: bool = True
    compute_virial: bool = True
    calculator: TransformersACECalculator = field(init=False)
    _atomic_numbers_by_type: np.ndarray = field(init=False, repr=False)
    last_energy: Optional[float] = field(default=None, init=False)
    last_max_force: Optional[float] = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.type_map:
            raise ValueError("type_map must contain at least one chemical symbol")
        try:
            numbers = [atomic_numbers[symbol] for symbol in self.type_map]
        except KeyError as exc:
            raise ValueError(f"Unknown chemical symbol in type_map: {exc}") from exc

        self.model_path = Path(self.model_path).expanduser().resolve()
        self._atomic_numbers_by_type = np.asarray(numbers, dtype=int)
        self.calculator = TransformersACECalculator(
            model_path=str(self.model_path),
            device=self.device,
        )

    def attach(self) -> None:
        """Register this object as the callback for the LAMMPS fix."""

        self.lmp.set_fix_external_callback(self.fix_id, _external_callback, self)

    def compute(self, ntimestep, nlocal, tag, x, fexternal) -> None:
        """Compute model forces for the current LAMMPS coordinates."""

        natoms = int(self.lmp.get_natoms())
        if self.require_single_rank and int(nlocal) != natoms:
            raise RuntimeError(
                "The current Transformers-ACE fix-external bridge is single-rank only: "
                f"nlocal={nlocal}, natoms={natoms}. Run with one MPI rank for validation, "
                "or implement the production pair style with ghost atoms."
            )

        atom_types = np.asarray(
            self.lmp.numpy.extract_atom("type", nelem=int(nlocal))[: int(nlocal)],
            dtype=int,
        )
        if atom_types.size != int(nlocal):
            raise RuntimeError("Could not read all local LAMMPS atom types")
        if atom_types.min(initial=1) < 1 or atom_types.max(initial=1) > len(self.type_map):
            raise ValueError(
                "LAMMPS atom type outside provided type_map: "
                f"found {atom_types.min()}..{atom_types.max()}, "
                f"type_map has {len(self.type_map)} entries"
            )

        positions = np.asarray(x[: int(nlocal), :], dtype=float).copy()
        numbers = self._atomic_numbers_by_type[atom_types - 1]
        boxlo, boxhi, xy, yz, xz, periodicity, _box_change = self.lmp.extract_box()
        cell = lammps_box_to_ase_cell(boxlo, boxhi, xy=xy, yz=yz, xz=xz)
        pbc = tuple(bool(v) for v in periodicity)

        atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=pbc)
        atoms.calc = self.calculator

        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        if forces.shape != (int(nlocal), 3):
            raise RuntimeError(f"Unexpected force array shape: {forces.shape}")

        fexternal[: int(nlocal), :] = forces
        self.lmp.fix_external_set_energy_global(self.fix_id, energy)

        if self.compute_virial:
            stress = atoms.get_stress()
            virial = ase_stress_to_lammps_virial(stress, atoms.get_volume())
            self.lmp.fix_external_set_virial_global(self.fix_id, virial)

        self.last_energy = energy
        self.last_max_force = float(np.linalg.norm(forces, axis=1).max(initial=0.0))
