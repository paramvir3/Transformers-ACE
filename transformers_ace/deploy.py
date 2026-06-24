"""Export Transformers-ACE checkpoints for native LAMMPS/LibTorch inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list

from transformers_ace import TransformersACECalculator


class LAMMPSEnergyModel(nn.Module):
    """Energy-only deploy wrapper for native LAMMPS pair styles.

    The wrapped graph returns one scalar energy. LAMMPS/LibTorch differentiates
    that energy with respect to positions and a symmetric strain tensor to
    obtain conservative forces and virial.
    """

    def __init__(
        self,
        model: nn.Module,
        atomic_energy_tensor: torch.Tensor | None = None,
        energy_shift_per_atom: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(model.hidden_dim)
        self.emb = model.emb
        self.ace = model.ace
        self.layers = model.layers
        self.readout = model.readout
        self.energy_shift_per_atom = float(energy_shift_per_atom)
        if atomic_energy_tensor is None:
            self.register_buffer("atomic_energy_tensor", torch.empty(0))
        else:
            self.register_buffer("atomic_energy_tensor", atomic_energy_tensor.detach().float())

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        cell: torch.Tensor,
        edge_index: torch.Tensor,
        edge_shift: torch.Tensor,
        strain: torch.Tensor,
        local_mask: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = torch.zeros((3, 3), dtype=pos.dtype, device=pos.device)
        epsilon[0, 0] = strain[0]
        epsilon[1, 1] = strain[1]
        epsilon[2, 2] = strain[2]
        epsilon[0, 1] = strain[3]
        epsilon[1, 0] = strain[3]
        epsilon[0, 2] = strain[4]
        epsilon[2, 0] = strain[4]
        epsilon[1, 2] = strain[5]
        epsilon[2, 1] = strain[5]

        deformation = torch.eye(3, dtype=pos.dtype, device=pos.device) + epsilon
        pos_deformed = pos @ deformation
        cell_deformed = cell @ deformation

        edge_vec = (
            pos_deformed[edge_index[0]]
            - pos_deformed[edge_index[1]]
            + edge_shift.to(dtype=pos.dtype) @ cell_deformed
        )
        edge_len = torch.norm(edge_vec, dim=1)

        h, edge_features, cutoff = self.ace(
            self.emb(z),
            edge_index,
            edge_vec,
            edge_len,
            return_edge_features=True,
        )
        receiver = edge_index[1]
        for layer in self.layers:
            h = layer(
                h,
                edge_features,
                receiver,
                edge_len,
                cutoff,
                temperature_scale=1.0,
            )

        atomic_energy = self.readout(h[:, : self.hidden_dim]).view(-1)
        mask = local_mask.to(dtype=atomic_energy.dtype)
        energy = torch.sum(atomic_energy * mask)

        if self.atomic_energy_tensor.numel() > 0:
            baseline = self.atomic_energy_tensor[z].to(dtype=atomic_energy.dtype)
            energy = energy + torch.sum(baseline * mask)
        else:
            energy = energy + self.energy_shift_per_atom * torch.sum(mask)
        return energy.reshape(())


def _synthetic_atoms(type_map: Sequence[str], cutoff: float) -> Atoms:
    numbers = [atomic_numbers[symbol] for symbol in type_map]
    spacing = min(max(0.35 * cutoff, 1.5), 3.0)
    positions = [[idx * spacing, 0.2 * (idx % 2), 0.15 * (idx % 3)] for idx in range(len(numbers))]
    cell_length = max(cutoff * 3.0, spacing * (len(numbers) + 2))
    return Atoms(numbers=numbers, positions=positions, cell=[cell_length] * 3, pbc=True)


def _example_tensors(atoms: Atoms, cutoff: float):
    atoms = atoms.copy()
    atoms.pbc = True
    i, j, shifts = neighbor_list("ijS", atoms, cutoff)
    if len(i) == 0:
        raise ValueError("Export example contains no neighbor edges; use a denser example structure")

    z = torch.tensor(atoms.numbers, dtype=torch.long)
    pos = torch.tensor(atoms.positions, dtype=torch.float32)
    cell = torch.tensor(atoms.cell.array, dtype=torch.float32)
    edge_index = torch.stack(
        [torch.tensor(j, dtype=torch.long), torch.tensor(i, dtype=torch.long)],
        dim=0,
    )
    edge_shift = torch.tensor(shifts, dtype=torch.float32)
    strain = torch.zeros(6, dtype=torch.float32)
    local_mask = torch.ones(len(atoms), dtype=torch.float32)
    return z, pos, cell, edge_index, edge_shift, strain, local_mask


def _metadata_text(type_map: Sequence[str], atomic_numbers_for_types: Sequence[int], cutoff: float) -> str:
    return "\n".join(
        [
            "format=transformers_ace_lammps_v1",
            "units=metal",
            f"r_max={float(cutoff):.12g}",
            "type_symbols=" + " ".join(type_map),
            "type_atomic_numbers=" + " ".join(str(int(z)) for z in atomic_numbers_for_types),
            "energy_output=scalar_eV",
            "force_convention=forces_from_negative_position_gradient",
            "virial_convention=minus_diagonal_strain_gradient_minus_half_shear_gradient",
            "",
        ]
    )


def export_lammps_model(
    checkpoint: Path,
    output: Path,
    type_map: Sequence[str],
    example_structure: Path | None = None,
    device: str = "cpu",
) -> None:
    calculator = TransformersACECalculator(model_path=str(checkpoint), device=device)
    model = calculator.model.eval()
    if int(getattr(model, "architecture_version", 0)) != 2:
        raise ValueError("Native LAMMPS export currently supports architecture_version=2 checkpoints")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    atomic_energy_tensor = calculator.atomic_energy_tensor
    if atomic_energy_tensor is not None:
        atomic_energy_tensor = atomic_energy_tensor.detach().cpu()

    deploy_model = LAMMPSEnergyModel(
        model.cpu(),
        atomic_energy_tensor=atomic_energy_tensor,
        energy_shift_per_atom=calculator.energy_shift_per_atom,
    ).eval()

    if example_structure is None:
        atoms = _synthetic_atoms(type_map, calculator.r_max)
    else:
        atoms = read(example_structure.expanduser())
    example_inputs = _example_tensors(atoms, calculator.r_max)

    traced = torch.jit.trace(deploy_model, example_inputs, check_trace=False)
    metadata = _metadata_text(
        type_map=type_map,
        atomic_numbers_for_types=[atomic_numbers[symbol] for symbol in type_map],
        cutoff=calculator.r_max,
    )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output), _extra_files={"metadata.txt": metadata})
    print(f"Wrote LAMMPS TorchScript model: {output}")
    print(metadata, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="Training checkpoint, e.g. model.pt")
    parser.add_argument("--output", required=True, type=Path, help="Output .transformers_ace.pt file")
    parser.add_argument("--type-map", nargs="+", required=True, help="Model type symbols, e.g. Cs Pb I")
    parser.add_argument("--example-structure", type=Path, default=None, help="Optional ASE-readable trace example")
    parser.add_argument("--device", default="cpu", help="Device used while exporting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_lammps_model(
        checkpoint=args.checkpoint.expanduser().resolve(),
        output=args.output,
        type_map=args.type_map,
        example_structure=args.example_structure,
        device=args.device,
    )


if __name__ == "__main__":
    main()
