# LAMMPS CsPbI3 Smoke Test

This folder contains a standalone LAMMPS smoke test for the native
`pair_style transformers_ace` interface. It mirrors the workflow used by
NequIP/Allegro/MACE-style LAMMPS plugins:

```lammps
pair_style      transformers_ace
pair_coeff      * * ../model.transformers_ace.pt Cs Pb I
```

The included run was tested with LAMMPS `30 Mar 2026 - Development` on one MPI
rank using `newton off`.

## Files

- `model.transformers_ace.pt`: exported TorchScript model for LAMMPS/LibTorch.
- `test_lammps_cspbi3/data.CPI`: 640-atom CsPbI3 starting structure.
- `test_lammps_cspbi3/in.transformers_ace`: NVT smoke test followed by a long
  NPT section that can be shortened for quick checks.

Generated LAMMPS outputs such as trajectories, restart files, `data.NVT`, and
`log.lammps` are intentionally ignored.

## Run

Patch and build LAMMPS as described in `../../docs/LAMMPS.md`, then run:

```bash
cd tests/run_lammps/test_lammps_cspbi3
/path/to/lammps/build/lmp -in in.transformers_ace
```

For a quick test, reduce the long NPT command in `in.transformers_ace` from
`run 50000000` to a small value such as `run 1000`.

For PLUMED rare-event dynamics, build LAMMPS with PLUMED and add your usual
PLUMED line to the input:

```lammps
fix plm all plumed plumedfile plumed.dat outfile plumed.out
```
