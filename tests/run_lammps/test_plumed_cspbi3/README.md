# PLUMED Delta-To-Perovskite CsPbI3 Test

This folder contains a standalone LAMMPS plus PLUMED biased-dynamics example
for testing the native `pair_style transformers_ace` interface on a
non-perovskite delta CsPbI3 to perovskite CsPbI3 transition coordinate.

The run uses the PLUMED `DSFTHREE` structure-factor reaction coordinate from
the custom PLUMED build and biases it with `OPES_EXPANDED`.

## Files

- `data.CPI`: 640-atom non-perovskite CsPbI3 starting structure.
- `in.transformers_ace`: LAMMPS input using `pair_style transformers_ace`,
  short NVT initialization, then long NPT biased dynamics with `fix plumed`.
- `plumed.dat`: PLUMED structure-factor collective variable and bias setup.
- `../model.transformers_ace.pt`: shared exported TorchScript model used by
  both the unbiased LAMMPS smoke test and this PLUMED example.

Generated files such as `COLVAR`, `DELTAFS`, `plumed.log`, trajectories,
restart files, and post-NVT `data.NVT`/`data.eq` files are ignored.

## Run

Build LAMMPS with both `pair_style transformers_ace` and PLUMED as described in
`../../../docs/LAMMPS.md`, then run:

```bash
cd tests/run_lammps/test_plumed_cspbi3
/path/to/lammps/build/lmp -in in.transformers_ace
```

The production section is intentionally long:

```lammps
run             50000000
```

For an installation check, reduce that line to a small value such as
`run 1000`.

The example assumes one MPI rank for the current native interface:

```bash
mpirun -np 1 /path/to/lammps/build/lmp -in in.transformers_ace
```
