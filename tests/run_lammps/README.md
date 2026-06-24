# LAMMPS CsPbI3 Tests

This folder contains standalone LAMMPS tests for the native
`pair_style transformers_ace` interface. It mirrors the workflow used by
NequIP/Allegro/MACE-style LAMMPS plugins:

```lammps
pair_style      transformers_ace
pair_coeff      * * ../model.transformers_ace.pt Cs Pb I
```

The included run was tested with LAMMPS `30 Mar 2026 - Development` on one MPI
rank using `newton off`.

## Unbiased Smoke Test

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

The tested PLUMED build path is:

```bash
git clone https://github.com/paramvir3/plumed2.git
cd plumed2
plumed_dir="${PWD}"
./configure --enable-modules=all --prefix="${PWD}"
make -j4
make install
source "${PWD}/sourceme.sh"
export PKG_CONFIG_PATH="${plumed_dir}/lib/pkgconfig:${PKG_CONFIG_PATH}"
```

## PLUMED Rare-Event Test

`test_plumed_cspbi3/` contains an explicit biased-dynamics example for a
non-perovskite delta CsPbI3 to perovskite CsPbI3 transition. It uses the same
shared `model.transformers_ace.pt`, a 640-atom starting structure, and a PLUMED
`DSFTHREE` structure-factor collective variable:

```bash
cd tests/run_lammps/test_plumed_cspbi3
/path/to/lammps/build/lmp -in in.transformers_ace
```

See `test_plumed_cspbi3/README.md` for the tracked inputs and ignored generated
outputs.
