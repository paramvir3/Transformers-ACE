# Native LAMMPS Pair Style

This directory contains the first native LibTorch pair style for running
Transformers-ACE from standalone LAMMPS:

```lammps
pair_style  transformers_ace
pair_coeff  * * model.transformers_ace.pt Cs Pb I
```

The pair style loads an exported TorchScript energy model. It does not use
Python during molecular dynamics.

## Export The Model

From the Transformers-ACE repository:

```bash
python -m transformers_ace.deploy \
  --checkpoint training/model.pt \
  --output model.transformers_ace.pt \
  --type-map Cs Pb I \
  --example-structure tests/cspbi3/structures/cubic_alpha_phase.vasp
```

The exported model returns only the scalar energy. The C++ pair style computes
forces and virial with LibTorch autograd:

```text
F_i = -dE/dR_i
W_xx, W_yy, W_zz = -dE/dstrain_xx, -dE/dstrain_yy, -dE/dstrain_zz
W_xy, W_xz, W_yz = -0.5 dE/dstrain_xy, -0.5 dE/dstrain_xz, -0.5 dE/dstrain_yz
```

## Patch LAMMPS

```bash
git clone --depth=1 https://github.com/lammps/lammps
cd /path/to/Transformers-ACE/lammps/pair_style
./patch_lammps.sh /path/to/lammps
```

Configure and build LAMMPS with LibTorch:

```bash
cd /path/to/lammps
mkdir -p build
cd build
cmake ../cmake \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
make -j
```

For PLUMED rare-event dynamics, build LAMMPS with your PLUMED-enabled setup as
usual; this pair style only replaces the force model.

## CsPbI3 Test

A working standalone test is included in:

```text
tests/run_lammps/test_lammps_cspbi3
```

After building LAMMPS:

```bash
cd /path/to/Transformers-ACE/tests/run_lammps/test_lammps_cspbi3
/path/to/lammps/build/lmp -in in.transformers_ace
```

## Run

```bash
/path/to/lammps/build/lmp -in in.transformers_ace
```

The first implementation is intentionally single-MPI-rank. It is meant to
validate the native LAMMPS route and PLUMED workflow before adding MPI domain
decomposition support.
