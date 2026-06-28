# Native LAMMPS Pair Style

This directory contains the first native LibTorch pair style for running
Transformers-ACE from standalone LAMMPS:

```lammps
pair_style  transformers_ace
pair_coeff  * * model.transformers_ace.pt Cs Pb I
```

For MPI or multi-GPU runs, keep LAMMPS `newton on` so force contributions on
ghost atoms are reverse-communicated to their owning ranks:

```lammps
newton      on
pair_style  transformers_ace
pair_coeff  * * model.transformers_ace.pt Cs Pb I
```

The optional device selector is:

```lammps
pair_style  transformers_ace device auto
pair_style  transformers_ace device cpu
pair_style  transformers_ace device cuda
pair_style  transformers_ace device cuda:0
```

`auto` uses CUDA when LibTorch sees GPUs and maps each MPI process to
`local_rank % visible_gpu_count`. Set `CUDA_VISIBLE_DEVICES` in the launcher to
control which GPUs are visible on each node.

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

For a multi-GPU node with one MPI rank per GPU:

```bash
mpirun -np 4 /path/to/lammps/build/lmp -in in.transformers_ace
```

Before production dynamics, compare `run 0` or a short NVE trajectory between
one rank and multiple ranks. Energies, forces, and virials should agree within
floating-point tolerance; larger differences usually indicate a neighbor-skin,
Newton, or GPU visibility problem.
