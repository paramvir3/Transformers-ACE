# LAMMPS Interface

Transformers-ACE has two LAMMPS routes:

- a native `pair_style transformers_ace` for standalone LAMMPS runs;
- a Python `fix external` bridge for rapid validation/debugging.

For standalone molecular dynamics, rare-event sampling, and large systems, use
the native pair style.

## Native Pair Style

The native pair style lets a LAMMPS input use:

```lammps
newton         on
pair_style      transformers_ace
pair_coeff      * * model.transformers_ace.pt Cs Pb I
```

For GPU runs, the optional device argument is:

```lammps
pair_style      transformers_ace device auto
pair_style      transformers_ace device cpu
pair_style      transformers_ace device cuda
pair_style      transformers_ace device cuda:0
```

`auto` is the production default. It uses CUDA when the LibTorch build can see
GPUs and maps each MPI process to `local_rank % visible_gpu_count`. In cluster
launchers, set `CUDA_VISIBLE_DEVICES` so each node exposes the GPUs assigned to
the job.

LAMMPS does not load the training checkpoint directly. First export a
TorchScript deploy model:

```bash
python -m transformers_ace.deploy \
  --checkpoint training/model.pt \
  --output model.transformers_ace.pt \
  --type-map Cs Pb I \
  --example-structure tests/cspbi3/structures/cubic_alpha_phase.vasp
```

Then patch and build LAMMPS:

```bash
git clone --depth=1 https://github.com/lammps/lammps
cd lammps

cd /path/to/Transformers-ACE/lammps/pair_style
bash patch_lammps.sh /path/to/lammps

cd /path/to/lammps
mkdir -p build
cd build
cmake ../cmake \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
make -j
```

## Native Pair Style With PLUMED

For rare-event dynamics, build the native `pair_style transformers_ace` together
with LAMMPS' PLUMED package. The sequence below is the tested macOS workflow
using the custom PLUMED repository with the `DSFTHREE` structure-factor
collective variable.

### 1. Install PLUMED

LAMMPS discovers an existing PLUMED build through `pkg-config`, so install it
first if it is missing:

```bash
brew install pkg-config
```

Then build and install PLUMED:

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

Optional persistent shell setup:

```bash
echo "source ${plumed_dir}/sourceme.sh" >> ~/.bashrc
echo "export PKG_CONFIG_PATH=\"${plumed_dir}/lib/pkgconfig:\${PKG_CONFIG_PATH}\"" >> ~/.bashrc
echo "export PYTHONPATH=\"${plumed_dir}/lib/plumed/python:\${PYTHONPATH}\"" >> ~/.bashrc
```

Check that PLUMED is visible:

```bash
pkg-config --modversion plumed
pkg-config --libs plumed
plumed --version
```

### 2. Patch and configure LAMMPS

Clone LAMMPS:

```bash
git clone --depth=1 https://github.com/lammps/lammps
cd lammps
```

Patch LAMMPS with the Transformers-ACE pair style:

```bash
cd /path/to/Transformers-ACE/lammps/pair_style
bash patch_lammps.sh /path/to/lammps
```

Configure LAMMPS from the same Python environment that has PyTorch installed:

```bash
cd /path/to/lammps
mkdir -p build
cd build
rm -rf CMakeCache.txt CMakeFiles

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DLAMMPS_EXCEPTIONS=yes \
  -DCMAKE_INSTALL_PREFIX="$(pwd)" \
  -DBUILD_MPI=ON \
  -DPKG_MANYBODY=yes \
  -DPKG_EXTRA-FIX=yes \
  -DPKG_EXTRA-PAIR=yes \
  -DPKG_EXTRA-DUMP=yes \
  -DPKG_MOLECULE=yes \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)');${plumed_dir}" \
  -DPKG_PLUMED=yes \
  -DPLUMED_MODE=shared \
  -DDOWNLOAD_PLUMED=no \
  ../cmake

make -j
```

If CMake cannot find `PkgConfig`, add this line to the `cmake` command:

```bash
-DPKG_CONFIG_EXECUTABLE=/opt/homebrew/bin/pkg-config
```

If CMake cannot find `plumed`, make sure `PKG_CONFIG_PATH` points to
`${plumed_dir}/lib/pkgconfig`, then rerun CMake from a clean cache.

The template input file is:

```text
lammps/pair_style/in.transformers_ace
```

Run it with:

```bash
/path/to/lammps/build/lmp -in in.transformers_ace
```

## CsPbI3 Standalone Test

A tested CsPbI3 LAMMPS input is included under:

```text
tests/run_lammps/test_lammps_cspbi3
```

It uses the native pair style and the exported model at:

```text
tests/run_lammps/model.transformers_ace.pt
```

Run it after patching/building LAMMPS:

```bash
cd tests/run_lammps/test_lammps_cspbi3
/path/to/lammps/build/lmp -in in.transformers_ace
```

The input first runs a short NVT smoke test. The following NPT section is long
by design for production-style testing; shorten `run 50000000` to `run 1000`
for a quick check.

For multi-GPU runs, launch one MPI rank per GPU:

```bash
cd tests/run_lammps/test_lammps_cspbi3
mpirun -np 4 /path/to/lammps/build/lmp -in in.transformers_ace
```

The native pair style evaluates the energy only for atoms owned by the current
MPI rank while including ghost atoms in the local neighbor graph. Gradients on
ghost coordinates are accumulated by LAMMPS reverse communication, so
`newton on` is required for parallel runs. Before production dynamics, compare
`run 0` and a short trajectory between one rank and the target MPI/GPU layout.
The total energy, forces, and virial should agree within floating-point
tolerance.

## CsPbI3 PLUMED Delta-To-Perovskite Test

A tested LAMMPS plus PLUMED biased-dynamics example is included under:

```text
tests/run_lammps/test_plumed_cspbi3
```

The example starts from a 640-atom non-perovskite delta CsPbI3 structure and
biases a PLUMED `DSFTHREE` structure-factor collective variable toward
perovskite CsPbI3. The LAMMPS input uses:

```lammps
pair_style      transformers_ace
pair_coeff      * * ../model.transformers_ace.pt Cs Pb I
fix             1 all plumed plumedfile plumed.dat outfile plumed.log
```

Run it after building LAMMPS with both Transformers-ACE and PLUMED:

```bash
cd tests/run_lammps/test_plumed_cspbi3
/path/to/lammps/build/lmp -in in.transformers_ace
```

The production section uses `run 50000000`. For an installation check, reduce
that value to `run 1000`.

## Python Validation Bridge

Transformers-ACE can also be tested inside LAMMPS through the LAMMPS
`fix external` callback interface. LAMMPS owns the molecular-dynamics loop,
thermostats, dumps, and optional PLUMED bias. Transformers-ACE supplies the
conservative model energy, forces, and global virial at each force evaluation.

This is a validation bridge for rare-event workflow testing. It is deliberately
single-MPI-rank at first because the current Python calculator builds its own
periodic neighbor list. For production MPI runs, use the compiled
`pair_style transformers_ace`, which consumes LAMMPS neighbor lists and ghost
atoms directly.

## Physics and Units

Use LAMMPS `units metal`, so the units match the trained checkpoint:

- energy: eV
- distance: Angstrom
- force: eV/Angstrom
- time: ps
- stress/pressure virial: eV

The callback evaluates one scalar potential energy,

```math
E = E_\theta(\mathbf R, \mathbf h),
```

then returns conservative forces,

```math
\mathbf F_i = -\frac{\partial E}{\partial \mathbf R_i}.
```

For variable-cell or pressure-aware runs, ASE stress is converted to the LAMMPS
global virial as

```math
\mathbf W = -V\boldsymbol\sigma.
```

The tensor order is also converted explicitly. ASE uses
`xx yy zz yz xz xy`; LAMMPS uses `xx yy zz xy xz yz`.

## Python Bridge Requirements

The Python validation bridge needs one Python environment containing:

```bash
python -m pip install -e .
```

and a LAMMPS build whose Python module can be imported from that same
environment:

```bash
python -c "from lammps import lammps; print(lammps().version())"
```

For biased rare-event dynamics with the bridge, the same LAMMPS build must also include
`fix plumed` and must be linked to your PLUMED installation.

## Convert a CsPbI3 Structure

From the repository root:

```bash
python -m transformers_ace.lammps.write_data \
  --structure tests/cspbi3/structures/cubic_alpha_phase.vasp \
  --output tests/cspbi3/lammps/cubic_alpha_2x2x2.data \
  --type-map Cs Pb I \
  --repeat 2 2 2
```

The `--type-map` order is important. It defines LAMMPS type 1, type 2, type 3,
etc. The same order must be passed to the MD driver.

## Python Bridge Force Check

Run zero MD steps first. This initializes LAMMPS, calls the Transformers-ACE
force callback once, and prints the model energy and maximum force.

```bash
python -m transformers_ace.lammps.run_md \
  --model training/model.pt \
  --data tests/cspbi3/lammps/cubic_alpha_2x2x2.data \
  --type-map Cs Pb I \
  --device cpu \
  --check-only
```

Use this before running long dynamics.

## Python Bridge Short NVT MD

```bash
python -m transformers_ace.lammps.run_md \
  --model training/model.pt \
  --data tests/cspbi3/lammps/cubic_alpha_2x2x2.data \
  --type-map Cs Pb I \
  --device cpu \
  --ensemble nvt \
  --temperature 300 \
  --timestep 0.001 \
  --steps 10000 \
  --dump tests/cspbi3/lammps/cubic_alpha_md.lammpstrj
```

The default timestep is 0.001 ps, i.e. 1 fs in LAMMPS metal units.

## Add PLUMED To The Python Bridge

Once unbiased MD is stable, add your PLUMED structure-factor input:

```bash
python -m transformers_ace.lammps.run_md \
  --model training/model.pt \
  --data tests/cspbi3/lammps/cubic_alpha_2x2x2.data \
  --type-map Cs Pb I \
  --device cpu \
  --ensemble nvt \
  --temperature 300 \
  --timestep 0.001 \
  --steps 10000 \
  --plumed plumed.dat \
  --plumed-out plumed.out
```

LAMMPS applies PLUMED bias forces in the same timestep loop as the
Transformers-ACE forces.

## Current Limitation

Run this validation bridge with one MPI rank:

```bash
mpirun -np 1 python -m transformers_ace.lammps.run_md ...
```

This is enough to test MD stability and PLUMED rare-event input files. For
large production simulations, the next step is a native compiled pair style
with LAMMPS ghost atoms and neighbor lists.
