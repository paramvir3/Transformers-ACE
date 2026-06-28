# Transformers-ACE

Transformers-ACE is a research implementation of local equivariant transformer
potentials built on Atomic Cluster Expansion (ACE) descriptors. It predicts a
single invariant total energy; forces and stresses are obtained by exact energy
derivatives so they remain conservative and suitable for molecular dynamics.

The model combines:

- periodic-correct local neighbor geometry;
- C2-cutoff radial functions, spherical harmonics, and learned body-order-two
  through body-order-four Clebsch-Gordan density contractions;
- strictly local neighbor-set attention with invariant weights and equivariant
  geometric values; updated hidden states are never sent between atom centers;
- energy-derived forces and symmetric stress;
- automatic train/validation plots for energy, force, stress, and total loss.

> **Research status:** This is experimental research software. Validate a
> checkpoint against independent structures, equations of state, phonons, and
> molecular-dynamics stability before using it for scientific conclusions.

## Install

Python 3.10-3.12 and the stable e3nn 0.6 series are supported. Full download, virtual-environment, Apple
Silicon, and optional accelerator instructions are in
[docs/INSTALL.md](docs/INSTALL.md).

```bash
git clone https://github.com/paramvir3/Transformers-ACE.git
cd Transformers-ACE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

python -c "import e3nn; print(e3nn.__version__)"
```

The canonical Python API is `transformers_ace`. The original `flashace` import
is retained so existing scripts and `.pt` checkpoints continue to work.

## Train

Prepare an ASE-readable extended XYZ trajectory containing energies, forces,
periodic cells, and optionally stresses. Copy and edit the example configuration:

```bash
cp configs/cspbi3.yaml my_training.yaml
python train.py --config my_training.yaml
```

Training saves the configured `.pt` checkpoint. By default it also writes:

```text
plots/training_curves.png
plots/training_history.csv
```

`model.pt` is the best validation checkpoint after the stress-weight ramp;
`model_last.pt` records the final epoch. The supplied CsPbI3 configuration uses
a deterministic blocked trajectory split so adjacent frames are not scattered
between training and validation. Set `optimizer: "muon"` in the YAML to use
Muon on transformer hidden matrices with auxiliary AdamW for embeddings,
readout layers, radial/tensor-product support weights, biases, normalization
weights, and other non-hidden parameters.

The equations, locality definition, body-order convention, and checkpoint
versioning are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The complete paper-style methods section is provided as
[LaTeX source](docs/TRANSFORMERS_ACE_METHODS.tex) and a
[rendered PDF](output/pdf/transformers_ace_methods.pdf).

## ASE Calculator

```python
from ase.io import read
from transformers_ace import TransformersACECalculator

atoms = read("POSCAR")
atoms.calc = TransformersACECalculator("model.pt", device="cpu")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()
```

## CsPbI3 Test

Five periodic CsPbI3 polymorph structures and scripts for phase energies and
geometry optimization are included under `tests/cspbi3`. See
[docs/CSPBI3_TEST.md](docs/CSPBI3_TEST.md) for the complete workflow.

```bash
python tests/cspbi3/evaluate_phases.py \
  --model /absolute/path/to/model.pt \
  --device cpu \
  --reference minimum
```

## LAMMPS and PLUMED

Native LAMMPS support is included for rare-event workflow testing. Export a
checkpoint to TorchScript, patch/build LAMMPS with `pair_style transformers_ace`,
and attach PLUMED as a normal LAMMPS fix. See [docs/LAMMPS.md](docs/LAMMPS.md).
The working CsPbI3 standalone LAMMPS smoke test is in
[`tests/run_lammps`](tests/run_lammps).

For MPI and multi-GPU runs, use `newton on` in the LAMMPS input. The native
pair style evaluates local owned-atom energies with ghost atoms in the
neighborhood and lets LAMMPS reverse-communicate ghost force components. With
`pair_style transformers_ace device auto`, each MPI rank maps to
`local_rank % visible_gpu_count`, so a typical one-node four-GPU run is:

```bash
mpirun -np 4 /path/to/lammps/build/lmp -in in.transformers_ace
```

Install PLUMED first:

```bash
brew install pkg-config

git clone https://github.com/paramvir3/plumed2.git
cd plumed2
plumed_dir="${PWD}"

./configure --enable-modules=all --prefix="${PWD}"
make -j4
make install
source "${PWD}/sourceme.sh"
export PKG_CONFIG_PATH="${plumed_dir}/lib/pkgconfig:${PKG_CONFIG_PATH}"
```

Patch and configure LAMMPS:

```bash
git clone --depth=1 https://github.com/lammps/lammps
cd lammps

cd /path/to/Transformers-ACE/lammps/pair_style
bash patch_lammps.sh /path/to/lammps

cd /path/to/lammps
mkdir -p build
cd build

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

```bash
python -m transformers_ace.deploy \
  --checkpoint training/model.pt \
  --output model.transformers_ace.pt \
  --type-map Cs Pb I \
  --example-structure tests/cspbi3/structures/cubic_alpha_phase.vasp
```

### PLUMED Rare-Event Example

The repository includes an explicit LAMMPS plus PLUMED biased-dynamics example
for a non-perovskite delta CsPbI3 to perovskite CsPbI3 transition:

```text
tests/run_lammps/test_plumed_cspbi3
```

It contains the LAMMPS input, PLUMED `DSFTHREE` structure-factor collective
variable, and the 640-atom CsPbI3 starting structure. The exact macOS
LAMMPS+PLUMED CMake build recipe is documented in
[docs/LAMMPS.md](docs/LAMMPS.md).

## Compatibility

These imports are equivalent:

```python
from transformers_ace import TransformersACE, TransformersACECalculator
from flashace import FlashACE, FlashACECalculator
```

## Acknowledgements

Transformers-ACE was developed with assistance from OpenAI Codex.

## License

See [LICENSE](LICENSE).
