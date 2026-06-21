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

Python 3.10-3.12 is recommended. Full download, virtual-environment, Apple
Silicon, and optional accelerator instructions are in
[docs/INSTALL.md](docs/INSTALL.md).

```bash
git clone https://github.com/paramvir3/Transformers-ACE.git
cd Transformers-ACE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
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
between training and validation.

The equations, locality definition, body-order convention, and checkpoint
versioning are described in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
The complete paper-style methods section is provided as
[LaTeX source](docs/TRANSFORMERS_ACE_METHODS.tex) and a
[rendered PDF](output/pdf/transformers_ace_methods.pdf).

For very large trajectories, stream every tenth frame without loading the full
file into memory:

```bash
python subsample_extxyz.py input.extxyz output_every10.extxyz --every 10
```

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
