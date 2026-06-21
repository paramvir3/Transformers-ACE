# Training Workspace

This directory contains the macOS CPU configuration and published artifacts
for the current 100-epoch CsPbI3 training run. The checkpoint, history, curves,
and configuration are kept together so the reported phase tests can be traced
to the exact model used.

- `train.extxyz`: the 979-frame dataset consumed by `config.yaml`;
- `model.pt`: the trained checkpoint used for the published phase evaluation;
- `plots/training_history.csv`: per-epoch train/validation metrics;
- `plots/training_curves.png`: the corresponding learning curves;
- `reproducibility.yaml`: hashes, versions, split details, and limitations;
- `requirements-reproduce.txt`: pinned direct runtime dependencies.

The larger source trajectory, periodic checkpoints, and additional generated
files remain ignored by Git.

Install the recorded direct dependencies inside a fresh environment:

```bash
python -m pip install -r training/requirements-reproduce.txt
python -m pip install -e .
```

From the repository root:

```bash
cd training
python ../train.py --config config.yaml
```

The default configuration reads `train.extxyz`, writes `model.pt`, and saves
the training history and curves under `plots/`. Update the thread count in
`config.yaml` when running on a machine with a different number of CPU cores.

Verify the published dataset, configuration, checkpoint, and plots:

```bash
shasum -a 256 config.yaml train.extxyz model.pt \
  plots/training_history.csv plots/training_curves.png
```

The train/validation partition is reproducible because `train.py` uses split
seed 42. This run did not record a global model-initialization and
batch-shuffle seed. Loading `model.pt` therefore reproduces its predictions,
but retraining is not expected to be bit-for-bit identical. See
`reproducibility.yaml` for the exact distinction.

## Stress And Variable Cells

Stress is obtained from the same scalar energy as the forces:

```text
F_i = -dE/dr_i
sigma_ab = (1/V) dE/d(epsilon_ab)
```

The loader expects ASE stress units (`eV/Angstrom^3`) and ASE Voigt order
`xx, yy, zz, yz, xz, xy`. Virials use the atomistic convention
`virial = -V * stress`. All six independent stress components enter the loss.

The CsPbI3 stresses are only about `1e-3 eV/Angstrom^3`, so their raw squared
error is much smaller than the force loss. The larger `stress_weight` in the
current configuration compensates for this unit scale. Compare the validation
stress RMSE as well as energy and force RMSE when selecting a checkpoint.

Do not enable random position displacement while reusing the original DFT
labels. New displaced or strained structures need newly evaluated energies,
forces, and stresses. For reliable variable-cell relaxation, include labeled
isotropic, uniaxial, and shear strains around every phase of interest and
validate energy-volume curves on structures excluded as complete trajectories
or phase groups. A random frame split can leak nearby trajectory frames into
both partitions and should not be treated as a transferability test.

For CUDA training, change `device` to `cuda`, enable `use_amp` when supported,
and adjust the data-loader worker count for the target system.
