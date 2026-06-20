# Training Workspace

This directory contains the macOS CPU configuration and exact published
artifacts for the 100-epoch CsPbI3 training run:

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

Verify the published files before running:

```bash
shasum -a 256 config.yaml train.extxyz model.pt \
  plots/training_history.csv plots/training_curves.png
```

The train/validation partition is reproducible because `train.py` uses split
seed 42. This historical run did not record a global model-initialization and
batch-shuffle seed. Loading `model.pt` therefore reproduces its predictions,
but retraining is not expected to be bit-for-bit identical. See
`reproducibility.yaml` for the exact distinction.

For CUDA training, change `device` to `cuda`, enable `use_amp` when supported,
and adjust the data-loader worker count for the target system.
