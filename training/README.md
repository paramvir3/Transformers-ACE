# Training Workspace

This directory contains the current macOS CPU configuration used for CsPbI3
training. Datasets, checkpoints, periodic checkpoints, and generated plots are
ignored by Git so they remain local.

From the repository root:

```bash
cd training
python ../train.py --config config.yaml
```

The default configuration reads `train.extxyz`, writes `model.pt`, and saves
the training history and curves under `plots/`. Update the thread count in
`config.yaml` when running on a machine with a different number of CPU cores.

For CUDA training, change `device` to `cuda`, enable `use_amp` when supported,
and adjust the data-loader worker count for the target system.
