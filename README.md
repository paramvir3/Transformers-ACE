# Flash-ACE
This repository is an attempt to make use attention mechanism (Transformers) on Atomic Cluster Expansion (Drautz, Phys. Rev. B 99, 2019) for making precise and scalable machine learning interatomic potentials

Please DO NOT USE as this is purely for research purposes

## Running training

`train.py` accepts `--config / -c` to point at any YAML file. If you omit the
flag, it will search for `config.yaml` in the repository root and then fall
back to `training/config.yaml`.

Example:

```bash
python train.py --config training/config.yaml
```


## The main problem 

a. Transfomer block improves force learning
b. Number of trainable parameters should stay closer to dataset to avoid overfitting

