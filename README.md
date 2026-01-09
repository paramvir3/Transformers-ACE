# Flash-ACE
This repository is an attempt to make use attention mechanism on Atomic Cluster Expansion (Drautz, Phys. Rev. B 99, 2019) for making precise and scalable machine learning interatomic potentials

Please DO NOT USE as this is purely for research purposes

## Running training

`train.py` accepts `--config / -c` to point at any YAML file. If you omit the
flag, it will search for `config.yaml` in the repository root and then fall
back to `training/config.yaml`.

Example:

```bash
python train.py --config training/config.yaml
```

Minimal example:

```bash
python train.py --config training/minimal_config.yaml
```


## The main problem 

Energies converge, not forces
