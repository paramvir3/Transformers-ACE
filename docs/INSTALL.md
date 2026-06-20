# Download And Installation

## Requirements

- macOS or Linux
- Python 3.10-3.12
- Git for cloning the repository
- A C/C++ build toolchain if an optional dependency requires compilation

Transformers-ACE runs on CPU and CUDA-capable PyTorch installations. Apple MPS
support depends on the PyTorch and e3nn operations available on the individual
machine; CPU is the reliable macOS default.

## Download With Git

```bash
git clone https://github.com/paramvir3/Transformers-ACE.git
cd Transformers-ACE
```

To update an existing checkout:

```bash
git pull
```

## Download A ZIP

Download the main branch from:

```text
https://github.com/paramvir3/Transformers-ACE/archive/refs/heads/main.zip
```

Extract the archive and enter the resulting `Transformers-ACE-main` directory.

## Create A Python Environment

Create the environment once:

```bash
python3 -m venv .venv
```

Activate it whenever opening a new terminal:

```bash
source .venv/bin/activate
```

Install Transformers-ACE and its dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For development tests:

```bash
python -m pip install -e '.[test]'
```

`torch-scatter` is optional because the current implementation has a native
PyTorch fallback. Install the acceleration extra only when a compatible wheel
is available:

```bash
python -m pip install -e '.[accelerated]'
```

## Verify The Installation

```bash
python -c "import torch, ase, e3nn, transformers_ace; print(torch.__version__)"
python -c "from transformers_ace import TransformersACECalculator; print('installation OK')"
```

## macOS CPU Training

Set `device: "cpu"` and `use_amp: false` in the training YAML. PyTorch normally
chooses a sensible thread count. It can be specified explicitly when desired:

```yaml
device: "cpu"
use_amp: false
torch_num_threads: 10
torch_num_interop_threads: 1
num_workers: 0
```

Use the actual number of CPU cores on the machine. `num_workers: 0` avoids
macOS shared-memory issues while tensor operations still use the configured
PyTorch threads.

## Legacy PyTorch Loading Error

Some combinations of recent PyTorch and older trusted e3nn/checkpoint files may
report a `weights_only` loading error. For checkpoints and dependencies from a
trusted installation, set this before running:

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
```

Do not disable safe loading for untrusted checkpoint files.

## Uninstall Or Leave The Environment

Leave the environment with:

```bash
deactivate
```

Delete `.venv` to remove the isolated environment completely.
