import tempfile
from pathlib import Path

import e3nn
import numpy as np
from packaging.version import Version
import torch

from flashace.checkpoint import load_checkpoint


def test_e3nn_stable_api_series_is_supported():
    version = Version(e3nn.__version__)
    assert Version("0.6.0") <= version < Version("0.7.0")


def test_numpy_scalar_checkpoint_loads_with_weights_only_mode():
    checkpoint = {
        "model_state_dict": {"weight": torch.arange(3, dtype=torch.float32)},
        "config": {"energy_shift_per_atom": np.float64(-1.25)},
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        torch.save(checkpoint, path)
        loaded = load_checkpoint(path, map_location="cpu")

    torch.testing.assert_close(
        loaded["model_state_dict"]["weight"],
        checkpoint["model_state_dict"]["weight"],
    )
    assert float(loaded["config"]["energy_shift_per_atom"]) == -1.25
