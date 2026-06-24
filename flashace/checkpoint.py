import importlib
import sys

import numpy as np
import torch


def _numpy_checkpoint_globals():
    """Return the NumPy scalar types used by Transformers-ACE checkpoints."""
    try:
        core = importlib.import_module("numpy._core")
        multiarray = importlib.import_module("numpy._core.multiarray")
        numeric = importlib.import_module("numpy._core.numeric")
    except ModuleNotFoundError:
        core = importlib.import_module("numpy.core")
        multiarray = importlib.import_module("numpy.core.multiarray")
        numeric = importlib.import_module("numpy.core.numeric")

    # NumPy 2 renamed its private implementation package. These aliases let a
    # checkpoint written by NumPy 2 load under NumPy 1 without disabling
    # PyTorch's restricted weights-only unpickler.
    sys.modules.setdefault("numpy._core", core)
    sys.modules.setdefault("numpy._core.multiarray", multiarray)
    sys.modules.setdefault("numpy._core.numeric", numeric)

    safe_globals = [
        (multiarray.scalar, "numpy._core.multiarray.scalar"),
        (multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
    ]
    for dtype_name in (
        "bool",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ):
        dtype_class = type(np.dtype(dtype_name))
        if dtype_class not in safe_globals:
            safe_globals.append(dtype_class)
    return safe_globals


def load_checkpoint(path, map_location):
    """Load a tensor/config checkpoint with PyTorch's restricted unpickler."""
    safe_globals = _numpy_checkpoint_globals()
    if hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals(safe_globals):
            return torch.load(path, map_location=map_location, weights_only=True)

    # PyTorch versions before safe_globals predate the default weights-only
    # behavior. Keep an explicit fallback for the supported torch>=2.2 range.
    return torch.load(path, map_location=map_location, weights_only=False)
