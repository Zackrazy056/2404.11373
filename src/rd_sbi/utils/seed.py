"""Global random seed helpers."""

from __future__ import annotations

import os
import random
from typing import Final

import numpy as np

_UINT32_MAX: Final[int] = 2**32 - 1


def _normalize_seed(seed: int) -> int:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed % _UINT32_MAX


def set_global_seed(seed: int, deterministic_torch: bool = True) -> int:
    """Set random seeds for Python, NumPy and Torch (if available).

    Returns the normalized seed that was applied.
    """
    seed_norm = _normalize_seed(seed)

    random.seed(seed_norm)
    np.random.seed(seed_norm)
    os.environ["PYTHONHASHSEED"] = str(seed_norm)

    try:
        import torch

        torch.manual_seed(seed_norm)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_norm)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed_norm
