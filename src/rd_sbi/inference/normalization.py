"""Normalization helpers for paper-faithful inference pipelines."""

from __future__ import annotations

import numpy as np
import torch


class UnitCubeBoxTransform:
    """Affine map between physical box priors and [0, 1]^d."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        low_arr = np.asarray(low, dtype=np.float64).reshape(-1)
        high_arr = np.asarray(high, dtype=np.float64).reshape(-1)
        if low_arr.shape != high_arr.shape:
            raise ValueError("low and high must have matching shapes")
        widths = high_arr - low_arr
        if np.any(widths <= 0.0):
            raise ValueError("all box widths must be positive")
        self.low = low_arr
        self.high = high_arr
        self.width = widths

    @property
    def dim(self) -> int:
        return int(self.low.shape[0])

    def forward_numpy(self, theta_physical: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_physical, dtype=np.float64)
        return (theta - self.low) / self.width

    def inverse_numpy(self, theta_unit: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta_unit, dtype=np.float64)
        return self.low + theta * self.width

    def forward_tensor(self, theta_physical: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.low, dtype=theta_physical.dtype, device=theta_physical.device)
        width = torch.as_tensor(self.width, dtype=theta_physical.dtype, device=theta_physical.device)
        return (theta_physical - low) / width

    def inverse_tensor(self, theta_unit: torch.Tensor) -> torch.Tensor:
        low = torch.as_tensor(self.low, dtype=theta_unit.dtype, device=theta_unit.device)
        width = torch.as_tensor(self.width, dtype=theta_unit.dtype, device=theta_unit.device)
        return low + theta_unit * width
