"""SNPE wrapper that re-samples additive noise during loss evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from sbi.inference import SNPE


@dataclass(frozen=True)
class NoiseResamplingConfig:
    """Configuration for dynamic noise augmentation during training."""

    enabled: bool = True
    noise_std: float = 1.0
    apply_in_validation: bool = False


class NoiseResamplingSNPE(SNPE):
    """SNPE that perturbs `x` with fresh Gaussian noise in `_loss`.

    This follows the data-augmentation idea in the paper appendix:
    re-sample noise at each loss evaluation so the estimator is less tied
    to specific noise realizations.
    """

    def __init__(
        self,
        *args: Any,
        noise_config: NoiseResamplingConfig | None = None,
        noise_sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.noise_config = noise_config or NoiseResamplingConfig()
        self.noise_sampler = noise_sampler
        self._last_training_epoch_seen: int | None = None

    def _sample_noise_like(self, x: torch.Tensor) -> torch.Tensor:
        if self.noise_sampler is not None:
            return self.noise_sampler(x)
        std = float(self.noise_config.noise_std)
        if std <= 0.0:
            return torch.zeros_like(x)
        return torch.randn_like(x) * std

    def _augment_x(self, x: torch.Tensor) -> torch.Tensor:
        if not self.noise_config.enabled:
            return x

        training_mode = bool(getattr(self._neural_net, "training", False))
        should_apply = training_mode or self.noise_config.apply_in_validation
        if not should_apply:
            return x

        if training_mode:
            epoch = int(getattr(self, "epoch", -1))
            if self._last_training_epoch_seen != epoch:
                self._last_training_epoch_seen = epoch
        return x + self._sample_noise_like(x)

    def _loss(  # type: ignore[override]
        self,
        theta: torch.Tensor,
        x: torch.Tensor,
        masks: torch.Tensor,
        proposal: Optional[Any],
        calibration_kernel: Callable,
        force_first_round_loss: bool = False,
    ) -> torch.Tensor:
        x_aug = self._augment_x(x)
        return super()._loss(
            theta=theta,
            x=x_aug,
            masks=masks,
            proposal=proposal,
            calibration_kernel=calibration_kernel,
            force_first_round_loss=force_first_round_loss,
        )
