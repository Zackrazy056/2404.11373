import torch
from torch.distributions import Independent, Uniform

from rd_sbi.inference.sbi_loss_patch import NoiseResamplingConfig, NoiseResamplingSNPE


def test_noise_resampling_changes_input_when_enabled() -> None:
    prior = Independent(Uniform(-torch.ones(2), torch.ones(2)), 1)
    snpe = NoiseResamplingSNPE(
        prior=prior,
        density_estimator="maf",
        noise_config=NoiseResamplingConfig(enabled=True, noise_std=0.1),
    )

    class _DummyNet:
        training = True

    snpe._neural_net = _DummyNet()  # type: ignore[attr-defined]
    x = torch.zeros(8, 4)
    x_aug = snpe._augment_x(x)
    assert not torch.allclose(x_aug, x)


def test_noise_resampling_no_change_when_disabled() -> None:
    prior = Independent(Uniform(-torch.ones(2), torch.ones(2)), 1)
    snpe = NoiseResamplingSNPE(
        prior=prior,
        density_estimator="maf",
        noise_config=NoiseResamplingConfig(enabled=False, noise_std=1.0),
    )

    class _DummyNet:
        training = True

    snpe._neural_net = _DummyNet()  # type: ignore[attr-defined]
    x = torch.zeros(8, 4)
    x_aug = snpe._augment_x(x)
    assert torch.allclose(x_aug, x)
