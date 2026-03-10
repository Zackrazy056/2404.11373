from pathlib import Path

import pytest
import torch
from torch.distributions import Independent, Uniform

from rd_sbi.config import load_yaml_config, validate_paper_case_config
from rd_sbi.inference.embedding_net import EmbeddingConfig, EmbeddingFCNet, build_nsf_density_estimator
from rd_sbi.inference.normalization import UnitCubeBoxTransform
from rd_sbi.inference.sbi_loss_patch import NoiseResamplingSNPE
from rd_sbi.inference.tsnpe_runner import TSNPEConfig, TSNPERunner, should_stop_by_volume_ratio


ROOT = Path(__file__).resolve().parents[1]


def test_embedding_fc_net_output_shape() -> None:
    model = EmbeddingFCNet(input_dim=408, num_hidden_layers=2, hidden_dim=150, output_dim=128)
    x = torch.randn(16, 408)
    y = model(x)
    assert y.shape == (16, 128)


def test_should_stop_by_volume_ratio() -> None:
    assert should_stop_by_volume_ratio(0.10, 0.085, 0.8) is True
    assert should_stop_by_volume_ratio(0.10, 0.05, 0.8) is False
    assert should_stop_by_volume_ratio(0.0, 0.05, 0.8) is False


def test_tsnpe_runner_minimal_rounds() -> None:
    torch.manual_seed(7)

    theta_dim = 2
    x_dim = 4
    low = -2.0 * torch.ones(theta_dim)
    high = 2.0 * torch.ones(theta_dim)
    prior = Independent(Uniform(low, high), 1)

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        t0 = theta[:, 0:1]
        t1 = theta[:, 1:2]
        return torch.cat(
            [
                t0 + 0.5 * t1,
                t0 - 0.25 * t1,
                t0 * t1,
                t0**2 + t1**2,
            ],
            dim=1,
        )

    theta_obs = torch.tensor([[0.4, -0.7]], dtype=torch.float32)
    x_obs = simulator(theta_obs)[0]

    density_builder = build_nsf_density_estimator(
        embedding_config=EmbeddingConfig(input_dim=x_dim, num_hidden_layers=2, hidden_dim=32, output_dim=16),
        hidden_features=32,
        num_transforms=3,
        num_bins=8,
        num_blocks=2,
    )

    cfg = TSNPEConfig(
        num_simulations_first_round=128,
        num_simulations_per_round=128,
        max_rounds=3,
        trunc_quantile=0.2,
        stopping_ratio=0.95,
        posterior_samples_for_hpd=256,
        prior_volume_mc_samples=512,
        rejection_candidate_batch=512,
        rejection_max_batches=64,
        training_batch_size=64,
        learning_rate=1e-3,
        validation_fraction=0.1,
        stop_after_epochs=2,
        max_num_epochs=6,
        discard_round1_after_first_update=True,
        device="cpu",
        show_progress_bars=False,
    )

    runner = TSNPERunner(
        prior=prior,
        simulator=simulator,
        x_observed=x_obs,
        density_estimator_builder=density_builder,
        config=cfg,
    )
    posterior, diagnostics = runner.run()

    assert posterior is not None
    assert len(diagnostics) >= 1
    assert diagnostics[0].num_simulations == 128
    assert diagnostics[0].epochs_trained_this_round is not None
    assert diagnostics[0].epochs_trained_this_round >= 1
    if len(diagnostics) > 1:
        assert diagnostics[1].epochs_trained_this_round is not None
        assert diagnostics[1].epochs_trained_this_round >= 1


def test_tsnpe_runner_uses_noise_resampling_when_enabled(monkeypatch) -> None:
    torch.manual_seed(11)

    theta_dim = 2
    x_dim = 4
    low = -2.0 * torch.ones(theta_dim)
    high = 2.0 * torch.ones(theta_dim)
    prior = Independent(Uniform(low, high), 1)

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        return torch.cat([theta, theta[:, :1] - theta[:, 1:2], theta[:, :1] + theta[:, 1:2]], dim=1)

    x_obs = simulator(torch.tensor([[0.1, -0.2]], dtype=torch.float32))[0]
    density_builder = build_nsf_density_estimator(
        embedding_config=EmbeddingConfig(input_dim=x_dim, num_hidden_layers=2, hidden_dim=16, output_dim=8),
        hidden_features=16,
        num_transforms=2,
        num_bins=8,
        num_blocks=2,
    )

    seen: dict[str, bool] = {"used": False}

    class RecordingNoiseResamplingSNPE(NoiseResamplingSNPE):
        def __init__(self, *args, **kwargs):  # noqa: ANN002,ANN003
            seen["used"] = True
            super().__init__(*args, **kwargs)

    monkeypatch.setattr("rd_sbi.inference.tsnpe_runner.NoiseResamplingSNPE", RecordingNoiseResamplingSNPE)

    cfg = TSNPEConfig(
        num_simulations_first_round=64,
        num_simulations_per_round=64,
        max_rounds=1,
        trunc_quantile=0.2,
        stopping_ratio=0.95,
        posterior_samples_for_hpd=128,
        prior_volume_mc_samples=128,
        rejection_candidate_batch=128,
        rejection_max_batches=16,
        training_batch_size=32,
        learning_rate=1e-3,
        validation_fraction=0.1,
        stop_after_epochs=1,
        max_num_epochs=2,
        discard_round1_after_first_update=True,
        device="cpu",
        show_progress_bars=False,
        varying_noise_enabled=True,
        varying_noise_std=1.0,
    )

    runner = TSNPERunner(
        prior=prior,
        simulator=simulator,
        x_observed=x_obs,
        density_estimator_builder=density_builder,
        config=cfg,
    )
    posterior, diagnostics = runner.run()

    assert posterior is not None
    assert diagnostics
    assert seen["used"] is True


def test_validate_paper_case_config_accepts_repo_cases() -> None:
    for case in ("kerr220", "kerr221", "kerr330"):
        cfg = load_yaml_config(ROOT / "configs" / "injections" / f"{case}.yaml")
        assert validate_paper_case_config(cfg, case) == []


def test_validate_paper_case_config_rejects_auto_qnm_method() -> None:
    cfg = load_yaml_config(ROOT / "configs" / "injections" / "kerr220.yaml")
    cfg["qnm"]["method"] = "auto"
    issues = validate_paper_case_config(cfg, "kerr220")
    assert "qnm.method=auto" in issues


def test_unit_cube_box_transform_roundtrip() -> None:
    transform = UnitCubeBoxTransform(low=[20.0, 0.0], high=[300.0, 0.99])
    physical = torch.tensor([[67.0, 0.67], [20.0, 0.0], [300.0, 0.99]], dtype=torch.float32)
    unit = transform.forward_tensor(physical)
    restored = transform.inverse_tensor(unit)
    assert torch.allclose(restored, physical, atol=1e-6)


def test_tsnpe_runner_reduces_round_budget_when_acceptance_is_too_low() -> None:
    prior = Independent(Uniform(torch.zeros(2), torch.ones(2)), 1)
    runner = TSNPERunner(
        prior=prior,
        simulator=lambda theta: theta,
        x_observed=torch.zeros(2),
        density_estimator_builder=lambda *_args, **_kwargs: None,
        config=TSNPEConfig(
            num_simulations_first_round=128,
            num_simulations_per_round=100_000,
            max_rounds=2,
            rejection_candidate_batch=16_384,
            rejection_max_batches=1_024,
            min_simulations_per_round_after_reduction=4_096,
            rejection_acceptance_safety_factor=0.8,
        ),
    )

    plan = runner._plan_round_simulation_budget(
        requested_num_simulations=100_000,
        estimated_acceptance_rate=6.6e-4,
    )

    assert plan.simulation_budget_adjusted is True
    assert plan.requested_num_simulations == 100_000
    assert plan.effective_num_simulations == 8858
    assert plan.estimated_max_accepted_under_budget == 8858
    assert plan.rejection_budget_max_draws == 16_384 * 1_024


def test_tsnpe_runner_round_budget_fails_early_when_too_small() -> None:
    prior = Independent(Uniform(torch.zeros(2), torch.ones(2)), 1)
    runner = TSNPERunner(
        prior=prior,
        simulator=lambda theta: theta,
        x_observed=torch.zeros(2),
        density_estimator_builder=lambda *_args, **_kwargs: None,
        config=TSNPEConfig(
            num_simulations_first_round=128,
            num_simulations_per_round=100_000,
            max_rounds=2,
            rejection_candidate_batch=1_024,
            rejection_max_batches=8,
            min_simulations_per_round_after_reduction=4_096,
            rejection_acceptance_safety_factor=0.8,
        ),
    )

    with pytest.raises(RuntimeError, match="rejection budget infeasible"):
        runner._plan_round_simulation_budget(
            requested_num_simulations=100_000,
            estimated_acceptance_rate=1.0e-4,
        )


def test_tsnpe_runner_relaxes_truncation_when_probe_acceptance_is_too_low() -> None:
    prior = Independent(Uniform(torch.zeros(2), torch.ones(2)), 1)
    runner = TSNPERunner(
        prior=prior,
        simulator=lambda theta: theta,
        x_observed=torch.zeros(2),
        density_estimator_builder=lambda *_args, **_kwargs: None,
        config=TSNPEConfig(
            adaptive_truncation_relaxation_enabled=True,
            min_probe_acceptance_rate_for_next_round=0.002,
        ),
    )
    log_prob_probe = torch.linspace(-100.0, 0.0, 50_000)
    original_threshold = -0.1

    plan = runner._build_truncation_plan_from_log_prob(
        threshold=original_threshold,
        log_prob_probe=log_prob_probe,
        round_idx=2,
    )

    assert plan.truncation_relaxed is True
    assert plan.target_min_probe_acceptance_rate == pytest.approx(0.002)
    assert plan.probe_acceptance_rate >= 0.002
    assert plan.threshold < original_threshold


def test_tsnpe_runner_does_not_relax_first_round_truncation() -> None:
    prior = Independent(Uniform(torch.zeros(2), torch.ones(2)), 1)
    runner = TSNPERunner(
        prior=prior,
        simulator=lambda theta: theta,
        x_observed=torch.zeros(2),
        density_estimator_builder=lambda *_args, **_kwargs: None,
        config=TSNPEConfig(
            adaptive_truncation_relaxation_enabled=True,
            min_probe_acceptance_rate_for_next_round=0.01,
        ),
    )
    log_prob_probe = torch.linspace(-100.0, 0.0, 10_000)
    original_threshold = -0.1

    plan = runner._build_truncation_plan_from_log_prob(
        threshold=original_threshold,
        log_prob_probe=log_prob_probe,
        round_idx=1,
    )

    assert plan.truncation_relaxed is False
    assert plan.threshold == pytest.approx(original_threshold)
