import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Independent, Uniform

from rd_sbi.eval.fig1_quality import CASES
from rd_sbi.inference.tsnpe_runner import TSNPEConfig, TSNPERunner


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(module_name: str, script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _write_case_artifacts(
    run_root: Path,
    case: str,
    *,
    valid: bool,
    seed: int,
) -> None:
    case_dir = run_root / case
    case_dir.mkdir(parents=True, exist_ok=True)
    npz_path = case_dir / f"{case}_sbi_posterior_20000.npz"
    summary_path = case_dir / f"{case}_run_summary.json"
    mode_suffixes = {
        "kerr220": ["A_220", "phi_220"],
        "kerr221": ["A_220", "phi_220", "A_221", "phi_221"],
        "kerr330": ["A_220", "phi_220", "A_330", "phi_330"],
    }
    extra_true = {
        "kerr220": [5.0e-21, 1.047],
        "kerr221": [8.92e-21, 1.047, 9.81e-21, 4.19],
        "kerr330": [30.0e-21, 1.047, 3.0e-21, 5.014],
    }

    rng = np.random.default_rng(seed)
    if valid:
        base_samples = np.column_stack(
            [
                np.clip(rng.normal(67.0, 4.0, size=20_000), 20.0, 300.0),
                np.clip(rng.normal(0.67, 0.03, size=20_000), 0.0, 0.99),
            ]
        ).astype(np.float32)
        diagnostics = [
            {
                "round_index": 1,
                "num_simulations": 50_000,
                "truncated_prior_volume": 0.80,
                "probe_acceptance_rate": 0.80,
                "volume_ratio_to_previous": None,
                "stop_by_ratio": False,
                "stop_eligible": False,
                "stop_reason": "first_round",
            },
            {
                "round_index": 2,
                "num_simulations": 100_000,
                "truncated_prior_volume": 0.40,
                "probe_acceptance_rate": 0.40,
                "volume_ratio_to_previous": 0.50,
                "stop_by_ratio": False,
                "stop_eligible": False,
                "stop_reason": "continue",
            },
            {
                "round_index": 3,
                "num_simulations": 100_000,
                "truncated_prior_volume": 0.35,
                "probe_acceptance_rate": 0.35,
                "volume_ratio_to_previous": 0.875,
                "stop_by_ratio": True,
                "stop_eligible": True,
                "stop_reason": "eligible_ratio_check",
            },
        ]
    else:
        base_samples = np.column_stack(
            [
                rng.uniform(20.0, 300.0, size=20_000),
                rng.uniform(0.0, 0.99, size=20_000),
            ]
        ).astype(np.float32)
        diagnostics = [
            {
                "round_index": 1,
                "num_simulations": 50_000,
                "truncated_prior_volume": 1.0,
                "probe_acceptance_rate": 1.0,
                "volume_ratio_to_previous": None,
                "stop_by_ratio": False,
                "stop_eligible": False,
                "stop_reason": "first_round",
            },
            {
                "round_index": 2,
                "num_simulations": 100_000,
                "truncated_prior_volume": 1.0,
                "probe_acceptance_rate": 1.0,
                "volume_ratio_to_previous": 1.0,
                "stop_by_ratio": False,
                "stop_eligible": False,
                "stop_reason": "no_volume_shrink",
            },
            {
                "round_index": 3,
                "num_simulations": 100_000,
                "truncated_prior_volume": 1.0,
                "probe_acceptance_rate": 1.0,
                "volume_ratio_to_previous": 1.0,
                "stop_by_ratio": False,
                "stop_eligible": False,
                "stop_reason": "no_volume_shrink",
            },
        ]

    extras = np.tile(np.asarray(extra_true[case], dtype=np.float32), (20_000, 1))
    samples = np.concatenate([base_samples, extras], axis=1)
    theta_true = np.asarray([67.0, 0.67, *extra_true[case]], dtype=np.float32)
    prior_low = [20.0, 0.0]
    prior_high = [300.0, 0.99]
    for name in mode_suffixes[case]:
        if name.startswith("A_"):
            prior_low.append(0.0)
            prior_high.append(50.0e-21)
        else:
            prior_low.append(0.0)
            prior_high.append(float(2.0 * np.pi))

    np.savez_compressed(
        npz_path,
        samples=samples,
        theta_true=theta_true,
        x_observed=np.zeros(408, dtype=np.float32),
    )
    summary = {
        "case": case,
        "output_npz": str(npz_path),
        "prior": {
            "param_order": ["M_f_msun", "chi_f", *mode_suffixes[case]],
            "low": prior_low,
            "high": prior_high,
        },
        "preprocessing": {
            "appendix_a_faithful": True,
            "input_dim": 408,
            "per_detector_bins": 204,
            "whitening_method": "psd_to_acf_to_toeplitz_to_cholesky",
            "x_standardization": {"applied": True, "paper_faithful": True},
            "theta_normalization": {"applied": True, "paper_faithful": True},
        },
        "waveform_model": {
            "kerr_mapping": {
                "backend": "fit",
                "paper_reference": "fits_[45]",
                "paper_faithful": True,
            }
        },
        "paper_case_validation": {
            "issues": [],
            "passed": True,
        },
        "training_contract": {
            "varying_noise": {"enabled": True},
        },
        "injection_context": {
            "ra_rad": 1.95,
            "dec_rad": -1.27,
            "psi_rad": 0.82,
            "gps_h1": 1126259462.42323,
        },
        "snr": {
            "measured_network_snr": 14.0,
            "per_detector": {"H1": 10.0, "L1": 9.8},
            "psd_source": {"H1": "h1_psd", "L1": "l1_psd"},
            "paper_faithful": True,
        },
        "params": {
            "num_sim_first": 50_000,
            "num_sim_round": 100_000,
            "posterior_samples": 20_000,
            "trunc_quantile": 1e-4,
            "stopping_ratio": 0.8,
            "min_rounds_before_stopping": 3,
            "require_volume_shrink_for_stopping": True,
            "fail_on_no_truncation_for_next_round": True,
            "truncation_probe_samples": 50_000,
        },
        "tsnpe_definition": {
            "truncation": "density-threshold HPD approximation",
        },
        "diagnostics": diagnostics,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_overlay_summary(path: Path, cases: tuple[str, ...]) -> None:
    summary = {
        "cases": {
            case: {
                "sbi_samples": 20_000,
                "pyring_samples": 20_000,
                "wasserstein_Mf": 0.1,
                "ks_Mf_D": 0.1,
            }
            for case in cases
        }
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _write_pyring_dir(pyring_dir: Path) -> None:
    pyring_dir.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        np.savez_compressed(
            pyring_dir / f"{case}_pyring.npz",
            M=np.linspace(60.0, 74.0, 128, dtype=np.float32),
            chi=np.linspace(0.6, 0.74, 128, dtype=np.float32),
        )
    (pyring_dir / "manifest_pyring.json").write_text(
        json.dumps(
            {
                "pyring_version": "2.3.0",
                "cpnest_live_points": 4096,
                "cpnest_max_mcmc_steps": 4094,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_no_shrink_run_is_rejected_by_audit_and_plotting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run_root"
    pyring_dir = tmp_path / "pyring"
    overlay_summary = run_root / "overlay_summary.json"
    coverage_summary = run_root / "coverage_summary.json"
    run_root.mkdir(parents=True, exist_ok=True)
    coverage_summary.write_text(
        json.dumps(
            {
                "num_catalogs": 100,
                "draws_per_catalog": 100,
                "test_prior": "final_truncated_prior",
                "coverage_definition": "hpd_gamma",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out_fig = tmp_path / "fig1_paper_sbi.png"

    for idx, case in enumerate(CASES):
        _write_case_artifacts(run_root, case, valid=(case != "kerr220"), seed=10 + idx)
    _write_pyring_dir(pyring_dir)
    _write_overlay_summary(overlay_summary, CASES)

    audit_module = _load_script_module("audit_fig1_script", "11_audit_fig1_paper_spec.py")
    findings = audit_module._check_artifacts(
        run_root,
        pyring_dir,
        overlay_summary,
        coverage_summary,
        pyring_dir / "manifest_pyring.json",
    )
    failing = {finding.key: finding.detail for finding in findings if not finding.ok}
    assert "artifact.kerr220.paper_grade_quality" in failing
    assert "final_round_no_effective_shrinkage" in failing["artifact.kerr220.paper_grade_quality"]

    plot_module = _load_script_module("plot_fig1_script_reject", "14_plot_fig1_paper_grade.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_fig1",
            "--run-root",
            str(run_root),
            "--overlay-summary",
            str(overlay_summary),
            "--output-figure",
            str(out_fig),
        ],
    )
    with pytest.raises(RuntimeError, match="failures"):
        plot_module.main()
    assert not out_fig.exists()

    grade = audit_module._grade_from_findings(findings, check_artifacts=True)
    assert grade["grade"] == "D"


def test_missing_case_prevents_paper_grade_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_root = tmp_path / "run_root"
    overlay_summary = run_root / "overlay_summary.json"
    out_fig = tmp_path / "fig1_paper_sbi.png"

    for idx, case in enumerate(CASES[:2]):
        _write_case_artifacts(run_root, case, valid=True, seed=30 + idx)
    _write_overlay_summary(overlay_summary, CASES)

    plot_module = _load_script_module("plot_fig1_script_missing", "14_plot_fig1_paper_grade.py")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_fig1",
            "--run-root",
            str(run_root),
            "--overlay-summary",
            str(overlay_summary),
            "--output-figure",
            str(out_fig),
        ],
    )
    with pytest.raises(RuntimeError, match="kerr330"):
        plot_module.main()
    assert not out_fig.exists()


def test_audit_grade_only_coverage_missing_is_b() -> None:
    audit_module = _load_script_module("audit_fig1_grade_only_coverage", "11_audit_fig1_paper_spec.py")
    findings = [
        audit_module.Finding(True, "artifact.kerr220.paper_grade_quality", "ok"),
        audit_module.Finding(True, "artifact.kerr221.paper_grade_quality", "ok"),
        audit_module.Finding(True, "artifact.kerr330.paper_grade_quality", "ok"),
        audit_module.Finding(False, "artifact.coverage.summary_exists", "missing"),
    ]
    grade = audit_module._grade_from_findings(findings, check_artifacts=True)
    assert grade["grade"] == "B"


def test_audit_grade_pipeline_incomplete_is_c() -> None:
    audit_module = _load_script_module("audit_fig1_grade_pipeline", "11_audit_fig1_paper_spec.py")
    findings = [
        audit_module.Finding(True, "artifact.kerr220.paper_grade_quality", "ok"),
        audit_module.Finding(True, "artifact.kerr221.paper_grade_quality", "ok"),
        audit_module.Finding(True, "artifact.kerr330.paper_grade_quality", "ok"),
        audit_module.Finding(False, "pipeline.fig1.uses_exact_whitening", "missing"),
        audit_module.Finding(False, "artifact.coverage.summary_exists", "missing"),
    ]
    grade = audit_module._grade_from_findings(findings, check_artifacts=True)
    assert grade["grade"] == "C"


def test_sample_posterior_cpu_safe_uses_fallback_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    prior = Independent(Uniform(-torch.ones(2), torch.ones(2)), 1)

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        return theta

    runner = TSNPERunner(
        prior=prior,
        simulator=simulator,
        x_observed=torch.zeros(2),
        density_estimator_builder=lambda *_args, **_kwargs: None,
        config=TSNPEConfig(device="cpu", show_progress_bars=False),
    )

    class _UnsafeEstimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))

        def sample(self, sample_shape, condition):  # noqa: ANN001
            raise RuntimeError("unsafe sampler")

    class _SafeEstimator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
            self.seen_condition_device: str | None = None

        def sample(self, sample_shape, condition):  # noqa: ANN001
            self.seen_condition_device = condition.device.type
            return torch.zeros(sample_shape[0], 1, 2, device=condition.device)

    runner.last_density_estimator = _UnsafeEstimator()
    safe_estimator = _SafeEstimator()
    monkeypatch.setattr(runner, "_cpu_safe_estimator_copy", lambda: safe_estimator)

    with pytest.raises(RuntimeError, match="unsafe sampler"):
        runner.sample_posterior(8, cpu_safe=False)

    samples = runner.sample_posterior(8, cpu_safe=True)
    assert samples.shape == (8, 2)
    assert samples.device.type == "cpu"
    assert safe_estimator.seen_condition_device == "cpu"
    mode_suffixes = {
        "kerr220": [],
        "kerr221": ["A_220", "phi_220", "A_221", "phi_221"],
        "kerr330": ["A_220", "phi_220", "A_330", "phi_330"],
    }
