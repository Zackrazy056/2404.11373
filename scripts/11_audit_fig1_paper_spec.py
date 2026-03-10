"""Audit Fig.1 reproduction against 2404.11373-oriented hard constraints.

The audit is intentionally stricter than "can draw a contour". It separates:
- static paper-faithful prerequisites (injection / preprocessing / model / TSNPE)
- runtime artifact checks (posterior shrinkage / overlay / sample validity)
- a final reproduction grade:
  A Paper-faithful
  B Figure-faithful but diagnostics incomplete
  C Pipeline-validated only
  D Invalid for paper comparison
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import validate_paper_case_config  # noqa: E402
from rd_sbi.eval.fig1_quality import load_posterior_npz, paper_grade_run_issues  # noqa: E402

CASES = ("kerr220", "kerr221", "kerr330")
EXPECTED_MODES = {
    "kerr220": {(2, 2, 0)},
    "kerr221": {(2, 2, 0), (2, 2, 1)},
    "kerr330": {(2, 2, 0), (3, 3, 0)},
}
EXPECTED_MODE_PARAMS = {
    "kerr220": {(2, 2, 0): {"amplitude": 5.0e-21, "phase": 1.047}},
    "kerr221": {(2, 2, 0): {"amplitude": 8.92e-21, "phase": 1.047}, (2, 2, 1): {"amplitude": 9.81e-21, "phase": 4.19}},
    "kerr330": {(2, 2, 0): {"amplitude": 30.0e-21, "phase": 1.047}, (3, 3, 0): {"amplitude": 3.0e-21, "phase": 5.014}},
}
EXPECTED_SNR = {"kerr220": 14.0, "kerr221": 14.0, "kerr330": 53.0}
EXPECTED_INCLINATION = {"kerr220": np.pi, "kerr221": np.pi, "kerr330": np.pi / 4.0}
EXPECTED_SOURCE = {
    "ra_rad": 1.95,
    "dec_rad": -1.27,
    "psi_rad": 0.82,
    "gps_h1": 1126259462.42323,
}
EXPECTED_DETECTORS = ("H1", "L1")
EXPECTED_PARAM_DIM = {"kerr220": 4, "kerr221": 6, "kerr330": 6}
EXPECTED_COVERAGE_SUMMARY = Path("reports/posteriors/fig1_coverage_summary.json")
EXPECTED_PYRING_MANIFEST = Path("reports/posteriors/pyring/manifest_pyring.json")
NOISE_PATCH_TOKEN = "NoiseResamplingSNPE"


@dataclass
class Finding:
    ok: bool
    key: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "key": self.key, "detail": self.detail}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Fig.1 reproduction constraints")
    parser.add_argument("--check-artifacts", action="store_true", help="Also require runtime output artifacts")
    parser.add_argument("--run-root", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--pyring-dir", type=Path, default=Path("reports/posteriors/pyring"))
    parser.add_argument("--pyring-manifest", type=Path, default=EXPECTED_PYRING_MANIFEST)
    parser.add_argument("--overlay-summary", type=Path, default=Path("reports/posteriors/fig1_overlay_summary.json"))
    parser.add_argument("--coverage-summary", type=Path, default=EXPECTED_COVERAGE_SUMMARY)
    parser.add_argument("--output-json", type=Path, default=Path("reports/logs/fig1_paper_spec_audit.json"))
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _find_latest_summary(run_root: Path, case: str) -> Path | None:
    matches = sorted(run_root.rglob(f"{case}_run_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _check_injections() -> list[Finding]:
    out: list[Finding] = []
    for case in CASES:
        cfg_path = ROOT / "configs" / "injections" / f"{case}.yaml"
        if not cfg_path.exists():
            out.append(Finding(False, f"injection.{case}.exists", f"missing {cfg_path}"))
            continue
        cfg = _load_yaml(cfg_path)
        cfg_issues = validate_paper_case_config(cfg, case)
        out.append(Finding(not cfg_issues, f"injection.{case}.paper_case_validation", f"issues={cfg_issues}"))
        mass = float(cfg["remnant"]["mass_msun"])
        chi = float(cfg["remnant"]["chi_f"])
        sr = float(cfg["data"]["sample_rate_hz"])
        dur = float(cfg["data"]["duration_s"])
        target_snr = float(cfg["target_snr"])
        modes = {(int(m["l"]), int(m["m"]), int(m["n"])) for m in cfg["modes"]}
        mode_cfg = {(int(m["l"]), int(m["m"]), int(m["n"])): m for m in cfg["modes"]}
        source = cfg["source"]
        out.append(Finding(abs(mass - 67.0) < 1e-9, f"injection.{case}.mass", f"mass_msun={mass}"))
        out.append(Finding(abs(chi - 0.67) < 1e-9, f"injection.{case}.chi", f"chi_f={chi}"))
        out.append(Finding(abs(sr - 2048.0) < 1e-9, f"injection.{case}.sample_rate", f"sample_rate_hz={sr}"))
        out.append(Finding(abs(dur - 0.1) < 1e-9, f"injection.{case}.duration", f"duration_s={dur}"))
        out.append(Finding(modes == EXPECTED_MODES[case], f"injection.{case}.modes", f"modes={sorted(modes)}"))
        out.append(Finding(abs(target_snr - EXPECTED_SNR[case]) < 1e-9, f"injection.{case}.target_snr", f"target_snr={target_snr}"))
        for mode_id, expected in EXPECTED_MODE_PARAMS[case].items():
            mode = mode_cfg.get(mode_id, {})
            out.append(
                Finding(
                    abs(float(mode.get("amplitude", np.nan)) - expected["amplitude"]) < 1e-24,
                    f"injection.{case}.{mode_id}.amplitude",
                    f"amplitude={mode.get('amplitude')}",
                )
            )
            out.append(
                Finding(
                    abs(float(mode.get("phase", np.nan)) - expected["phase"]) < 1e-9,
                    f"injection.{case}.{mode_id}.phase",
                    f"phase={mode.get('phase')}",
                )
            )
        out.append(
            Finding(
                abs(float(source["inclination_rad"]) - EXPECTED_INCLINATION[case]) < 1e-9,
                f"injection.{case}.inclination",
                f"inclination_rad={source['inclination_rad']}",
            )
        )
        for key, expected in EXPECTED_SOURCE.items():
            out.append(
                Finding(
                    abs(float(source[key]) - expected) < 1e-9,
                    f"injection.{case}.{key}",
                    f"{key}={source[key]}",
                )
            )
        out.append(
            Finding(
                tuple(cfg["detectors"]) == EXPECTED_DETECTORS,
                f"injection.{case}.detectors",
                f"detectors={cfg['detectors']}",
            )
        )
        out.append(
            Finding(
                bool(cfg.get("use_detector_time_delay", False)) is True,
                f"injection.{case}.use_detector_time_delay",
                f"use_detector_time_delay={cfg.get('use_detector_time_delay')}",
            )
        )
        out.append(
            Finding(
                str(cfg.get("reference_detector", "")) == "H1",
                f"injection.{case}.reference_detector",
                f"reference_detector={cfg.get('reference_detector')}",
            )
        )
        out.append(
            Finding(
                str(cfg.get("qnm", {}).get("method", "")).strip().lower() == "fit",
                f"injection.{case}.kerr_mapping_reference45",
                f"qnm.method={cfg.get('qnm', {}).get('method')}",
            )
        )
    # Kerr330 inclination must differ from pi-like Kerr220/221
    c220 = _load_yaml(ROOT / "configs/injections/kerr220.yaml")
    c330 = _load_yaml(ROOT / "configs/injections/kerr330.yaml")
    inc220 = float(c220["source"]["inclination_rad"])
    inc330 = float(c330["source"]["inclination_rad"])
    out.append(Finding(abs(inc220 - inc330) > 1e-3, "injection.kerr330.inclination_differs", f"inc220={inc220}, inc330={inc330}"))
    return out


def _check_model_defaults() -> list[Finding]:
    cfg = _load_yaml(ROOT / "configs/model/tsnpe_nsf.yaml")
    emb = cfg["embedding"]
    den = cfg["density_estimator"]
    tsnpe = cfg["tsnpe"]
    tr = cfg["training"]
    out = [
        Finding(int(emb["input_dim"]) == 408, "model.embedding.input_dim", f"input_dim={emb['input_dim']}"),
        Finding(int(emb["num_hidden_layers"]) == 2, "model.embedding.num_hidden_layers", f"num_hidden_layers={emb['num_hidden_layers']}"),
        Finding(int(emb["hidden_dim"]) == 150, "model.embedding.hidden_dim", f"hidden_dim={emb['hidden_dim']}"),
        Finding(int(emb["output_dim"]) == 128, "model.embedding.output_dim", f"output_dim={emb['output_dim']}"),
        Finding(int(den["hidden_features"]) == 150, "model.nsf.hidden_features", f"hidden_features={den['hidden_features']}"),
        Finding(int(den["num_transforms"]) == 5, "model.nsf.num_transforms", f"num_transforms={den['num_transforms']}"),
        Finding(int(den["num_bins"]) == 10, "model.nsf.num_bins", f"num_bins={den['num_bins']}"),
        Finding(int(den["num_blocks"]) == 2, "model.nsf.num_blocks", f"num_blocks={den['num_blocks']}"),
        Finding(bool(den["batch_norm"]) is True, "model.nsf.batch_norm", f"batch_norm={den['batch_norm']}"),
        Finding(int(tr["training_batch_size"]) == 512, "model.training.batch", f"training_batch_size={tr['training_batch_size']}"),
        Finding(abs(float(tr["learning_rate"]) - 1e-3) < 1e-12, "model.training.learning_rate", f"learning_rate={tr['learning_rate']}"),
        Finding(abs(float(tr["validation_fraction"]) - 0.1) < 1e-12, "model.training.validation_fraction", f"validation_fraction={tr['validation_fraction']}"),
        Finding(int(tsnpe["num_simulations_first_round"]) == 50_000, "model.tsnpe.num_sim_first", f"num_simulations_first_round={tsnpe['num_simulations_first_round']}"),
        Finding(int(tsnpe["num_simulations_per_round"]) == 100_000, "model.tsnpe.num_sim_round", f"num_simulations_per_round={tsnpe['num_simulations_per_round']}"),
        Finding(abs(float(tsnpe["trunc_quantile"]) - 1e-4) < 1e-12, "model.tsnpe.trunc_quantile", f"trunc_quantile={tsnpe['trunc_quantile']}"),
        Finding(abs(float(tsnpe["stopping_ratio"]) - 0.8) < 1e-12, "model.tsnpe.stopping_ratio", f"stopping_ratio={tsnpe['stopping_ratio']}"),
    ]
    return out


def _check_script_defaults_and_prior() -> list[Finding]:
    text = (ROOT / "scripts/08_run_fig1_paper_precision.py").read_text(encoding="utf-8")
    out: list[Finding] = []
    patterns = {
        "script.default.num_sim_first": r"--num-sim-first\"\,\s*type=int,\s*default=50_000",
        "script.default.num_sim_round": r"--num-sim-round\"\,\s*type=int,\s*default=100_000",
        "script.default.trunc_quantile": r"--trunc-quantile\"\,\s*type=float,\s*default=1e-4",
        "script.default.stopping_ratio": r"--stopping-ratio\"\,\s*type=float,\s*default=0\.8",
        "script.default.batch_size": r"--training-batch-size\"\,\s*type=int,\s*default=512",
        "script.default.probe_samples": r"--truncation-probe-samples\"\,\s*type=int,\s*default=50_000",
        "script.default.min_rounds_before_stopping": r"--min-rounds-before-stopping\"\,\s*type=int,\s*default=3",
        "script.default.max_volume_for_stopping": r"--max-volume-for-stopping\"\,\s*type=float,\s*default=0\.95",
        "script.prior.mass_range": r"low = \[20\.0, 0\.0\].*high = \[300\.0, 0\.99\]",
        "script.prior.a220_floor": r"0\.1e-21 if \(l, m, n\) == \(2, 2, 0\) else 0\.0",
        "script.prior.mode_amplitude_high": r"high\.append\(50\.0e-21\)",
        "script.prior.phase_high": r"high\.append\(2\.0 \* np\.pi\)",
    }
    for key, pat in patterns.items():
        ok = re.search(pat, text, flags=re.DOTALL) is not None
        out.append(Finding(ok, key, f"pattern={pat}"))
    return out


def _check_tsnpe_hpd_impl() -> list[Finding]:
    text = (ROOT / "src/rd_sbi/inference/tsnpe_runner.py").read_text(encoding="utf-8")
    checks = [
        ("tsnpe.hpd.threshold_quantile", "torch.quantile(log_prob_post, self.config.trunc_quantile)"),
        ("tsnpe.hpd.accept_condition", "keep = logp >= threshold"),
        ("tsnpe.stop.rule", "return bool((current_volume / previous_volume) > stopping_ratio)"),
    ]
    return [Finding(token in text, key, f"token={token}") for key, token in checks]


def _check_data_pipeline() -> list[Finding]:
    whitening_text = (ROOT / "src/rd_sbi/noise/whitening.py").read_text(encoding="utf-8")
    fig1_text = (ROOT / "scripts/08_run_fig1_paper_precision.py").read_text(encoding="utf-8")
    patch_text = (ROOT / "src/rd_sbi/inference/sbi_loss_patch.py").read_text(encoding="utf-8")
    uses_exact_whitening = (
        "covariance_from_one_sided_psd" in fig1_text
        and "cholesky_lower_with_jitter" in fig1_text
        and "solve_triangular" in fig1_text
    )
    uses_varying_noise = "varying_noise_enabled=(not args.disable_varying_noise)" in fig1_text
    out = [
        Finding("acf_from_one_sided_psd" in whitening_text, "pipeline.whitening.has_psd_to_acf", "src/rd_sbi/noise/whitening.py"),
        Finding("toeplitz" in whitening_text, "pipeline.whitening.has_toeplitz_covariance", "src/rd_sbi/noise/whitening.py"),
        Finding("solve_triangular" in whitening_text and "cholesky_lower_with_jitter" in whitening_text, "pipeline.whitening.has_cholesky_whitener", "src/rd_sbi/noise/whitening.py"),
        Finding(uses_exact_whitening, "pipeline.fig1.uses_exact_whitening", "scripts/08_run_fig1_paper_precision.py"),
        Finding(NOISE_PATCH_TOKEN in patch_text, "pipeline.training.has_noise_resampling_impl", "src/rd_sbi/inference/sbi_loss_patch.py"),
        Finding(uses_varying_noise, "pipeline.training.uses_varying_noise", "scripts/08_run_fig1_paper_precision.py"),
        Finding('z_score_x="independent"' in fig1_text, "pipeline.training.x_standardization_matches_paper", "scripts/08_run_fig1_paper_precision.py"),
        Finding('z_score_theta=None' in fig1_text or 'z_score_theta="none"' in fig1_text, "pipeline.training.theta_normalization_matches_paper", "scripts/08_run_fig1_paper_precision.py"),
    ]
    return out


def _sample_bounds_ok(samples: np.ndarray, prior_low: np.ndarray, prior_high: np.ndarray) -> bool:
    if samples.ndim != 2 or prior_low.ndim != 1 or prior_high.ndim != 1:
        return False
    if samples.shape[1] != prior_low.shape[0] or samples.shape[1] != prior_high.shape[0]:
        return False
    return bool(np.all(samples >= prior_low[None, :]) and np.all(samples <= prior_high[None, :]))


def _check_artifacts(run_root: Path, pyring_dir: Path, overlay_summary: Path, coverage_summary: Path, pyring_manifest: Path) -> list[Finding]:
    out: list[Finding] = []
    found_cases: list[str] = []
    for case in CASES:
        summary_path = _find_latest_summary(run_root, case)
        if summary_path is None:
            out.append(Finding(False, f"artifact.{case}.run_summary", f"no summary under {run_root}"))
            continue
        found_cases.append(case)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        params = summary.get("params", {})
        out.append(Finding(int(params.get("num_sim_first", -1)) == 50_000, f"artifact.{case}.num_sim_first", f"{summary_path} -> {params.get('num_sim_first')}"))
        out.append(Finding(int(params.get("num_sim_round", -1)) == 100_000, f"artifact.{case}.num_sim_round", f"{summary_path} -> {params.get('num_sim_round')}"))
        out.append(Finding(abs(float(params.get("trunc_quantile", -1.0)) - 1e-4) < 1e-12, f"artifact.{case}.trunc_quantile", f"{summary_path} -> {params.get('trunc_quantile')}"))
        out.append(Finding(abs(float(params.get("stopping_ratio", -1.0)) - 0.8) < 1e-12, f"artifact.{case}.stopping_ratio", f"{summary_path} -> {params.get('stopping_ratio')}"))
        out.append(Finding(int(params.get("min_rounds_before_stopping", -1)) >= 3, f"artifact.{case}.min_rounds_before_stopping", f"{summary_path} -> {params.get('min_rounds_before_stopping')}"))
        out.append(Finding(bool(params.get("require_volume_shrink_for_stopping", False)) is True, f"artifact.{case}.require_volume_shrink_for_stopping", f"{summary_path} -> {params.get('require_volume_shrink_for_stopping')}"))
        out.append(Finding(bool(params.get("fail_on_no_truncation_for_next_round", False)) is True, f"artifact.{case}.fail_on_no_truncation_for_next_round", f"{summary_path} -> {params.get('fail_on_no_truncation_for_next_round')}"))
        out.append(Finding(int(params.get("truncation_probe_samples", -1)) >= 50_000, f"artifact.{case}.truncation_probe_samples", f"{summary_path} -> {params.get('truncation_probe_samples')}"))
        out.append(Finding("prior" in summary and "tsnpe_definition" in summary, f"artifact.{case}.audit_fields", f"{summary_path} includes prior/tsnpe_definition"))
        paper_case_validation = summary.get("paper_case_validation", {})
        out.append(
            Finding(
                bool(paper_case_validation.get("passed", False)) is True,
                f"artifact.{case}.paper_case_validation",
                f"{summary_path} -> paper_case_validation={paper_case_validation}",
            )
        )
        preprocessing = summary.get("preprocessing", {})
        training_contract = summary.get("training_contract", {})
        out.append(
            Finding(
                bool(preprocessing.get("appendix_a_faithful", False)) is True,
                f"artifact.{case}.appendix_a_faithful",
                f"{summary_path} -> appendix_a_faithful={preprocessing.get('appendix_a_faithful')}",
            )
        )
        out.append(
            Finding(
                preprocessing.get("input_dim") == 408 and preprocessing.get("per_detector_bins") == 204,
                f"artifact.{case}.input_shape_contract",
                f"{summary_path} -> input_dim={preprocessing.get('input_dim')} per_detector_bins={preprocessing.get('per_detector_bins')}",
            )
        )
        out.append(
            Finding(
                preprocessing.get("whitening_method") == "psd_to_acf_to_toeplitz_to_cholesky",
                f"artifact.{case}.whitening_method",
                f"{summary_path} -> whitening_method={preprocessing.get('whitening_method')}",
            )
        )
        x_norm = preprocessing.get("x_standardization", {})
        theta_norm = preprocessing.get("theta_normalization", {})
        out.append(Finding(bool(x_norm.get("paper_faithful", False)) is True, f"artifact.{case}.x_standardization", f"{summary_path} -> x_standardization={x_norm}"))
        out.append(Finding(bool(theta_norm.get("paper_faithful", False)) is True, f"artifact.{case}.theta_normalization", f"{summary_path} -> theta_normalization={theta_norm}"))
        varying_noise = training_contract.get("varying_noise", {})
        out.append(Finding(bool(varying_noise.get("enabled", False)) is True, f"artifact.{case}.varying_noise_enabled", f"{summary_path} -> varying_noise={varying_noise}"))
        waveform_model = summary.get("waveform_model", {})
        kerr_mapping = waveform_model.get("kerr_mapping", {})
        out.append(
            Finding(
                bool(kerr_mapping.get("paper_faithful", False)) is True,
                f"artifact.{case}.kerr_mapping_reference45",
                f"{summary_path} -> kerr_mapping={kerr_mapping}",
            )
        )
        snr_block = summary.get("snr", {})
        snr_ok = (
            isinstance(snr_block, dict)
            and snr_block.get("measured_network_snr") is not None
            and snr_block.get("per_detector") is not None
            and snr_block.get("psd_source") is not None
        )
        out.append(Finding(snr_ok, f"artifact.{case}.snr_summary_complete", f"{summary_path} -> snr={snr_block}"))
        out.append(
            Finding(
                bool(snr_block.get("paper_faithful", False)) is True,
                f"artifact.{case}.snr_matches_target",
                f"{summary_path} -> snr={snr_block}",
            )
        )
        injection_context = summary.get("injection_context", {})
        out.append(
            Finding(
                abs(float(injection_context.get("ra_rad", np.nan)) - EXPECTED_SOURCE["ra_rad"]) < 1e-9,
                f"artifact.{case}.ra_rad",
                f"{summary_path} -> {injection_context.get('ra_rad')}",
            )
        )
        out.append(
            Finding(
                abs(float(injection_context.get("dec_rad", np.nan)) - EXPECTED_SOURCE["dec_rad"]) < 1e-9,
                f"artifact.{case}.dec_rad",
                f"{summary_path} -> {injection_context.get('dec_rad')}",
            )
        )
        out.append(
            Finding(
                abs(float(injection_context.get("psi_rad", np.nan)) - EXPECTED_SOURCE["psi_rad"]) < 1e-9,
                f"artifact.{case}.psi_rad",
                f"{summary_path} -> {injection_context.get('psi_rad')}",
            )
        )
        out.append(
            Finding(
                abs(float(injection_context.get("gps_h1", np.nan)) - EXPECTED_SOURCE["gps_h1"]) < 1e-9,
                f"artifact.{case}.gps_h1",
                f"{summary_path} -> {injection_context.get('gps_h1')}",
            )
        )
        param_order = summary.get("prior", {}).get("param_order", [])
        out.append(
            Finding(
                len(param_order) == EXPECTED_PARAM_DIM[case],
                f"artifact.{case}.param_dim",
                f"{summary_path} -> len(param_order)={len(param_order)}",
            )
        )
        npz_path = Path(summary.get("output_npz", ""))
        out.append(Finding(npz_path.exists(), f"artifact.{case}.posterior_npz_exists", str(npz_path)))
        if npz_path.exists():
            samples, theta_true = load_posterior_npz(npz_path)
            with np.load(npz_path, allow_pickle=True) as npz:
                x_observed = np.asarray(npz["x_observed"], dtype=np.float64) if "x_observed" in npz else None
            n_samples = int(samples.shape[0])
            out.append(Finding(n_samples == 20_000, f"artifact.{case}.posterior_samples_20k", f"n_samples={n_samples}"))
            prior_low = np.asarray(summary.get("prior", {}).get("low", []), dtype=np.float64)
            prior_high = np.asarray(summary.get("prior", {}).get("high", []), dtype=np.float64)
            out.append(Finding(_sample_bounds_ok(samples, prior_low, prior_high), f"artifact.{case}.samples_within_prior", str(npz_path)))
            out.append(
                Finding(
                    x_observed is not None and x_observed.shape == (408,),
                    f"artifact.{case}.x_observed_shape",
                    f"x_observed_shape={None if x_observed is None else x_observed.shape}",
                )
            )
            issues = paper_grade_run_issues(summary, samples=samples, theta_true=theta_true)
            out.append(
                Finding(
                    not issues,
                    f"artifact.{case}.paper_grade_quality",
                    "ok" if not issues else ",".join(issues),
                )
            )

    out.append(
        Finding(
            set(found_cases) == set(CASES),
            "artifact.paper_grade.all_cases_present",
            f"found_cases={sorted(found_cases)}",
        )
    )

    for case in CASES:
        p = pyring_dir / f"{case}_pyring.npz"
        out.append(Finding(p.exists(), f"artifact.{case}.pyring_npz_exists", str(p)))

    out.append(Finding(pyring_manifest.exists(), "artifact.pyring.manifest_exists", str(pyring_manifest)))
    if pyring_manifest.exists():
        s = json.loads(pyring_manifest.read_text(encoding="utf-8"))
        out.append(Finding(str(s.get("pyring_version")) == "2.3.0", "artifact.pyring.version", f"version={s.get('pyring_version')}"))
        out.append(Finding(int(s.get("cpnest_live_points", -1)) == 4096, "artifact.pyring.live_points", f"cpnest_live_points={s.get('cpnest_live_points')}"))
        out.append(Finding(int(s.get("cpnest_max_mcmc_steps", -1)) == 4094, "artifact.pyring.max_mcmc_steps", f"cpnest_max_mcmc_steps={s.get('cpnest_max_mcmc_steps')}"))

    out.append(Finding(overlay_summary.exists(), "artifact.overlay.summary_exists", str(overlay_summary)))
    if overlay_summary.exists():
        s = json.loads(overlay_summary.read_text(encoding="utf-8"))
        case_block = s.get("cases", {})
        out.append(
            Finding(
                set(case_block.keys()) == set(CASES),
                "artifact.overlay.all_cases_present",
                f"overlay_cases={sorted(case_block.keys())}",
            )
        )
        for case in CASES:
            ok = case in case_block and "wasserstein_Mf" in case_block.get(case, {}) and "ks_Mf_D" in case_block.get(case, {})
            out.append(Finding(ok, f"artifact.overlay.{case}.metrics", f"metrics_present={ok}"))
            counts_ok = (
                case in case_block
                and int(case_block.get(case, {}).get("sbi_samples", -1)) == 20_000
                and int(case_block.get(case, {}).get("pyring_samples", -1)) == 20_000
            )
            out.append(Finding(counts_ok, f"artifact.overlay.{case}.sample_counts_20k", f"counts={case_block.get(case, {})}"))
    out.append(Finding(coverage_summary.exists(), "artifact.coverage.summary_exists", str(coverage_summary)))
    if coverage_summary.exists():
        s = json.loads(coverage_summary.read_text(encoding="utf-8"))
        out.append(Finding(int(s.get("num_catalogs", -1)) == 100, "artifact.coverage.num_catalogs", f"num_catalogs={s.get('num_catalogs')}"))
        out.append(Finding(int(s.get("draws_per_catalog", -1)) == 100, "artifact.coverage.draws_per_catalog", f"draws_per_catalog={s.get('draws_per_catalog')}"))
        out.append(Finding(str(s.get("test_prior")) == "final_truncated_prior", "artifact.coverage.test_prior", f"test_prior={s.get('test_prior')}"))
        out.append(Finding(str(s.get("coverage_definition")) == "hpd_gamma", "artifact.coverage.definition", f"coverage_definition={s.get('coverage_definition')}"))
    return out


def _grade_from_findings(findings: list[Finding], *, check_artifacts: bool) -> dict[str, Any]:
    failed = sorted(f.key for f in findings if not f.ok)
    if not check_artifacts:
        return {
            "grade": "preflight-only",
            "reason": "artifact checks were not requested",
            "failed_keys": failed,
        }

    invalid_prefixes = (
        "artifact.paper_grade.all_cases_present",
        "artifact.kerr220.paper_grade_quality",
        "artifact.kerr221.paper_grade_quality",
        "artifact.kerr330.paper_grade_quality",
        "artifact.kerr220.samples_within_prior",
        "artifact.kerr221.samples_within_prior",
        "artifact.kerr330.samples_within_prior",
        "artifact.kerr220.pyring_npz_exists",
        "artifact.kerr221.pyring_npz_exists",
        "artifact.kerr330.pyring_npz_exists",
        "artifact.pyring.",
        "artifact.overlay.",
        "artifact.kerr220.snr_matches_target",
        "artifact.kerr221.snr_matches_target",
        "artifact.kerr330.snr_matches_target",
    )
    if any(key.startswith(invalid_prefixes) for key in failed):
        return {
            "grade": "D",
            "reason": "runtime artifacts are invalid for paper comparison",
            "failed_keys": failed,
        }

    if not failed:
        return {
            "grade": "A",
            "reason": "all implemented paper-faithful checks passed",
            "failed_keys": failed,
        }

    coverage_prefixes = ("artifact.coverage.",)
    if all(key.startswith(coverage_prefixes) for key in failed):
        return {
            "grade": "B",
            "reason": "figure-faithful checks passed; only Appendix B coverage is still incomplete",
            "failed_keys": failed,
        }

    return {
        "grade": "C",
        "reason": "pipeline is partially validated, but paper-faithful requirements remain incomplete",
        "failed_keys": failed,
    }


def main() -> None:
    args = parse_args()
    run_root = (ROOT / args.run_root).resolve() if not args.run_root.is_absolute() else args.run_root
    pyring_dir = (ROOT / args.pyring_dir).resolve() if not args.pyring_dir.is_absolute() else args.pyring_dir
    pyring_manifest = (ROOT / args.pyring_manifest).resolve() if not args.pyring_manifest.is_absolute() else args.pyring_manifest
    overlay_summary = (ROOT / args.overlay_summary).resolve() if not args.overlay_summary.is_absolute() else args.overlay_summary
    coverage_summary = (ROOT / args.coverage_summary).resolve() if not args.coverage_summary.is_absolute() else args.coverage_summary
    output_json = (ROOT / args.output_json).resolve() if not args.output_json.is_absolute() else args.output_json

    findings: list[Finding] = []
    findings.extend(_check_injections())
    findings.extend(_check_model_defaults())
    findings.extend(_check_script_defaults_and_prior())
    findings.extend(_check_tsnpe_hpd_impl())
    findings.extend(_check_data_pipeline())
    if args.check_artifacts:
        findings.extend(_check_artifacts(run_root, pyring_dir, overlay_summary, coverage_summary, pyring_manifest))

    ok = all(f.ok for f in findings)
    grade = _grade_from_findings(findings, check_artifacts=bool(args.check_artifacts))
    report = {
        "overall_ok": ok,
        "check_artifacts": bool(args.check_artifacts),
        "reproduction_grade": grade,
        "findings": [f.as_dict() for f in findings],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved audit report: {output_json}")
    print(f"overall_ok={ok}")
    if not ok:
        bad = [f.key for f in findings if not f.ok]
        print(f"failed_keys={bad}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
