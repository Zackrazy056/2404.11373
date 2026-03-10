"""Quality gates for Fig.1 paper-grade reproduction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


CASES = ("kerr220", "kerr221", "kerr330")
REQUIRED_DIAGNOSTIC_KEYS = {
    "round_index",
    "num_simulations",
    "truncated_prior_volume",
    "probe_acceptance_rate",
    "volume_ratio_to_previous",
    "stop_by_ratio",
    "stop_eligible",
    "stop_reason",
}
EFFECTIVE_SHRINK_THRESHOLD = 0.999
MAX_CI90_FRACTION_OF_PRIOR = 0.9
STRICT_EARLY_STOP_MAX_FINAL_VOLUME = 0.5
STRICT_EARLY_STOP_MAX_PREVIOUS_VOLUME = 0.8


def summary_has_paper_budget(summary: dict[str, Any]) -> bool:
    params = summary.get("params", {})
    return (
        int(params.get("num_sim_first", -1)) == 50_000
        and int(params.get("num_sim_round", -1)) == 100_000
        and abs(float(params.get("trunc_quantile", -1.0)) - 1e-4) < 1e-12
        and abs(float(params.get("stopping_ratio", -1.0)) - 0.8) < 1e-12
        and int(params.get("posterior_samples", -1)) == 20_000
    )


def diagnostics_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics = summary.get("diagnostics", [])
    if not isinstance(diagnostics, list):
        return []
    return [diag for diag in diagnostics if isinstance(diag, dict)]


def diagnostics_are_complete(diagnostics: list[dict[str, Any]]) -> bool:
    return bool(diagnostics) and all(REQUIRED_DIAGNOSTIC_KEYS.issubset(diag.keys()) for diag in diagnostics)


def diagnostic_has_effective_shrinkage(
    diagnostic: dict[str, Any],
    *,
    threshold: float = EFFECTIVE_SHRINK_THRESHOLD,
) -> bool:
    try:
        volume = float(diagnostic["truncated_prior_volume"])
        probe_acceptance = float(diagnostic["probe_acceptance_rate"])
    except (KeyError, TypeError, ValueError):
        return False
    return volume < threshold and probe_acceptance < threshold


def run_has_strict_round_termination(summary: dict[str, Any], diagnostics: list[dict[str, Any]]) -> bool:
    if len(diagnostics) >= 3:
        return True
    if len(diagnostics) < 2:
        return False

    params = summary.get("params", {})
    stopping_ratio = float(params.get("stopping_ratio", 0.8))
    previous_diag = diagnostics[-2]
    latest_diag = diagnostics[-1]
    ratio = latest_diag.get("volume_ratio_to_previous")
    if ratio is None:
        return False

    return (
        bool(latest_diag.get("stop_by_ratio", False))
        and bool(latest_diag.get("stop_eligible", False))
        and diagnostic_has_effective_shrinkage(previous_diag)
        and diagnostic_has_effective_shrinkage(latest_diag)
        and float(previous_diag["truncated_prior_volume"]) <= STRICT_EARLY_STOP_MAX_PREVIOUS_VOLUME
        and float(latest_diag["truncated_prior_volume"]) <= STRICT_EARLY_STOP_MAX_FINAL_VOLUME
        and float(ratio) > stopping_ratio
    )


def credible_region_issues(
    samples: np.ndarray,
    theta_true: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
) -> list[str]:
    issues: list[str] = []
    if samples.ndim != 2 or samples.shape[1] < 2:
        return ["samples_shape_invalid"]
    if theta_true.size < 2 or prior_low.size < 2 or prior_high.size < 2:
        return ["truth_or_prior_missing"]

    labels = ("M_f_msun", "chi_f")
    for idx, label in enumerate(labels):
        values = np.asarray(samples[:, idx], dtype=np.float64)
        truth = float(theta_true[idx])
        prior_width = float(prior_high[idx] - prior_low[idx])
        lo, hi = np.percentile(values, [5.0, 95.0]).tolist()
        if not (lo <= truth <= hi):
            issues.append(f"{label}.truth_outside_90")
        if prior_width <= 0.0:
            issues.append(f"{label}.prior_width_invalid")
            continue
        ci_fraction = (float(hi) - float(lo)) / prior_width
        if ci_fraction >= MAX_CI90_FRACTION_OF_PRIOR:
            issues.append(f"{label}.ci90_too_wide")
    return issues


def paper_grade_run_issues(
    summary: dict[str, Any],
    *,
    samples: np.ndarray | None = None,
    theta_true: np.ndarray | None = None,
) -> list[str]:
    issues: list[str] = []
    if not summary_has_paper_budget(summary):
        issues.append("not_paper_budget")

    diagnostics = diagnostics_from_summary(summary)
    if not diagnostics_are_complete(diagnostics):
        issues.append("diagnostics_incomplete")
        return issues

    if not any(diagnostic_has_effective_shrinkage(diag) for diag in diagnostics):
        issues.append("no_effective_shrinkage")
    if not diagnostic_has_effective_shrinkage(diagnostics[-1]):
        issues.append("final_round_no_effective_shrinkage")
    if not run_has_strict_round_termination(summary, diagnostics):
        issues.append("insufficient_rounds_or_weak_early_stop")

    if samples is not None and theta_true is not None:
        prior = summary.get("prior", {})
        prior_low = np.asarray(prior.get("low", []), dtype=np.float64)
        prior_high = np.asarray(prior.get("high", []), dtype=np.float64)
        issues.extend(credible_region_issues(samples, theta_true, prior_low, prior_high))

    return issues


def load_summary_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def load_posterior_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as npz:
        samples = np.asarray(npz["samples"], dtype=np.float64)
        theta_true = np.asarray(npz["theta_true"], dtype=np.float64).reshape(-1)
    return samples, theta_true
