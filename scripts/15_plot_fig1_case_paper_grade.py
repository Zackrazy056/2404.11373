"""Plot a single Fig.1 case with paper-grade contour style."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.eval.fig1_quality import load_posterior_npz, load_summary_json, paper_grade_run_issues  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one paper-grade Fig.1 case")
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output-figure", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=90)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path)


def _contour_threshold(hist: np.ndarray, credibility: float) -> float:
    flat = hist.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order]) / np.sum(flat)
    idx = np.searchsorted(cdf, credibility)
    idx = min(idx, flat.size - 1)
    return float(flat[order[idx]])


def _summary_path_from_args(args: argparse.Namespace) -> Path:
    if args.summary_path is not None:
        return _resolve(args.summary_path)
    if args.run_dir is None:
        raise ValueError("Provide either --summary-path or --run-dir")
    return _resolve(args.run_dir) / f"{args.case}_run_summary.json"


def _valid_samples_within_prior(
    samples: np.ndarray,
    prior_low: np.ndarray,
    prior_high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.all((samples >= prior_low[None, :]) & (samples <= prior_high[None, :]), axis=1)
    return samples[mask], mask


def _axis_limits(values: np.ndarray, truth: float, low: float, high: float) -> tuple[float, float]:
    if values.size == 0:
        return float(low), float(high)
    data_lo = min(float(np.min(values)), float(truth))
    data_hi = max(float(np.max(values)), float(truth))
    width = max(data_hi - data_lo, 1e-6)
    pad = 0.08 * width
    return max(float(low), data_lo - pad), min(float(high), data_hi + pad)


def main() -> None:
    args = parse_args()
    summary_path = _summary_path_from_args(args)
    summary = load_summary_json(summary_path)
    npz_path = Path(summary["output_npz"])
    samples, theta_true = load_posterior_npz(npz_path)
    issues = paper_grade_run_issues(summary, samples=samples, theta_true=theta_true)
    if issues:
        raise RuntimeError(f"Run is not paper-grade: {issues}")

    prior_low = np.asarray(summary["prior"]["low"], dtype=float)
    prior_high = np.asarray(summary["prior"]["high"], dtype=float)
    valid_samples, valid_mask = _valid_samples_within_prior(samples, prior_low, prior_high)
    if valid_samples.shape[0] == 0:
        raise RuntimeError("No posterior samples remain after enforcing physical prior bounds")

    m = valid_samples[:, 0]
    chi = valid_samples[:, 1]
    hist, x_edges, y_edges = np.histogram2d(m, chi, bins=args.bins, density=True)
    hist = gaussian_filter(hist, sigma=args.smooth_sigma)
    lv90 = _contour_threshold(hist, 0.90)
    lv68 = _contour_threshold(hist, 0.68)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")

    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ax.contourf(xx, yy, hist, levels=14, cmap="Blues", alpha=0.35)
    ax.contour(xx, yy, hist, levels=sorted([lv90, lv68]), colors=["#1f77b4", "#0b4ea2"], linewidths=[1.4, 2.2])
    ax.axvline(float(theta_true[0]), color="black", linestyle="--", linewidth=1.2)
    ax.axhline(float(theta_true[1]), color="black", linestyle="--", linewidth=1.2)
    ax.set_title(args.case.upper(), fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$M_f\ [M_\odot]$")
    ax.set_ylabel(r"$\chi_f$")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_xlim(*_axis_limits(m, float(theta_true[0]), float(prior_low[0]), float(prior_high[0])))
    ax.set_ylim(*_axis_limits(chi, float(theta_true[1]), float(prior_low[1]), float(prior_high[1])))

    out_fig = _resolve(args.output_figure)
    out_json = _resolve(args.output_summary)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=320)
    plt.close(fig)

    payload = {
        "case": args.case,
        "summary_path": str(summary_path),
        "npz_path": str(npz_path),
        "output_figure": str(out_fig),
        "issues": [],
        "m_ci90": [float(x) for x in np.percentile(m, [5, 95]).tolist()],
        "chi_ci90": [float(x) for x in np.percentile(chi, [5, 95]).tolist()],
        "num_samples_total": int(samples.shape[0]),
        "num_samples_in_prior": int(valid_samples.shape[0]),
        "num_samples_outside_prior": int((~valid_mask).sum()),
        "axis_limits": {
            "m": [float(v) for v in ax.get_xlim()],
            "chi": [float(v) for v in ax.get_ylim()],
        },
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved figure: {out_fig}")
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
