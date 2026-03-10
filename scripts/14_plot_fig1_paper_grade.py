"""Generate paper-grade Fig.1 style posterior contours from paper-precision runs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.eval.fig1_quality import CASES, load_posterior_npz, load_summary_json, paper_grade_run_issues  # noqa: E402


@dataclass(frozen=True)
class CasePosterior:
    case: str
    summary_path: Path
    npz_path: Path
    m: np.ndarray
    chi: np.ndarray
    m_true: float
    chi_true: float
    total_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot paper-grade Fig.1 posterior contours")
    parser.add_argument("--run-root", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--cases", type=str, default="kerr220,kerr221,kerr330")
    parser.add_argument("--overlay-summary", type=Path, default=None)
    parser.add_argument("--output-figure", type=Path, default=Path("reports/figures/fig1_paper_sbi.png"))
    parser.add_argument("--output-summary", type=Path, default=Path("reports/figures/fig1_paper_sbi_plot_summary.json"))
    parser.add_argument("--bins", type=int, default=90)
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path)


def _resolve_optional(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (ROOT / path)


def _find_latest_case(run_root: Path, case: str) -> tuple[Path | None, str | None]:
    all_summaries = sorted(run_root.rglob(f"{case}_run_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in all_summaries:
        try:
            s = load_summary_json(p)
            npz_path = Path(s["output_npz"])
            samples, theta_true = load_posterior_npz(npz_path)
            issues = paper_grade_run_issues(s, samples=samples, theta_true=theta_true)
        except Exception:  # noqa: BLE001
            continue
        if not issues:
            return p, None
    if not all_summaries:
        return None, "no run_summary found"
    return None, "no paper-grade run found"


def _load_case(summary_path: Path, case: str) -> CasePosterior:
    s = load_summary_json(summary_path)
    npz_path = Path(s["output_npz"])
    samples, theta_true = load_posterior_npz(npz_path)
    return CasePosterior(
        case=case,
        summary_path=summary_path,
        npz_path=npz_path,
        m=samples[:, 0],
        chi=samples[:, 1],
        m_true=float(theta_true[0]),
        chi_true=float(theta_true[1]),
        total_seconds=float(s.get("total_seconds", float("nan"))),
    )


def _cred_level_threshold(hist: np.ndarray, credibility: float) -> float:
    flat = hist.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order]) / np.sum(flat)
    idx = np.searchsorted(cdf, credibility)
    idx = min(idx, flat.size - 1)
    return float(flat[order[idx]])


def _plot_case(ax: plt.Axes, c: CasePosterior, bins: int, smooth_sigma: float) -> dict[str, float]:
    finite = np.isfinite(c.m) & np.isfinite(c.chi)
    m = c.m[finite]
    chi = c.chi[finite]
    hist, x_edges, y_edges = np.histogram2d(m, chi, bins=bins, density=True)
    hist = gaussian_filter(hist, sigma=smooth_sigma)
    lv90 = _cred_level_threshold(hist, 0.90)
    lv68 = _cred_level_threshold(hist, 0.68)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")

    ax.contourf(xx, yy, hist, levels=14, cmap="Blues", alpha=0.35)
    ax.contour(xx, yy, hist, levels=sorted([lv90, lv68]), colors=["#1f77b4", "#0b4ea2"], linewidths=[1.4, 2.2])
    ax.axvline(c.m_true, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(c.chi_true, color="black", linestyle="--", linewidth=1.2)
    ax.set_title(c.case.upper(), fontsize=12, fontweight="bold")
    ax.set_xlabel(r"$M_f\ [M_\odot]$")
    ax.set_ylabel(r"$\chi_f$")
    ax.grid(alpha=0.18, linewidth=0.6)

    m_p50 = float(np.percentile(m, 50))
    c_p50 = float(np.percentile(chi, 50))
    m_ci = np.percentile(m, [5, 95]).tolist()
    c_ci = np.percentile(chi, [5, 95]).tolist()
    ax.text(
        0.02,
        0.98,
        f"N={m.size}\n$M_f^{{50}}$={m_p50:.2f}\n$\\chi_f^{{50}}$={c_p50:.4f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75, edgecolor="none"),
    )

    return {
        "n_samples": int(m.size),
        "m_p50": m_p50,
        "chi_p50": c_p50,
        "m_ci90": [float(m_ci[0]), float(m_ci[1])],
        "chi_ci90": [float(c_ci[0]), float(c_ci[1])],
        "total_seconds": float(c.total_seconds),
        "summary_path": str(c.summary_path),
        "npz_path": str(c.npz_path),
    }


def main() -> None:
    args = parse_args()
    run_root = _resolve(args.run_root)
    overlay_summary = _resolve_optional(args.overlay_summary) or (run_root / "overlay_summary.json")
    selected = [x.strip().lower() for x in args.cases.split(",") if x.strip()]
    selected = [x for x in selected if x in CASES]
    if not selected:
        raise ValueError(f"No valid case selected. choices={CASES}")
    if set(selected) != set(CASES):
        raise ValueError(f"Paper-grade Fig.1 requires all cases: {CASES}")
    if not overlay_summary.exists():
        raise RuntimeError(f"Paper-grade Fig.1 requires overlay summary: {overlay_summary}")
    overlay = json.loads(overlay_summary.read_text(encoding="utf-8"))
    overlay_cases = set(overlay.get("cases", {}).keys())
    if overlay_cases != set(selected):
        raise RuntimeError(f"Overlay summary missing cases: expected={sorted(selected)} got={sorted(overlay_cases)}")

    loaded: list[CasePosterior] = []
    failures: dict[str, str] = {}
    for case in selected:
        summary, reason = _find_latest_case(run_root, case)
        if summary is None:
            failures[case] = reason or "no paper-grade run found"
            continue
        try:
            loaded.append(_load_case(summary, case))
        except Exception as exc:  # noqa: BLE001
            failures[case] = str(exc)

    if failures:
        raise RuntimeError(f"Paper-grade Fig.1 requires all cases. failures={failures}")
    if len(loaded) != len(selected):
        raise RuntimeError(f"Paper-grade Fig.1 requires all cases. loaded={[c.case for c in loaded]}")

    fig, axes = plt.subplots(1, len(loaded), figsize=(5.0 * len(loaded), 4.4), constrained_layout=True)
    if len(loaded) == 1:
        axes = [axes]

    metrics: dict[str, Any] = {"cases": {}, "overlay_summary": str(overlay_summary)}
    for ax, case_data in zip(axes, loaded):
        metrics["cases"][case_data.case] = _plot_case(ax, case_data, bins=args.bins, smooth_sigma=args.smooth_sigma)

    fig.suptitle("Fig.1 Reproduction (SBI, Paper-Grade Runs)", fontsize=13, fontweight="bold")
    out_fig = _resolve(args.output_figure)
    out_json = _resolve(args.output_summary)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=320)
    plt.close(fig)
    out_json.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved figure: {out_fig}")
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
