"""Overlay Fig.1 SBI posteriors with pyRing posteriors and report metrics.

Expected SBI files (one of):
  <sbi-dir>/<case>_sbi_posterior_20000.npz
  <sbi-dir>/<case>_sbi_posterior.npz

Expected pyRing files:
  <pyring-dir>/<case>_pyring.npz
with any compatible key pair for mass/spin, e.g.:
  M + chi, Mf_msun + chi_f, mass + spin.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import ks_2samp, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]

CASES = ("kerr220", "kerr221", "kerr330")
M_KEYS = ("M", "Mf", "Mf_msun", "M_f_msun", "mass", "mf")
CHI_KEYS = ("chi", "chi_f", "spin", "chi_final", "chif")


@dataclass(frozen=True)
class PosteriorPair:
    case: str
    m_sbi: np.ndarray
    chi_sbi: np.ndarray
    m_pyr: np.ndarray
    chi_pyr: np.ndarray
    m_true: float | None
    chi_true: float | None
    sbi_path: Path
    pyring_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overlay SBI and pyRing posteriors for Fig.1")
    parser.add_argument("--cases", type=str, default="kerr220,kerr221,kerr330")
    parser.add_argument("--sbi-dir", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--pyring-dir", type=Path, default=Path("reports/posteriors/pyring"))
    parser.add_argument("--output-figure", type=Path, default=Path("reports/figures/fig1_overlay_sbi_pyring.png"))
    parser.add_argument("--summary-path", type=Path, default=Path("reports/posteriors/fig1_overlay_summary.json"))
    parser.add_argument("--smooth-sigma", type=float, default=1.0)
    parser.add_argument("--bins", type=int, default=80)
    return parser.parse_args()


def _find_existing(*candidates: Path) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def _extract_by_keys(obj: Any, keys: tuple[str, ...]) -> np.ndarray | None:
    for k in keys:
        if k in obj:
            arr = np.asarray(obj[k], dtype=np.float64).reshape(-1)
            return arr
    return None


def _load_sbi_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float | None, float | None]:
    z = np.load(path, allow_pickle=True)
    if "samples" not in z:
        raise ValueError(f"{path} missing 'samples'")
    samples = np.asarray(z["samples"], dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError(f"{path} has invalid samples shape {samples.shape}")
    m_true = None
    chi_true = None
    if "theta_true" in z:
        t = np.asarray(z["theta_true"], dtype=np.float64).reshape(-1)
        if t.size >= 2:
            m_true, chi_true = float(t[0]), float(t[1])
    elif "true_theta" in z:
        t = np.asarray(z["true_theta"], dtype=np.float64).reshape(-1)
        if t.size >= 2:
            m_true, chi_true = float(t[0]), float(t[1])
    return samples[:, 0], samples[:, 1], m_true, chi_true


def _load_pyring_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    m = _extract_by_keys(z, M_KEYS)
    chi = _extract_by_keys(z, CHI_KEYS)
    if m is None or chi is None:
        raise ValueError(f"{path} cannot find mass/spin keys from {M_KEYS} and {CHI_KEYS}")
    keep = np.isfinite(m) & np.isfinite(chi)
    return m[keep], chi[keep]


def _contour_threshold(hist: np.ndarray, credibility: float) -> float:
    flat = hist.ravel()
    if flat.size == 0 or np.sum(flat) <= 0:
        return 0.0
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order]) / np.sum(flat)
    idx = np.searchsorted(cdf, credibility)
    idx = min(idx, flat.size - 1)
    return float(flat[order[idx]])


def _draw_contours(
    ax: plt.Axes,
    m: np.ndarray,
    chi: np.ndarray,
    *,
    bins: int,
    smooth_sigma: float,
    color: str,
    linewidths: tuple[float, float],
) -> None:
    if m.size < 20:
        ax.scatter(m, chi, s=3, alpha=0.2, c=color)
        return
    hist, x_edges, y_edges = np.histogram2d(m, chi, bins=bins, density=True)
    hist = gaussian_filter(hist, sigma=smooth_sigma)
    lv90 = _contour_threshold(hist, 0.90)
    lv68 = _contour_threshold(hist, 0.68)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
    levels = sorted([lv90, lv68])
    ax.contour(xx, yy, hist, levels=levels, colors=[color, color], linewidths=list(linewidths))


def _area_ratio_90(pair: PosteriorPair, bins: int, smooth_sigma: float) -> float:
    m_all = np.concatenate([pair.m_sbi, pair.m_pyr])
    chi_all = np.concatenate([pair.chi_sbi, pair.chi_pyr])
    m_min, m_max = float(np.min(m_all)), float(np.max(m_all))
    c_min, c_max = float(np.min(chi_all)), float(np.max(chi_all))
    m_pad = max((m_max - m_min) * 0.05, 1e-6)
    c_pad = max((c_max - c_min) * 0.05, 1e-6)
    m_range = [m_min - m_pad, m_max + m_pad]
    c_range = [c_min - c_pad, c_max + c_pad]

    hs, x_edges, y_edges = np.histogram2d(pair.m_sbi, pair.chi_sbi, bins=bins, range=[m_range, c_range], density=True)
    hp, _, _ = np.histogram2d(pair.m_pyr, pair.chi_pyr, bins=bins, range=[m_range, c_range], density=True)
    hs = gaussian_filter(hs, sigma=smooth_sigma)
    hp = gaussian_filter(hp, sigma=smooth_sigma)
    ts = _contour_threshold(hs, 0.90)
    tp = _contour_threshold(hp, 0.90)
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])
    area_s = float(np.sum(hs >= ts) * dx * dy)
    area_p = float(np.sum(hp >= tp) * dx * dy)
    if area_p <= 0:
        return float("nan")
    return area_s / area_p


def _collect_pair(case: str, sbi_dir: Path, pyring_dir: Path) -> PosteriorPair:
    sbi_file = _find_existing(
        sbi_dir / f"{case}_sbi_posterior_20000.npz",
        sbi_dir / f"{case}_sbi_posterior.npz",
    )
    if sbi_file is None:
        raise FileNotFoundError(f"SBI posterior missing for {case} in {sbi_dir}")
    pyr_file = pyring_dir / f"{case}_pyring.npz"
    if not pyr_file.exists():
        raise FileNotFoundError(f"pyRing posterior missing: {pyr_file}")
    m_sbi, chi_sbi, m_true, chi_true = _load_sbi_npz(sbi_file)
    m_pyr, chi_pyr = _load_pyring_npz(pyr_file)
    keep_s = np.isfinite(m_sbi) & np.isfinite(chi_sbi)
    keep_p = np.isfinite(m_pyr) & np.isfinite(chi_pyr)
    return PosteriorPair(
        case=case,
        m_sbi=m_sbi[keep_s],
        chi_sbi=chi_sbi[keep_s],
        m_pyr=m_pyr[keep_p],
        chi_pyr=chi_pyr[keep_p],
        m_true=m_true,
        chi_true=chi_true,
        sbi_path=sbi_file,
        pyring_path=pyr_file,
    )


def main() -> None:
    args = parse_args()
    selected = [c.strip().lower() for c in args.cases.split(",") if c.strip()]
    selected = [c for c in selected if c in CASES]
    if not selected:
        raise ValueError(f"No valid case in --cases. Choices: {','.join(CASES)}")

    sbi_dir = (ROOT / args.sbi_dir).resolve() if not args.sbi_dir.is_absolute() else args.sbi_dir
    pyring_dir = (ROOT / args.pyring_dir).resolve() if not args.pyring_dir.is_absolute() else args.pyring_dir

    pairs: list[PosteriorPair] = []
    failures: dict[str, str] = {}
    for case in selected:
        try:
            pairs.append(_collect_pair(case, sbi_dir, pyring_dir))
        except Exception as exc:  # noqa: BLE001
            failures[case] = str(exc)

    if not pairs:
        raise RuntimeError(f"No case can be overlaid. failures={failures}")

    fig, axes = plt.subplots(1, len(pairs), figsize=(4.8 * len(pairs), 4.4), constrained_layout=True)
    if len(pairs) == 1:
        axes = [axes]

    summary: dict[str, Any] = {"cases": {}, "failures": failures}
    for ax, pair in zip(axes, pairs):
        _draw_contours(ax, pair.m_sbi, pair.chi_sbi, bins=args.bins, smooth_sigma=args.smooth_sigma, color="tab:blue", linewidths=(1.0, 1.8))
        _draw_contours(ax, pair.m_pyr, pair.chi_pyr, bins=args.bins, smooth_sigma=args.smooth_sigma, color="tab:orange", linewidths=(1.0, 1.8))
        if pair.m_true is not None:
            ax.axvline(pair.m_true, color="black", linestyle="--", linewidth=1.0)
        if pair.chi_true is not None:
            ax.axhline(pair.chi_true, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(pair.case)
        ax.set_xlabel("M_f [Msun]")
        ax.set_ylabel("chi_f")
        ax.plot([], [], color="tab:blue", linewidth=1.8, label="SBI")
        ax.plot([], [], color="tab:orange", linewidth=1.8, label="pyRing")
        ax.legend(loc="best", frameon=False)

        metrics = {
            "sbi_samples": int(pair.m_sbi.size),
            "pyring_samples": int(pair.m_pyr.size),
            "wasserstein_Mf": float(wasserstein_distance(pair.m_sbi, pair.m_pyr)),
            "wasserstein_chi": float(wasserstein_distance(pair.chi_sbi, pair.chi_pyr)),
            "ks_Mf_D": float(ks_2samp(pair.m_sbi, pair.m_pyr).statistic),
            "ks_chi_D": float(ks_2samp(pair.chi_sbi, pair.chi_pyr).statistic),
            "area_ratio_90_sbi_over_pyring": float(_area_ratio_90(pair, bins=args.bins, smooth_sigma=args.smooth_sigma)),
            "sbi_path": str(pair.sbi_path),
            "pyring_path": str(pair.pyring_path),
        }
        summary["cases"][pair.case] = metrics

    out_fig = (ROOT / args.output_figure).resolve() if not args.output_figure.is_absolute() else args.output_figure
    out_sum = (ROOT / args.summary_path).resolve() if not args.summary_path.is_absolute() else args.summary_path
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)
    out_sum.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved overlay figure: {out_fig}")
    print(f"Saved overlay summary: {out_sum}")
    if failures:
        print(f"Cases skipped: {sorted(failures.keys())}")


if __name__ == "__main__":
    main()
