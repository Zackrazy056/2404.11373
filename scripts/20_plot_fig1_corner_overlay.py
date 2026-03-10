"""Plot Fig.1-style corner overlays between SBI and pyRing posteriors.

Two rendering styles are supported:
- paper: smooth KDE-based display figure, prefers raw pyRing posterior samples
- audit: histogram/smoothed-density figure for diagnostic comparison
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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
CASES = ("kerr220", "kerr221", "kerr330")
M_KEYS = ("M", "Mf", "Mf_msun", "M_f_msun", "mass", "mf")
CHI_KEYS = ("chi", "chi_f", "spin", "chi_final", "chif")


@dataclass(frozen=True)
class PosteriorPair:
    case: str
    title: str
    m_sbi: np.ndarray
    chi_sbi: np.ndarray
    m_pyr: np.ndarray
    chi_pyr: np.ndarray
    m_true: float
    chi_true: float
    sbi_path: Path
    pyring_path: Path
    pyring_source_kind: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig.1 corner-style SBI/pyRing overlays")
    parser.add_argument("--cases", type=str, default="kerr220,kerr221,kerr330")
    parser.add_argument("--sbi-case-dirs", type=str, default="")
    parser.add_argument("--sbi-root", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--pyring-dir", type=Path, default=Path("reports/posteriors/pyring"))
    parser.add_argument("--output-figure", type=Path, default=Path("reports/figures/fig1_corner_overlay.png"))
    parser.add_argument("--output-summary", type=Path, default=Path("reports/figures/fig1_corner_overlay_summary.json"))
    parser.add_argument("--style", type=str, choices=["paper", "audit"], default="paper")
    parser.add_argument("--bins", type=int, default=70)
    parser.add_argument("--smooth-sigma", type=float, default=1.2)
    parser.add_argument("--paper-grid-size", type=int, default=180)
    parser.add_argument("--paper-kde-bw", type=float, default=0.22)
    parser.add_argument("--audit-kde-bw", type=float, default=0.28)
    parser.add_argument("--max-kde-samples", type=int, default=8000)
    parser.add_argument("--prefer-raw-pyring", action="store_true", default=False)
    parser.add_argument("--disable-raw-pyring", action="store_true")
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _extract_by_keys(obj: Any, keys: tuple[str, ...]) -> np.ndarray | None:
    for key in keys:
        if key in obj:
            return np.asarray(obj[key], dtype=np.float64).reshape(-1)
    return None


def _load_sbi_npz(path: Path) -> tuple[np.ndarray, np.ndarray, float, float]:
    z = np.load(path, allow_pickle=True)
    samples = np.asarray(z["samples"], dtype=np.float64)
    theta_true = np.asarray(z["theta_true"], dtype=np.float64).reshape(-1)
    return samples[:, 0], samples[:, 1], float(theta_true[0]), float(theta_true[1])


def _load_pyring_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    m = _extract_by_keys(z, M_KEYS)
    chi = _extract_by_keys(z, CHI_KEYS)
    if m is None or chi is None:
        raise ValueError(f"{path} missing mass/spin keys")
    keep = np.isfinite(m) & np.isfinite(chi)
    return m[keep], chi[keep]


def _load_pyring_structured(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    posterior_dat = run_dir / "Nested_sampler" / "posterior.dat"
    if posterior_dat.exists():
        data = np.genfromtxt(posterior_dat, names=True, deletechars="")
        data = np.atleast_1d(data)
        if "Mf" not in (data.dtype.names or ()) or "af" not in (data.dtype.names or ()):
            raise ValueError(f"{posterior_dat} missing Mf/af columns")
        return np.asarray(data["Mf"], dtype=np.float64), np.asarray(data["af"], dtype=np.float64)
    raise FileNotFoundError(f"No raw posterior.dat found in {run_dir}")


def _load_manifest(pyring_dir: Path) -> dict[str, Any]:
    manifest_path = pyring_dir / "manifest_pyring.json"
    if not manifest_path.exists():
        return {"cases": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_pyring_source(case: str, pyring_dir: Path, *, prefer_raw: bool) -> tuple[np.ndarray, np.ndarray, Path, str]:
    manifest = _load_manifest(pyring_dir)
    case_meta = manifest.get("cases", {}).get(case, {})

    candidate_paths: list[tuple[Path, str]] = []
    if prefer_raw:
        raw_stable_npz = case_meta.get("raw_stable_npz")
        raw_run_npz = case_meta.get("raw_run_npz")
        run_dir = case_meta.get("run_dir")
        if raw_stable_npz:
            candidate_paths.append((Path(raw_stable_npz), "raw_npz"))
        if raw_run_npz:
            candidate_paths.append((Path(raw_run_npz), "raw_npz"))
        if run_dir:
            candidate_paths.append((Path(run_dir), "posterior_dat"))
        candidate_paths.append((pyring_dir / f"{case}_pyring_raw.npz", "raw_npz"))

    stable_npz = case_meta.get("stable_npz")
    run_npz = case_meta.get("run_npz")
    if stable_npz:
        candidate_paths.append((Path(stable_npz), "resampled_npz"))
    if run_npz:
        candidate_paths.append((Path(run_npz), "resampled_npz"))
    candidate_paths.append((pyring_dir / f"{case}_pyring.npz", "resampled_npz"))

    seen: set[str] = set()
    for candidate, kind in candidate_paths:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if kind == "posterior_dat":
            if candidate.exists():
                try:
                    m, chi = _load_pyring_structured(candidate)
                    return m, chi, candidate / "Nested_sampler" / "posterior.dat", kind
                except Exception:
                    continue
        else:
            if candidate.exists():
                try:
                    m, chi = _load_pyring_npz(candidate)
                    return m, chi, candidate, kind
                except Exception:
                    continue
    raise FileNotFoundError(f"No usable pyRing posterior found for {case} in {pyring_dir}")


def _latest_sbi_dir(root: Path, case: str) -> Path:
    candidates = sorted(root.rglob(f"{case}_sbi_posterior_20000.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No SBI posterior found for {case} under {root}")
    return candidates[0].parent


def _parse_case_dirs(spec: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for token in [x.strip() for x in spec.split(",") if x.strip()]:
        if "=" not in token:
            raise ValueError(f"Invalid --sbi-case-dirs token: {token}")
        case, path = token.split("=", 1)
        out[case.strip().lower()] = _resolve(Path(path.strip()))
    return out


def _collect_pair(case: str, sbi_dir: Path, pyring_dir: Path, *, prefer_raw_pyring: bool) -> PosteriorPair:
    sbi_path = sbi_dir / f"{case}_sbi_posterior_20000.npz"
    if not sbi_path.exists():
        sbi_path = sbi_dir / f"{case}_sbi_posterior.npz"
    if not sbi_path.exists():
        raise FileNotFoundError(f"Missing SBI posterior for {case}: {sbi_dir}")
    m_sbi, chi_sbi, m_true, chi_true = _load_sbi_npz(sbi_path)
    m_pyr, chi_pyr, pyring_path, source_kind = _resolve_pyring_source(case, pyring_dir, prefer_raw=prefer_raw_pyring)
    return PosteriorPair(
        case=case,
        title=case.replace("kerr", "Kerr"),
        m_sbi=m_sbi,
        chi_sbi=chi_sbi,
        m_pyr=m_pyr,
        chi_pyr=chi_pyr,
        m_true=m_true,
        chi_true=chi_true,
        sbi_path=sbi_path,
        pyring_path=pyring_path,
        pyring_source_kind=source_kind,
    )


def _metric_dict(pair: PosteriorPair) -> dict[str, Any]:
    return {
        "sbi_samples": int(pair.m_sbi.size),
        "pyring_samples": int(pair.m_pyr.size),
        "sbi_median": [float(np.median(pair.m_sbi)), float(np.median(pair.chi_sbi))],
        "pyring_median": [float(np.median(pair.m_pyr)), float(np.median(pair.chi_pyr))],
        "truth": [float(pair.m_true), float(pair.chi_true)],
        "sbi_path": str(pair.sbi_path),
        "pyring_path": str(pair.pyring_path),
        "pyring_source_kind": pair.pyring_source_kind,
    }


def _contour_threshold(hist: np.ndarray, credibility: float) -> float:
    flat = hist.ravel()
    order = np.argsort(flat)[::-1]
    cdf = np.cumsum(flat[order]) / np.sum(flat)
    idx = np.searchsorted(cdf, credibility)
    idx = min(idx, flat.size - 1)
    return float(flat[order[idx]])


def _hist2d(m: np.ndarray, chi: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray, smooth_sigma: float) -> np.ndarray:
    hist, _, _ = np.histogram2d(m, chi, bins=[x_edges, y_edges], density=True)
    return gaussian_filter(hist, sigma=smooth_sigma)


def _kde_1d(values: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    kde = gaussian_kde(values, bw_method=bw)
    return np.asarray(kde(grid), dtype=np.float64)


def _kde_2d(m: np.ndarray, chi: np.ndarray, xx: np.ndarray, yy: np.ndarray, bw: float) -> np.ndarray:
    kde = gaussian_kde(np.vstack([m, chi]), bw_method=bw)
    values = kde(np.vstack([xx.ravel(), yy.ravel()]))
    return values.reshape(xx.shape)


def _thin_for_kde(values_a: np.ndarray, values_b: np.ndarray, max_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if values_a.size <= max_samples or max_samples <= 0:
        return values_a, values_b
    idx = np.linspace(0, values_a.size - 1, max_samples).astype(int)
    return values_a[idx], values_b[idx]


def _plot_case_paper(
    fig: plt.Figure,
    spec: Any,
    pair: PosteriorPair,
    *,
    grid_size: int,
    kde_bw: float,
    max_kde_samples: int,
    show_legend: bool,
) -> dict[str, Any]:
    nested = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
        width_ratios=[4.0, 1.2],
        height_ratios=[1.2, 4.0],
        wspace=0.04,
        hspace=0.04,
    )
    ax_top = fig.add_subplot(nested[0, 0])
    ax_main = fig.add_subplot(nested[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(nested[1, 1], sharey=ax_main)

    m_all = np.concatenate([pair.m_sbi, pair.m_pyr, np.array([pair.m_true])])
    chi_all = np.concatenate([pair.chi_sbi, pair.chi_pyr, np.array([pair.chi_true])])
    m_pad = max((float(np.max(m_all)) - float(np.min(m_all))) * 0.08, 1e-6)
    chi_pad = max((float(np.max(chi_all)) - float(np.min(chi_all))) * 0.08, 1e-6)
    m_grid = np.linspace(float(np.min(m_all)) - m_pad, float(np.max(m_all)) + m_pad, grid_size)
    chi_grid = np.linspace(float(np.min(chi_all)) - chi_pad, float(np.max(chi_all)) + chi_pad, grid_size)
    xx, yy = np.meshgrid(m_grid, chi_grid, indexing="ij")

    m_pyr_kde, chi_pyr_kde = _thin_for_kde(pair.m_pyr, pair.chi_pyr, max_kde_samples)
    m_sbi_kde, chi_sbi_kde = _thin_for_kde(pair.m_sbi, pair.chi_sbi, max_kde_samples)
    z_pyr = _kde_2d(m_pyr_kde, chi_pyr_kde, xx, yy, kde_bw)
    z_sbi = _kde_2d(m_sbi_kde, chi_sbi_kde, xx, yy, kde_bw)
    lv90_pyr = _contour_threshold(z_pyr, 0.90)
    lv68_pyr = _contour_threshold(z_pyr, 0.68)
    lv90_sbi = _contour_threshold(z_sbi, 0.90)
    lv68_sbi = _contour_threshold(z_sbi, 0.68)

    ax_main.contour(xx, yy, z_pyr, levels=sorted([lv90_pyr, lv68_pyr]), colors=["#f18f01", "#f18f01"], linewidths=[1.6, 2.8])
    ax_main.contour(xx, yy, z_sbi, levels=sorted([lv90_sbi, lv68_sbi]), colors=["#5b4bdb", "#5b4bdb"], linewidths=[1.6, 2.8])
    ax_main.axvline(pair.m_true, color="black", linestyle="--", linewidth=1.3, alpha=0.75)
    ax_main.axhline(pair.chi_true, color="black", linestyle="--", linewidth=1.3, alpha=0.75)
    ax_main.plot(pair.m_true, pair.chi_true, marker="+", markersize=12, markeredgewidth=2.0, color="black")
    ax_main.set_xlabel(r"$M\ [M_\odot]$", fontsize=15)
    ax_main.set_ylabel(r"$\chi$", fontsize=15)
    ax_main.set_title(pair.title, fontsize=17, pad=7)
    ax_main.set_facecolor("#e6e6e6")
    ax_main.minorticks_on()

    dens_top_pyr = _kde_1d(m_pyr_kde, m_grid, kde_bw)
    dens_top_sbi = _kde_1d(m_sbi_kde, m_grid, kde_bw)
    ax_top.plot(m_grid, dens_top_pyr, color="#f18f01", linewidth=2.1)
    ax_top.plot(m_grid, dens_top_sbi, color="#5b4bdb", linewidth=2.1)
    ax_top.axvline(pair.m_true, color="black", linestyle="--", linewidth=1.3, alpha=0.75)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.set_facecolor("#e6e6e6")

    dens_right_pyr = _kde_1d(chi_pyr_kde, chi_grid, kde_bw)
    dens_right_sbi = _kde_1d(chi_sbi_kde, chi_grid, kde_bw)
    ax_right.plot(dens_right_pyr, chi_grid, color="#f18f01", linewidth=2.1)
    ax_right.plot(dens_right_sbi, chi_grid, color="#5b4bdb", linewidth=2.1)
    ax_right.axhline(pair.chi_true, color="black", linestyle="--", linewidth=1.3, alpha=0.75)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.set_facecolor("#e6e6e6")

    if show_legend:
        ax_main.plot([], [], color="#f18f01", linewidth=2.1, label="pyRing")
        ax_main.plot([], [], color="#5b4bdb", linewidth=2.1, label="sbi")
        ax_main.plot([], [], color="black", linestyle="--", linewidth=1.3, label=pair.title)
        ax_main.legend(frameon=False, loc="upper left", fontsize=12)

    return _metric_dict(pair)


def _plot_case_audit(
    fig: plt.Figure,
    spec: Any,
    pair: PosteriorPair,
    *,
    bins: int,
    smooth_sigma: float,
    kde_bw: float,
    show_legend: bool,
) -> dict[str, Any]:
    nested = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=spec,
        width_ratios=[4.0, 1.2],
        height_ratios=[1.2, 4.0],
        wspace=0.04,
        hspace=0.04,
    )
    ax_top = fig.add_subplot(nested[0, 0])
    ax_main = fig.add_subplot(nested[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(nested[1, 1], sharey=ax_main)

    m_all = np.concatenate([pair.m_sbi, pair.m_pyr, np.array([pair.m_true])])
    chi_all = np.concatenate([pair.chi_sbi, pair.chi_pyr, np.array([pair.chi_true])])
    m_pad = max((float(np.max(m_all)) - float(np.min(m_all))) * 0.06, 1e-6)
    chi_pad = max((float(np.max(chi_all)) - float(np.min(chi_all))) * 0.06, 1e-6)
    m_edges = np.linspace(float(np.min(m_all)) - m_pad, float(np.max(m_all)) + m_pad, bins + 1)
    chi_edges = np.linspace(float(np.min(chi_all)) - chi_pad, float(np.max(chi_all)) + chi_pad, bins + 1)
    hist_sbi = _hist2d(pair.m_sbi, pair.chi_sbi, m_edges, chi_edges, smooth_sigma)
    hist_pyr = _hist2d(pair.m_pyr, pair.chi_pyr, m_edges, chi_edges, smooth_sigma)
    lv90_sbi = _contour_threshold(hist_sbi, 0.90)
    lv68_sbi = _contour_threshold(hist_sbi, 0.68)
    lv90_pyr = _contour_threshold(hist_pyr, 0.90)
    lv68_pyr = _contour_threshold(hist_pyr, 0.68)
    x_centers = 0.5 * (m_edges[:-1] + m_edges[1:])
    y_centers = 0.5 * (chi_edges[:-1] + chi_edges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")

    ax_main.contour(xx, yy, hist_pyr, levels=sorted([lv90_pyr, lv68_pyr]), colors=["#f18f01", "#f18f01"], linewidths=[1.4, 2.4])
    ax_main.contour(xx, yy, hist_sbi, levels=sorted([lv90_sbi, lv68_sbi]), colors=["#5b4bdb", "#5b4bdb"], linewidths=[1.4, 2.4])
    ax_main.axvline(pair.m_true, color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax_main.axhline(pair.chi_true, color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax_main.plot(pair.m_true, pair.chi_true, marker="+", markersize=11, markeredgewidth=1.8, color="black")
    ax_main.set_xlabel(r"$M\ [M_\odot]$", fontsize=14)
    ax_main.set_ylabel(r"$\chi$", fontsize=14)
    ax_main.set_title(pair.title, fontsize=15, pad=6)
    ax_main.set_facecolor("#ededed")
    ax_main.minorticks_on()

    top_grid = np.linspace(m_edges[0], m_edges[-1], max(160, bins * 2))
    right_grid = np.linspace(chi_edges[0], chi_edges[-1], max(160, bins * 2))
    ax_top.plot(top_grid, _kde_1d(pair.m_pyr, top_grid, kde_bw), color="#f18f01", linewidth=1.9)
    ax_top.plot(top_grid, _kde_1d(pair.m_sbi, top_grid, kde_bw), color="#5b4bdb", linewidth=1.9)
    ax_top.axvline(pair.m_true, color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.tick_params(axis="y", left=False, labelleft=False)
    ax_top.set_facecolor("#ededed")

    ax_right.plot(_kde_1d(pair.chi_pyr, right_grid, kde_bw), right_grid, color="#f18f01", linewidth=1.9)
    ax_right.plot(_kde_1d(pair.chi_sbi, right_grid, kde_bw), right_grid, color="#5b4bdb", linewidth=1.9)
    ax_right.axhline(pair.chi_true, color="black", linestyle="--", linewidth=1.2, alpha=0.75)
    ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_right.tick_params(axis="y", left=False, labelleft=False)
    ax_right.set_facecolor("#ededed")

    if show_legend:
        ax_main.plot([], [], color="#f18f01", linewidth=1.9, label="pyRing")
        ax_main.plot([], [], color="#5b4bdb", linewidth=1.9, label="sbi")
        ax_main.plot([], [], color="black", linestyle="--", linewidth=1.2, label=pair.title)
        ax_main.legend(frameon=False, loc="upper left", fontsize=11)

    return _metric_dict(pair)


def main() -> None:
    args = parse_args()
    prefer_raw_pyring = (args.style == "paper" and not args.disable_raw_pyring) or args.prefer_raw_pyring

    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
        }
    )

    selected = [case.strip().lower() for case in args.cases.split(",") if case.strip()]
    selected = [case for case in selected if case in CASES]
    if not selected:
        raise ValueError(f"No valid case selected; choices={CASES}")

    sbi_root = _resolve(args.sbi_root)
    pyring_dir = _resolve(args.pyring_dir)
    explicit_dirs = _parse_case_dirs(args.sbi_case_dirs)

    pairs: list[PosteriorPair] = []
    for case in selected:
        sbi_dir = explicit_dirs[case] if case in explicit_dirs else _latest_sbi_dir(sbi_root, case)
        pairs.append(_collect_pair(case, sbi_dir, pyring_dir, prefer_raw_pyring=prefer_raw_pyring))

    fig = plt.figure(figsize=(4.6 * len(pairs), 5.2), constrained_layout=True)
    outer = GridSpec(1, len(pairs), figure=fig, wspace=0.18)

    summary: dict[str, Any] = {
        "style": args.style,
        "prefer_raw_pyring": bool(prefer_raw_pyring),
        "cases": {},
    }
    for idx, pair in enumerate(pairs):
        if args.style == "paper":
            summary["cases"][pair.case] = _plot_case_paper(
                fig,
                outer[idx],
                pair,
                grid_size=args.paper_grid_size,
                kde_bw=args.paper_kde_bw,
                max_kde_samples=args.max_kde_samples,
                show_legend=(idx == 0),
            )
        else:
            summary["cases"][pair.case] = _plot_case_audit(
                fig,
                outer[idx],
                pair,
                bins=args.bins,
                smooth_sigma=args.smooth_sigma,
                kde_bw=args.audit_kde_bw,
                show_legend=(idx == 0),
            )

    out_fig = _resolve(args.output_figure)
    out_json = _resolve(args.output_summary)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=320, facecolor="white")
    plt.close(fig)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved figure: {out_fig}")
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
