"""Start reproducing paper Fig.1 with TSNPE posteriors for Kerr injections.

This script trains TSNPE models for Kerr220/Kerr221/Kerr330 from fresh
prior-sampled simulations, then plots M_f-chi_f posterior contours.

For pyRing overlay and quantitative comparison, run:
  scripts/10_overlay_fig1_pyring.py
after this script writes SBI posterior files.
"""

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
import torch
from sbi.utils import BoxUniform
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import load_yaml_config  # noqa: E402
from rd_sbi.detector.patterns import (  # noqa: E402
    antenna_pattern,
    detector_strain,
    gmst_from_gps,
    h1_geometry,
    l1_geometry,
    time_delay_from_geocenter_s,
)
from rd_sbi.inference.embedding_net import EmbeddingConfig, build_nsf_density_estimator  # noqa: E402
from rd_sbi.inference.tsnpe_runner import TSNPEConfig, TSNPERunner  # noqa: E402
from rd_sbi.qnm.kerr import kerr_qnm_physical  # noqa: E402
from rd_sbi.utils.seed import set_global_seed  # noqa: E402
from rd_sbi.waveforms.ringdown import RingdownMode, build_time_array, generate_ringdown_polarizations  # noqa: E402


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    cfg_path: Path
    mode_ids: list[tuple[int, int, int]]


CASE_SPECS = [
    CaseSpec("kerr220", "Kerr220", Path("configs/injections/kerr220.yaml"), [(2, 2, 0)]),
    CaseSpec("kerr221", "Kerr221", Path("configs/injections/kerr221.yaml"), [(2, 2, 0), (2, 2, 1)]),
    CaseSpec("kerr330", "Kerr330", Path("configs/injections/kerr330.yaml"), [(2, 2, 0), (3, 3, 0)]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start Fig.1 reproduction using TSNPE")
    parser.add_argument("--seed", type=int, default=240411373)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-sim-first", type=int, default=384)
    parser.add_argument("--num-sim-round", type=int, default=512)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--posterior-samples", type=int, default=1500)
    parser.add_argument(
        "--cases",
        type=str,
        default="kerr220,kerr221,kerr330",
        help="Comma-separated subset of cases: kerr220,kerr221,kerr330",
    )
    parser.add_argument("--output-figure", type=Path, default=Path("reports/figures/fig1_start_sbi.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/posteriors/fig1_start"))
    parser.add_argument("--show-progress-bars", action="store_true")
    parser.add_argument("--stop-after-epochs", type=int, default=6)
    parser.add_argument("--max-num-epochs", type=int, default=40)
    parser.add_argument("--hpd-samples", type=int, default=512)
    parser.add_argument("--volume-mc-samples", type=int, default=1024)
    return parser.parse_args()


def _mode_tag(mode_id: tuple[int, int, int]) -> str:
    l, m, n = mode_id
    return f"{l}{m}{n}"


def _build_param_names(mode_ids: list[tuple[int, int, int]]) -> list[str]:
    names = ["M", "chi"]
    for mode in mode_ids:
        tag = _mode_tag(mode)
        names.extend([f"A{tag}", f"phi{tag}"])
    return names


def _prior_bounds(mode_ids: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    low = [20.0, 0.0]
    high = [300.0, 0.99]
    for l, m, n in mode_ids:
        if (l, m, n) == (2, 2, 0):
            low.append(0.1e-21)
        else:
            low.append(0.0)
        high.append(50.0e-21)
        low.append(0.0)
        high.append(2.0 * np.pi)
    return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)


def _theta_from_config(cfg: dict[str, Any], mode_ids: list[tuple[int, int, int]]) -> np.ndarray:
    out = [float(cfg["remnant"]["mass_msun"]), float(cfg["remnant"]["chi_f"])]
    mode_cfg = {(int(m["l"]), int(m["m"]), int(m["n"])): m for m in cfg["modes"]}
    for mode_id in mode_ids:
        m = mode_cfg[mode_id]
        out.extend([float(m["amplitude"]), float(m["phase"])])
    return np.array(out, dtype=np.float32)


def _simulate_batch(cfg: dict[str, Any], mode_ids: list[tuple[int, int, int]], theta: torch.Tensor) -> torch.Tensor:
    ra_rad = float(cfg["source"]["ra_rad"])
    dec_rad = float(cfg["source"]["dec_rad"])
    psi_rad = float(cfg["source"]["psi_rad"])
    inclination_rad = float(cfg["source"]["inclination_rad"])
    gps_h1 = float(cfg["source"]["gps_h1"])
    leap_seconds = int(cfg["source"].get("leap_seconds", 18))
    sample_rate_hz = float(cfg["data"]["sample_rate_hz"])
    duration_s = float(cfg["data"]["duration_s"])
    t_start_s = float(cfg["data"]["t_start_s"])
    use_delay = bool(cfg.get("use_detector_time_delay", True))
    method = str(cfg.get("qnm", {}).get("method", "auto"))
    spin_weight = int(cfg.get("qnm", {}).get("spin_weight", -2))

    gmst_rad = gmst_from_gps(gps_h1, leap_seconds=leap_seconds)
    h1 = h1_geometry()
    l1 = l1_geometry()

    delay_h1 = time_delay_from_geocenter_s(h1, ra_rad=ra_rad, dec_rad=dec_rad, gmst_rad=gmst_rad)
    delay_l1 = time_delay_from_geocenter_s(l1, ra_rad=ra_rad, dec_rad=dec_rad, gmst_rad=gmst_rad)

    fplus_h1, fcross_h1 = antenna_pattern(h1, ra_rad=ra_rad, dec_rad=dec_rad, psi_rad=psi_rad, gmst_rad=gmst_rad)
    fplus_l1, fcross_l1 = antenna_pattern(l1, ra_rad=ra_rad, dec_rad=dec_rad, psi_rad=psi_rad, gmst_rad=gmst_rad)

    t = build_time_array(sample_rate_hz=sample_rate_hz, duration_s=duration_s, start_time_s=0.0)
    out: list[np.ndarray] = []
    theta_np = theta.detach().cpu().numpy()

    for row in theta_np:
        mass = float(row[0])
        chi = float(row[1])
        modes: list[RingdownMode] = []

        cursor = 2
        for mode_id in mode_ids:
            amp = float(row[cursor])
            phase = float(row[cursor + 1])
            cursor += 2
            qnm = kerr_qnm_physical(
                l=mode_id[0],
                m=mode_id[1],
                n=mode_id[2],
                mass_msun=mass,
                chi_f=chi,
                method=method,
                spin_weight=spin_weight,
            )
            modes.append(
                RingdownMode(
                    l=mode_id[0],
                    m=mode_id[1],
                    n=mode_id[2],
                    amplitude=amp,
                    phase=phase,
                    frequency_hz=qnm.frequency_hz,
                    damping_time_s=qnm.damping_time_s,
                )
            )

        t_h1 = t_start_s
        t_l1 = t_start_s
        if use_delay:
            t_l1 = t_start_s + (delay_l1 - delay_h1)

        pol_h1 = generate_ringdown_polarizations(t, modes=modes, inclination_rad=inclination_rad, t_start_s=t_h1)
        pol_l1 = generate_ringdown_polarizations(t, modes=modes, inclination_rad=inclination_rad, t_start_s=t_l1)

        strain_h1 = detector_strain(pol_h1.h_plus, pol_h1.h_cross, fplus_h1, fcross_h1)
        strain_l1 = detector_strain(pol_l1.h_plus, pol_l1.h_cross, fplus_l1, fcross_l1)
        x = np.concatenate([strain_h1, strain_l1], axis=0).astype(np.float32)
        out.append(x)

    return torch.from_numpy(np.stack(out, axis=0))


def _plot_case(ax: plt.Axes, m_samples: np.ndarray, chi_samples: np.ndarray, m_true: float, chi_true: float, title: str) -> None:
    finite = np.isfinite(m_samples) & np.isfinite(chi_samples)
    m = m_samples[finite]
    c = chi_samples[finite]
    if m.size < 10:
        ax.scatter(m, c, s=8, c="tab:blue", alpha=0.5, label="sbi")
    else:
        hist, x_edges, y_edges = np.histogram2d(m, c, bins=64, density=True)
        hist = gaussian_filter(hist, sigma=1.0)

        flat = hist.ravel()
        order = np.argsort(flat)[::-1]
        cdf = np.cumsum(flat[order]) / np.sum(flat)
        level90 = flat[order[np.searchsorted(cdf, 0.90)]]
        level68 = flat[order[np.searchsorted(cdf, 0.68)]]

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        xx, yy = np.meshgrid(x_centers, y_centers, indexing="ij")
        ax.contour(xx, yy, hist, levels=sorted([level90, level68]), colors=["tab:blue", "tab:blue"], linewidths=[1.0, 1.8])
        ax.scatter(m[:: max(1, m.size // 500)], c[:: max(1, c.size // 500)], s=2, c="tab:blue", alpha=0.15)

    ax.axvline(m_true, color="black", linestyle="--", linewidth=1.0)
    ax.axhline(chi_true, color="black", linestyle="--", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("M_f [Msun]")
    ax.set_ylabel("chi_f")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_keys = {x.strip().lower() for x in args.cases.split(",") if x.strip()}
    cases = [c for c in CASE_SPECS if c.key in selected_keys]
    if not cases:
        raise ValueError("No valid cases selected. Use subset of: kerr220,kerr221,kerr330")

    fig, axes = plt.subplots(1, len(cases), figsize=(4.6 * len(cases), 4.2), constrained_layout=True)
    if len(cases) == 1:
        axes = [axes]
    summary: dict[str, Any] = {"cases": {}}

    for ax, case in zip(axes, cases):
        cfg = load_yaml_config(ROOT / case.cfg_path)
        param_names = _build_param_names(case.mode_ids)
        low, high = _prior_bounds(case.mode_ids)
        prior = BoxUniform(low=torch.from_numpy(low), high=torch.from_numpy(high))
        true_theta = _theta_from_config(cfg, case.mode_ids)

        def simulator(theta: torch.Tensor) -> torch.Tensor:
            return _simulate_batch(cfg, case.mode_ids, theta)

        x_observed = simulator(torch.from_numpy(true_theta[None, :]))[0]

        emb_cfg = EmbeddingConfig(input_dim=408, num_hidden_layers=2, hidden_dim=64, output_dim=32)
        density_builder = build_nsf_density_estimator(
            embedding_config=emb_cfg,
            hidden_features=64,
            num_transforms=3,
            num_bins=8,
            num_blocks=2,
            z_score_theta="independent",
            z_score_x="independent",
            batch_norm=True,
        )

        runner = TSNPERunner(
            prior=prior,
            simulator=simulator,
            x_observed=x_observed,
            density_estimator_builder=density_builder,
            config=TSNPEConfig(
                num_simulations_first_round=args.num_sim_first,
                num_simulations_per_round=args.num_sim_round,
                max_rounds=args.max_rounds,
                trunc_quantile=1e-4,
                stopping_ratio=0.8,
                posterior_samples_for_hpd=args.hpd_samples,
                prior_volume_mc_samples=args.volume_mc_samples,
                rejection_candidate_batch=min(args.volume_mc_samples, 1024),
                rejection_max_batches=256,
                training_batch_size=128,
                learning_rate=1e-3,
                validation_fraction=0.1,
                stop_after_epochs=args.stop_after_epochs,
                max_num_epochs=args.max_num_epochs,
                show_train_summary=False,
                device=args.device,
                show_progress_bars=args.show_progress_bars,
            ),
        )
        posterior, diagnostics = runner.run()
        _ = posterior  # explicit: kept for future pyRing-comparable sampling.
        density_estimator = runner.last_density_estimator
        if density_estimator is None:
            raise RuntimeError("Missing density estimator after TSNPE run")

        samples = density_estimator.sample((args.posterior_samples,), condition=x_observed.reshape(1, -1))
        samples_np = samples.squeeze(1).detach().cpu().numpy()

        m_samples = samples_np[:, 0]
        chi_samples = samples_np[:, 1]
        _plot_case(ax, m_samples, chi_samples, m_true=true_theta[0], chi_true=true_theta[1], title=case.title)

        case_npz = out_dir / f"{case.key}_sbi_posterior.npz"
        np.savez_compressed(
            case_npz,
            samples=samples_np.astype(np.float32),
            x_observed=x_observed.detach().cpu().numpy().astype(np.float32),
            true_theta=true_theta.astype(np.float32),
            param_names=np.array(param_names, dtype=object),
        )
        summary["cases"][case.key] = {
            "config": str(case.cfg_path),
            "param_names": param_names,
            "n_samples": int(samples_np.shape[0]),
            "true_M": float(true_theta[0]),
            "true_chi": float(true_theta[1]),
            "m_mean": float(np.mean(m_samples)),
            "chi_mean": float(np.mean(chi_samples)),
            "diagnostics": [d.__dict__ for d in diagnostics],
            "posterior_file": str(case_npz),
        }

    fig_path = ROOT / args.output_figure
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    summary_path = out_dir / "fig1_start_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
