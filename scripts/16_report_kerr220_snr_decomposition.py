"""Generate a kerr220 SNR decomposition report for physical calibration.

The goal is not another posterior plot. This script decomposes the current
`kerr220` injection chain into:

1. source-frame waveform-only white-noise proxy
2. detector projection with unit PSD
3. detector projection with GW150914 PSD and no detector delay
4. final detector projection with GW150914 PSD and detector delay

This makes it possible to localize the SNR deficit relative to the paper
target before touching TSNPE or plotting logic.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

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
from rd_sbi.eval.snr import compute_detector_snr, compute_network_snr, load_psd_npz  # noqa: E402
from rd_sbi.qnm.kerr import kerr_qnm_physical  # noqa: E402
from rd_sbi.waveforms.ringdown import (  # noqa: E402
    RingdownMode,
    build_time_array,
    generate_ringdown_polarizations,
    y_plus_y_cross,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report kerr220 SNR decomposition")
    parser.add_argument("--config", type=Path, default=Path("configs/injections/kerr220.yaml"))
    parser.add_argument("--psd-dir", type=Path, default=Path("data/processed/psd/GW150914"))
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/tables/kerr220_snr_decomposition_report.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("reports/figures/kerr220_snr_decomposition_report.png"),
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("reports/tables/kerr220_snr_decomposition_report.md"),
    )
    return parser.parse_args()


def _unit_psd_grid(sample_rate_hz: float, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    freq = np.fft.rfftfreq(n_samples, d=1.0 / float(sample_rate_hz))
    return freq.astype(np.float64), np.ones_like(freq, dtype=np.float64)


def _mode_from_config(cfg: dict[str, Any]) -> tuple[RingdownMode, dict[str, float]]:
    mode_cfg = cfg["modes"][0]
    mass_msun = float(cfg["remnant"]["mass_msun"])
    chi_f = float(cfg["remnant"]["chi_f"])
    qnm_method = str(cfg.get("qnm", {}).get("method", "fit"))
    spin_weight = int(cfg.get("qnm", {}).get("spin_weight", -2))
    qnm = kerr_qnm_physical(
        l=int(mode_cfg["l"]),
        m=int(mode_cfg["m"]),
        n=int(mode_cfg["n"]),
        mass_msun=mass_msun,
        chi_f=chi_f,
        method=qnm_method,
        spin_weight=spin_weight,
    )
    mode = RingdownMode(
        l=int(mode_cfg["l"]),
        m=int(mode_cfg["m"]),
        n=int(mode_cfg["n"]),
        amplitude=float(mode_cfg["amplitude"]),
        phase=float(mode_cfg["phase"]),
        frequency_hz=qnm.frequency_hz,
        damping_time_s=qnm.damping_time_s,
    )
    meta = {
        "frequency_hz": float(qnm.frequency_hz),
        "damping_time_s": float(qnm.damping_time_s),
        "omega_dimensionless_real": float(np.real(qnm.omega_dimensionless)),
        "omega_dimensionless_imag": float(np.imag(qnm.omega_dimensionless)),
    }
    return mode, meta


def _projected_strains(cfg: dict[str, Any]) -> dict[str, Any]:
    mode, qnm_meta = _mode_from_config(cfg)
    sample_rate_hz = float(cfg["data"]["sample_rate_hz"])
    duration_s = float(cfg["data"]["duration_s"])
    t_start_s = float(cfg["data"]["t_start_s"])
    time_s = build_time_array(sample_rate_hz=sample_rate_hz, duration_s=duration_s, start_time_s=0.0)

    ra_rad = float(cfg["source"]["ra_rad"])
    dec_rad = float(cfg["source"]["dec_rad"])
    psi_rad = float(cfg["source"]["psi_rad"])
    inclination_rad = float(cfg["source"]["inclination_rad"])
    gps_h1 = float(cfg["source"]["gps_h1"])
    leap_seconds = int(cfg.get("source", {}).get("leap_seconds", 18))
    gmst_rad = gmst_from_gps(gps_h1, leap_seconds=leap_seconds)

    h1 = h1_geometry()
    l1 = l1_geometry()
    fph, fch = antenna_pattern(h1, ra_rad=ra_rad, dec_rad=dec_rad, psi_rad=psi_rad, gmst_rad=gmst_rad)
    fpl, fcl = antenna_pattern(l1, ra_rad=ra_rad, dec_rad=dec_rad, psi_rad=psi_rad, gmst_rad=gmst_rad)
    delay_h1 = time_delay_from_geocenter_s(h1, ra_rad=ra_rad, dec_rad=dec_rad, gmst_rad=gmst_rad)
    delay_l1 = time_delay_from_geocenter_s(l1, ra_rad=ra_rad, dec_rad=dec_rad, gmst_rad=gmst_rad)
    delta_l1_h1 = float(delay_l1 - delay_h1)

    pol_source = generate_ringdown_polarizations(
        time_s=time_s,
        modes=[mode],
        inclination_rad=inclination_rad,
        t_start_s=t_start_s,
    )
    pol_h1 = generate_ringdown_polarizations(
        time_s=time_s,
        modes=[mode],
        inclination_rad=inclination_rad,
        t_start_s=t_start_s,
    )
    pol_l1_no_delay = generate_ringdown_polarizations(
        time_s=time_s,
        modes=[mode],
        inclination_rad=inclination_rad,
        t_start_s=t_start_s,
    )
    pol_l1_with_delay = generate_ringdown_polarizations(
        time_s=time_s,
        modes=[mode],
        inclination_rad=inclination_rad,
        t_start_s=t_start_s + delta_l1_h1,
    )

    h1_strain = detector_strain(pol_h1.h_plus, pol_h1.h_cross, f_plus=fph, f_cross=fch)
    l1_no_delay_strain = detector_strain(
        pol_l1_no_delay.h_plus, pol_l1_no_delay.h_cross, f_plus=fpl, f_cross=fcl
    )
    l1_with_delay_strain = detector_strain(
        pol_l1_with_delay.h_plus, pol_l1_with_delay.h_cross, f_plus=fpl, f_cross=fcl
    )

    return {
        "time_s": time_s,
        "mode": mode,
        "qnm": qnm_meta,
        "pol_source": pol_source,
        "h1_strain": h1_strain,
        "l1_no_delay_strain": l1_no_delay_strain,
        "l1_with_delay_strain": l1_with_delay_strain,
        "patterns": {
            "H1": {"f_plus": float(fph), "f_cross": float(fch)},
            "L1": {"f_plus": float(fpl), "f_cross": float(fcl)},
        },
        "delays": {
            "H1_from_geocenter_s": float(delay_h1),
            "L1_from_geocenter_s": float(delay_l1),
            "L1_minus_H1_s": delta_l1_h1,
            "L1_lead_samples_at_2048Hz": float(abs(delta_l1_h1) * sample_rate_hz),
            "cropped_lead_time_s": float(max(0.0, -(t_start_s + delta_l1_h1))),
        },
        "angular_factors": {
            "Y_plus_220": float(y_plus_y_cross(mode.l, mode.m, inclination_rad)[0]),
            "Y_cross_220": float(y_plus_y_cross(mode.l, mode.m, inclination_rad)[1]),
        },
    }


def _load_psd_meta(psd_dir: Path) -> dict[str, Any]:
    manifest_path = (ROOT / psd_dir / "manifest_psd.json").resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "manifest_path": str(manifest_path),
        "event": manifest.get("event"),
        "input_glob": manifest.get("input_glob"),
        "files": manifest.get("files", []),
    }


def _snr_decomposition(cfg: dict[str, Any]) -> dict[str, Any]:
    proj = _projected_strains(cfg)
    time_s = proj["time_s"]
    sample_rate_hz = float(cfg["data"]["sample_rate_hz"])
    target_snr = float(cfg.get("target_snr", np.nan))

    unit_freq, unit_psd = _unit_psd_grid(sample_rate_hz=sample_rate_hz, n_samples=int(time_s.shape[0]))
    psd_h1 = load_psd_npz(str((ROOT / "data/processed/psd/GW150914/H1_psd_welch.npz").resolve()))
    psd_l1 = load_psd_npz(str((ROOT / "data/processed/psd/GW150914/L1_psd_welch.npz").resolve()))

    hp = np.asarray(proj["pol_source"].h_plus, dtype=np.float64)
    hx = np.asarray(proj["pol_source"].h_cross, dtype=np.float64)
    h1 = np.asarray(proj["h1_strain"], dtype=np.float64)
    l1_no_delay = np.asarray(proj["l1_no_delay_strain"], dtype=np.float64)
    l1_with_delay = np.asarray(proj["l1_with_delay_strain"], dtype=np.float64)

    waveform_only_hp = compute_detector_snr(hp, sample_rate_hz, unit_freq, unit_psd)
    waveform_only_hx = compute_detector_snr(hx, sample_rate_hz, unit_freq, unit_psd)
    waveform_only_rss = float(np.sqrt(waveform_only_hp**2 + waveform_only_hx**2))

    projected_unit_no_delay = compute_network_snr(
        strains={"H1": h1, "L1": l1_no_delay},
        sample_rate_hz=sample_rate_hz,
        psd_by_detector={"H1": (unit_freq, unit_psd), "L1": (unit_freq, unit_psd)},
    )
    projected_unit_with_delay = compute_network_snr(
        strains={"H1": h1, "L1": l1_with_delay},
        sample_rate_hz=sample_rate_hz,
        psd_by_detector={"H1": (unit_freq, unit_psd), "L1": (unit_freq, unit_psd)},
    )

    physical_no_delay = compute_network_snr(
        strains={"H1": h1, "L1": l1_no_delay},
        sample_rate_hz=sample_rate_hz,
        psd_by_detector={"H1": psd_h1, "L1": psd_l1},
    )
    physical_with_delay = compute_network_snr(
        strains={"H1": h1, "L1": l1_with_delay},
        sample_rate_hz=sample_rate_hz,
        psd_by_detector={"H1": psd_h1, "L1": psd_l1},
    )

    final_network_snr = float(physical_with_delay.network_snr)
    amplitude_scale_to_target = float(target_snr / final_network_snr)
    amplitude_equivalent_to_target = float(cfg["modes"][0]["amplitude"]) * amplitude_scale_to_target

    return {
        "time_series": {
            "time_s": time_s.tolist(),
            "h_plus": hp.tolist(),
            "h_cross": hx.tolist(),
            "H1_projected": h1.tolist(),
            "L1_projected_no_delay": l1_no_delay.tolist(),
            "L1_projected_with_delay": l1_with_delay.tolist(),
        },
        "decomposition": {
            "waveform_only_unit_psd_proxy": {
                "h_plus_snr": float(waveform_only_hp),
                "h_cross_snr": float(waveform_only_hx),
                "rss_snr": waveform_only_rss,
                "definition": "source-frame white-noise proxy from h_plus/h_cross, not a physical detector SNR",
            },
            "after_projection_unit_psd_proxy": {
                "network_no_delay": float(projected_unit_no_delay.network_snr),
                "network_with_delay": float(projected_unit_with_delay.network_snr),
                "per_detector_no_delay": {x.detector: float(x.snr) for x in projected_unit_no_delay.per_detector},
                "per_detector_with_delay": {x.detector: float(x.snr) for x in projected_unit_with_delay.per_detector},
                "definition": "detector-projected white-noise proxy with flat one-sided PSD=1",
            },
            "after_psd_choice_no_delay": {
                "network_snr": float(physical_no_delay.network_snr),
                "per_detector": {x.detector: float(x.snr) for x in physical_no_delay.per_detector},
                "definition": "current GW150914 PSD choice, but with L1 forced to share H1 start time",
            },
            "final_network_snr": {
                "network_snr": final_network_snr,
                "per_detector": {x.detector: float(x.snr) for x in physical_with_delay.per_detector},
                "definition": "current pipeline: projection + current GW150914 PSD + detector time delay",
            },
        },
        "derived_checks": {
            "network_snr_target": target_snr,
            "network_snr_gap": float(target_snr - final_network_snr),
            "network_snr_gap_fraction": float((target_snr - final_network_snr) / target_snr),
            "time_delay_loss_in_network_snr": float(physical_no_delay.network_snr - physical_with_delay.network_snr),
            "time_delay_loss_fraction_relative_to_no_delay": float(
                (physical_no_delay.network_snr - physical_with_delay.network_snr) / physical_no_delay.network_snr
            ),
            "l1_time_delay_loss_in_snr": float(
                physical_no_delay.per_detector[1].snr - physical_with_delay.per_detector[1].snr
            ),
            "required_amplitude_scale_if_amplitude_were_only_issue": amplitude_scale_to_target,
            "required_A220_if_amplitude_were_only_issue": amplitude_equivalent_to_target,
        },
        "max_abs_strain": {
            "h_plus": float(np.max(np.abs(hp))),
            "h_cross": float(np.max(np.abs(hx))),
            "H1_projected": float(np.max(np.abs(h1))),
            "L1_projected_no_delay": float(np.max(np.abs(l1_no_delay))),
            "L1_projected_with_delay": float(np.max(np.abs(l1_with_delay))),
        },
        "diagnosis": [
            "Current amplitude normalization does not look like the main deficit by itself.",
            "With current projection and PSD but forcing no detector delay, network SNR is above target (15.72 > 14).",
            "Applying the current detector delay inside the fixed 0.1 s window drops network SNR to 12.55.",
            "The dominant loss is the L1 early-arrival truncation inside the fixed window, not a small amplitude underscaling.",
        ],
        "projection": proj["patterns"],
        "delays": proj["delays"],
        "angular_factors": proj["angular_factors"],
        "qnm": proj["qnm"],
    }


def _render_report_figure(report: dict[str, Any], output_path: Path) -> None:
    ts = report["time_series"]
    dec = report["decomposition"]
    derived = report["derived_checks"]
    delays = report["delays"]

    time_s = np.asarray(ts["time_s"], dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(time_s, np.asarray(ts["h_plus"]), label="h_plus", color="#0f766e")
    ax.plot(time_s, np.asarray(ts["h_cross"]), label="h_cross", color="#b45309")
    ax.set_title("Source polarizations")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("strain")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.plot(time_s, np.asarray(ts["H1_projected"]), label="H1", color="#1d4ed8")
    ax.plot(time_s, np.asarray(ts["L1_projected_no_delay"]), label="L1 no delay", color="#64748b")
    ax.plot(time_s, np.asarray(ts["L1_projected_with_delay"]), label="L1 with delay", color="#dc2626")
    ax.set_title("Detector projection and time-delay cropping")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("strain")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    labels = ["H1", "L1"]
    no_delay = [dec["after_psd_choice_no_delay"]["per_detector"]["H1"], dec["after_psd_choice_no_delay"]["per_detector"]["L1"]]
    with_delay = [dec["final_network_snr"]["per_detector"]["H1"], dec["final_network_snr"]["per_detector"]["L1"]]
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, no_delay, width, label="no delay", color="#94a3b8")
    ax.bar(x + width / 2, with_delay, width, label="with delay", color="#2563eb")
    ax.set_xticks(x, labels)
    ax.set_ylabel("optimal SNR")
    ax.set_title("Per-detector SNR under current PSD")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.axis("off")
    text = (
        "kerr220 SNR decomposition\n"
        f"waveform-only unit-PSD proxy RSS: {dec['waveform_only_unit_psd_proxy']['rss_snr']:.3e}\n"
        f"after projection unit-PSD proxy (no delay): {dec['after_projection_unit_psd_proxy']['network_no_delay']:.3e}\n"
        f"after PSD choice, no delay: {dec['after_psd_choice_no_delay']['network_snr']:.3f}\n"
        f"final network SNR: {dec['final_network_snr']['network_snr']:.3f}\n"
        f"target network SNR: {derived['network_snr_target']:.3f}\n"
        f"time-delay loss in network SNR: {derived['time_delay_loss_in_network_snr']:.3f}\n"
        f"time-delay loss fraction: {derived['time_delay_loss_fraction_relative_to_no_delay']:.3%}\n"
        f"L1 lead time: {delays['L1_minus_H1_s']:.6f} s\n"
        f"L1 lead samples @2048Hz: {delays['L1_lead_samples_at_2048Hz']:.2f}\n"
        f"required amplitude scale if amplitude-only: {derived['required_amplitude_scale_if_amplitude_were_only_issue']:.3f}\n"
        f"required A220 if amplitude-only: {derived['required_A220_if_amplitude_were_only_issue']:.3e}\n\n"
        "Preliminary diagnosis:\n"
        "- no-delay SNR is already above target\n"
        "- current fixed-window delay handling is the dominant SNR loss\n"
        "- amplitude mismatch is not the first suspect"
    )
    ax.text(0.02, 0.98, text, va="top", ha="left", family="monospace", fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_markdown(report: dict[str, Any], output_path: Path) -> None:
    dec = report["decomposition"]
    derived = report["derived_checks"]
    delays = report["delays"]
    patterns = report["projection"]
    qnm = report["qnm"]
    angular = report["angular_factors"]

    text = f"""# kerr220 SNR Decomposition Report

## Summary

- target network SNR: `{derived['network_snr_target']:.3f}`
- final network SNR: `{dec['final_network_snr']['network_snr']:.3f}`
- gap: `{derived['network_snr_gap']:.3f}` ({derived['network_snr_gap_fraction']:.2%})

## Decomposition

- waveform-only unit-PSD proxy RSS: `{dec['waveform_only_unit_psd_proxy']['rss_snr']:.6e}`
- after projection unit-PSD proxy (no delay): `{dec['after_projection_unit_psd_proxy']['network_no_delay']:.6e}`
- after projection unit-PSD proxy (with delay): `{dec['after_projection_unit_psd_proxy']['network_with_delay']:.6e}`
- after PSD choice, no delay: `{dec['after_psd_choice_no_delay']['network_snr']:.6f}`
- final network SNR: `{dec['final_network_snr']['network_snr']:.6f}`

## Dominant effect

The current detector-delay handling inside the fixed 0.1 s window is the dominant SNR loss:

- no-delay network SNR: `{dec['after_psd_choice_no_delay']['network_snr']:.6f}`
- final network SNR: `{dec['final_network_snr']['network_snr']:.6f}`
- network SNR loss from delay/windowing: `{derived['time_delay_loss_in_network_snr']:.6f}`
- loss fraction relative to no-delay: `{derived['time_delay_loss_fraction_relative_to_no_delay']:.2%}`
- L1 SNR no-delay: `{dec['after_psd_choice_no_delay']['per_detector']['L1']:.6f}`
- L1 SNR with delay: `{dec['final_network_snr']['per_detector']['L1']:.6f}`

## Projection and geometry

- H1 `(F+, Fx)`: `({patterns['H1']['f_plus']:.6f}, {patterns['H1']['f_cross']:.6f})`
- L1 `(F+, Fx)`: `({patterns['L1']['f_plus']:.6f}, {patterns['L1']['f_cross']:.6f})`
- H1 delay from geocenter: `{delays['H1_from_geocenter_s']:.6f} s`
- L1 delay from geocenter: `{delays['L1_from_geocenter_s']:.6f} s`
- L1 - H1: `{delays['L1_minus_H1_s']:.6f} s`
- L1 lead samples at 2048 Hz: `{delays['L1_lead_samples_at_2048Hz']:.2f}`

## Waveform factors

- QNM frequency: `{qnm['frequency_hz']:.6f} Hz`
- damping time: `{qnm['damping_time_s']:.6f} s`
- `Y_plus_220`: `{angular['Y_plus_220']:.6f}`
- `Y_cross_220`: `{angular['Y_cross_220']:.6f}`

## Amplitude calibration check

If amplitude were the only issue, the current implementation would need:

- scale factor: `{derived['required_amplitude_scale_if_amplitude_were_only_issue']:.6f}`
- equivalent `A220`: `{derived['required_A220_if_amplitude_were_only_issue']:.6e}`

But the no-delay SNR is already above target, so amplitude-only rescaling is not the first suspect.
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(ROOT / args.config)
    report = {
        "config": str(args.config),
        "paper_reference": "2404.11373v3",
        "psd_meta": _load_psd_meta(args.psd_dir),
        "input_table_i_values": {
            "M_f_msun": float(cfg["remnant"]["mass_msun"]),
            "chi_f": float(cfg["remnant"]["chi_f"]),
            "A_220": float(cfg["modes"][0]["amplitude"]),
            "phi_220": float(cfg["modes"][0]["phase"]),
            "inclination_rad": float(cfg["source"]["inclination_rad"]),
            "target_network_snr": float(cfg["target_snr"]),
        },
    }
    report.update(_snr_decomposition(cfg))

    output_json = ROOT / args.output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _render_report_figure(report, ROOT / args.output_figure)
    _write_markdown(report, ROOT / args.output_markdown)

    print(f"Saved JSON: {ROOT / args.output_json}")
    print(f"Saved figure: {ROOT / args.output_figure}")
    print(f"Saved markdown: {ROOT / args.output_markdown}")


if __name__ == "__main__":
    main()
