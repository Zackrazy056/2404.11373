"""Ringdown injection generation from config dictionaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rd_sbi.detector.patterns import (
    antenna_pattern,
    detector_strain,
    gmst_from_gps,
    h1_geometry,
    l1_geometry,
    time_delay_from_geocenter_s,
)
from rd_sbi.qnm.kerr import kerr_qnm_physical
from rd_sbi.waveforms.ringdown import RingdownMode, build_time_array, generate_ringdown_polarizations


DETECTORS = {
    "H1": h1_geometry,
    "L1": l1_geometry,
}


def _required(cfg: dict[str, Any], path: str) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing required config key: {path}")
        cur = cur[key]
    return cur


def build_detector_timing_context(
    *,
    detectors: list[str],
    use_detector_time_delay: bool,
    reference_detector: str,
    t_start_s: float,
    ra_rad: float,
    dec_rad: float,
    gmst_rad: float,
) -> dict[str, Any]:
    """Build detector-dependent start times without cropping the earliest arrival.

    The reference detector still defines relative delays, but the common window is
    shifted so that the earliest detector starts at ``t_start_s`` rather than
    falling before the sampled segment.
    """
    if reference_detector not in DETECTORS:
        raise ValueError(f"Unsupported reference_detector={reference_detector}")

    if not use_detector_time_delay:
        zero_delays = {det: 0.0 for det in detectors}
        start_times = {det: float(t_start_s) for det in detectors}
        return {
            "time_delay_from_geocenter_s": zero_delays,
            "relative_delay_to_reference_s": zero_delays,
            "t_start_detector_s": start_times,
            "window_anchor_shift_s": 0.0,
            "earliest_detector": detectors[0] if detectors else reference_detector,
            "earliest_relative_delay_s": 0.0,
            "window_anchor_strategy": "disabled_no_detector_delay",
            "reference_detector": reference_detector,
        }

    delay_detectors = list(dict.fromkeys([*detectors, reference_detector]))
    geocenter_delays = {
        det: float(
            time_delay_from_geocenter_s(
                detector=DETECTORS[det](),
                ra_rad=ra_rad,
                dec_rad=dec_rad,
                gmst_rad=gmst_rad,
            )
        )
        for det in delay_detectors
    }
    delay_ref = geocenter_delays[reference_detector]
    relative_delays = {
        det: float(geocenter_delays[det] - delay_ref)
        for det in detectors
    }
    earliest_detector = min(relative_delays, key=relative_delays.get)
    earliest_relative_delay = float(relative_delays[earliest_detector])
    anchor_shift = max(0.0, -earliest_relative_delay)
    t_start_by_detector = {
        det: float(t_start_s + relative_delays[det] + anchor_shift)
        for det in detectors
    }
    return {
        "time_delay_from_geocenter_s": {det: geocenter_delays[det] for det in detectors},
        "relative_delay_to_reference_s": relative_delays,
        "t_start_detector_s": t_start_by_detector,
        "window_anchor_shift_s": float(anchor_shift),
        "earliest_detector": earliest_detector,
        "earliest_relative_delay_s": earliest_relative_delay,
        "window_anchor_strategy": "preserve_earliest_arrival_in_common_window",
        "reference_detector": reference_detector,
    }


def generate_injection(
    cfg: dict[str, Any],
    output_path: Path,
    *,
    override_seed: int | None = None,
    override_noise_std: float | None = None,
) -> tuple[Path, Path]:
    """Generate one injection file and metadata JSON from config."""
    name = str(_required(cfg, "name"))
    mass_msun = float(_required(cfg, "remnant.mass_msun"))
    chi_f = float(_required(cfg, "remnant.chi_f"))
    sample_rate_hz = float(_required(cfg, "data.sample_rate_hz"))
    duration_s = float(_required(cfg, "data.duration_s"))
    t_start_s = float(_required(cfg, "data.t_start_s"))

    ra_rad = float(_required(cfg, "source.ra_rad"))
    dec_rad = float(_required(cfg, "source.dec_rad"))
    psi_rad = float(_required(cfg, "source.psi_rad"))
    inclination_rad = float(_required(cfg, "source.inclination_rad"))
    gps_h1 = float(_required(cfg, "source.gps_h1"))
    leap_seconds = int(cfg.get("source", {}).get("leap_seconds", 18))

    qnm_cfg = dict(cfg.get("qnm", {}))
    method = str(qnm_cfg.get("method", "auto"))
    spin_weight = int(qnm_cfg.get("spin_weight", -2))
    global_alpha_r = float(qnm_cfg.get("alpha_r", 0.0))
    global_alpha_i = float(qnm_cfg.get("alpha_i", 0.0))

    detectors = [str(x) for x in cfg.get("detectors", ["H1", "L1"])]
    unknown = [d for d in detectors if d not in DETECTORS]
    if unknown:
        raise ValueError(f"Unsupported detectors: {unknown}; supported={sorted(DETECTORS)}")

    noise_std_cfg = float(cfg.get("noise_std", 0.0))
    noise_std = noise_std_cfg if override_noise_std is None else float(override_noise_std)
    seed_cfg = int(cfg.get("seed", 0))
    seed = seed_cfg if override_seed is None else int(override_seed)
    rng = np.random.default_rng(seed)

    use_detector_time_delay = bool(cfg.get("use_detector_time_delay", True))
    reference_detector = str(cfg.get("reference_detector", "H1"))
    if reference_detector not in DETECTORS:
        raise ValueError(f"Unsupported reference_detector={reference_detector}")

    time_s = build_time_array(sample_rate_hz=sample_rate_hz, duration_s=duration_s, start_time_s=0.0)
    if time_s.shape[0] != 204 and np.isclose(sample_rate_hz, 2048.0) and np.isclose(duration_s, 0.1):
        raise RuntimeError("Expected 204 bins for 2048 Hz x 0.1 s setup")

    ringdown_modes: list[RingdownMode] = []
    qnm_records: list[dict[str, Any]] = []
    for mode_cfg in cfg.get("modes", []):
        l = int(mode_cfg["l"])
        m = int(mode_cfg["m"])
        n = int(mode_cfg["n"])
        amp = float(mode_cfg["amplitude"])
        phase = float(mode_cfg["phase"])
        alpha_r = float(mode_cfg.get("alpha_r", global_alpha_r))
        alpha_i = float(mode_cfg.get("alpha_i", global_alpha_i))

        qnm = kerr_qnm_physical(
            l=l,
            m=m,
            n=n,
            mass_msun=mass_msun,
            chi_f=chi_f,
            alpha_r=alpha_r,
            alpha_i=alpha_i,
            method=method,
            spin_weight=spin_weight,
        )
        ringdown_modes.append(
            RingdownMode(
                l=l,
                m=m,
                n=n,
                amplitude=amp,
                phase=phase,
                frequency_hz=qnm.frequency_hz,
                damping_time_s=qnm.damping_time_s,
            )
        )
        qnm_records.append(
            {
                "l": l,
                "m": m,
                "n": n,
                "amplitude": amp,
                "phase": phase,
                "frequency_hz": qnm.frequency_hz,
                "damping_time_s": qnm.damping_time_s,
                "omega_dimensionless_real": float(np.real(qnm.omega_dimensionless)),
                "omega_dimensionless_imag": float(np.imag(qnm.omega_dimensionless)),
            }
        )

    if not ringdown_modes:
        raise ValueError("No modes found in config")

    gmst_rad = gmst_from_gps(gps_h1, leap_seconds=leap_seconds)
    timing = build_detector_timing_context(
        detectors=detectors,
        use_detector_time_delay=use_detector_time_delay,
        reference_detector=reference_detector,
        t_start_s=t_start_s,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        gmst_rad=gmst_rad,
    )

    detector_data: dict[str, np.ndarray] = {}
    detector_meta: dict[str, dict[str, float]] = {}
    for det in detectors:
        geometry = DETECTORS[det]()
        f_plus, f_cross = antenna_pattern(
            detector=geometry,
            ra_rad=ra_rad,
            dec_rad=dec_rad,
            psi_rad=psi_rad,
            gmst_rad=gmst_rad,
        )
        delay_det = float(timing["time_delay_from_geocenter_s"][det])
        t_start_det = float(timing["t_start_detector_s"][det])
        relative_delay = float(timing["relative_delay_to_reference_s"][det])

        pol = generate_ringdown_polarizations(
            time_s=time_s,
            modes=ringdown_modes,
            inclination_rad=inclination_rad,
            t_start_s=t_start_det,
        )
        strain = detector_strain(pol.h_plus, pol.h_cross, f_plus=f_plus, f_cross=f_cross)
        if noise_std > 0.0:
            strain = strain + rng.normal(loc=0.0, scale=noise_std, size=strain.shape[0])
        detector_data[det] = strain.astype(np.float64)
        detector_meta[det] = {
            "f_plus": float(f_plus),
            "f_cross": float(f_cross),
            "time_delay_from_geocenter_s": float(delay_det),
            "relative_delay_to_reference_s": relative_delay,
            "t_start_detector_s": float(t_start_det),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        time_s=time_s,
        sample_rate_hz=np.array(sample_rate_hz),
        duration_s=np.array(duration_s),
        **{f"strain_{det}": arr for det, arr in detector_data.items()},
    )

    metadata = {
        "name": name,
        "mass_msun": mass_msun,
        "chi_f": chi_f,
        "source": {
            "ra_rad": ra_rad,
            "dec_rad": dec_rad,
            "psi_rad": psi_rad,
            "inclination_rad": inclination_rad,
            "gps_h1": gps_h1,
            "leap_seconds": leap_seconds,
            "gmst_rad": gmst_rad,
        },
        "detectors": detector_meta,
        "modes": qnm_records,
        "noise_std": noise_std,
        "seed": seed,
        "use_detector_time_delay": use_detector_time_delay,
        "reference_detector": reference_detector,
        "window_anchor": {
            "strategy": str(timing["window_anchor_strategy"]),
            "shift_s": float(timing["window_anchor_shift_s"]),
            "earliest_detector": str(timing["earliest_detector"]),
            "earliest_relative_delay_s": float(timing["earliest_relative_delay_s"]),
        },
        "output_npz": str(output_path),
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path, meta_path
