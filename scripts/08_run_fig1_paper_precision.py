"""Run paper-precision TSNPE for Fig.1 cases (SBI side).

Defaults follow paper-scale settings:
- round1 simulations: 50k
- later rounds: 100k
- trunc_quantile epsilon: 1e-4
- stopping_ratio: 0.8

This script saves per-case posterior samples and timing summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.linalg import solve_triangular
from scipy.signal import butter, sosfiltfilt
from sbi.utils import BoxUniform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import load_yaml_config, validate_paper_case_config  # noqa: E402
from rd_sbi.detector.patterns import (  # noqa: E402
    antenna_pattern,
    detector_strain,
    gmst_from_gps,
    h1_geometry,
    l1_geometry,
    time_delay_from_geocenter_s,
)
from rd_sbi.eval.snr import compute_network_snr, load_psd_npz, resample_psd_to_rfft_grid  # noqa: E402
from rd_sbi.inference.embedding_net import EmbeddingConfig, build_nsf_density_estimator  # noqa: E402
from rd_sbi.inference.normalization import UnitCubeBoxTransform  # noqa: E402
from rd_sbi.inference.tsnpe_runner import TSNPEConfig, TSNPERunner  # noqa: E402
from rd_sbi.noise.whitening import cholesky_lower_with_jitter, covariance_from_one_sided_psd  # noqa: E402
from rd_sbi.qnm.kerr import kerr_qnm_physical  # noqa: E402
from rd_sbi.simulator.injection import build_detector_timing_context  # noqa: E402
from rd_sbi.utils.seed import set_global_seed  # noqa: E402
from rd_sbi.waveforms.ringdown import RingdownMode, build_time_array, generate_ringdown_polarizations  # noqa: E402


@dataclass(frozen=True)
class CaseSpec:
    key: str
    cfg_path: Path
    mode_ids: list[tuple[int, int, int]]


CASE_SPECS = [
    CaseSpec("kerr220", Path("configs/injections/kerr220.yaml"), [(2, 2, 0)]),
    CaseSpec("kerr221", Path("configs/injections/kerr221.yaml"), [(2, 2, 0), (2, 2, 1)]),
    CaseSpec("kerr330", Path("configs/injections/kerr330.yaml"), [(2, 2, 0), (3, 3, 0)]),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-precision TSNPE for Fig.1")
    parser.add_argument("--case", type=str, choices=["kerr220", "kerr221", "kerr330"], required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--observation-timing-mode",
        type=str,
        choices=["shared_window_anchor", "detector_local_truncation"],
        default="shared_window_anchor",
        help="Only changes how per-detector observed segments are constructed before whitening.",
    )
    parser.add_argument("--override-sample-rate-hz", type=float, default=None)
    parser.add_argument("--override-duration-s", type=float, default=None)
    parser.add_argument("--bandpass-min-hz", type=float, default=None)
    parser.add_argument("--bandpass-max-hz", type=float, default=None)
    parser.add_argument("--seed", type=int, default=240411373)
    parser.add_argument("--num-sim-first", type=int, default=50_000)
    parser.add_argument("--num-sim-round", type=int, default=100_000)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--trunc-quantile", type=float, default=1e-4)
    parser.add_argument("--stopping-ratio", type=float, default=0.8)
    parser.add_argument("--posterior-samples", type=int, default=20_000)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--max-num-epochs", type=int, default=1_000)
    parser.add_argument("--truncation-device", type=str, choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--truncation-probe-samples", type=int, default=50_000)
    parser.add_argument("--min-rounds-before-stopping", type=int, default=3)
    parser.add_argument("--max-volume-for-stopping", type=float, default=0.95)
    parser.add_argument("--no-truncation-volume-threshold", type=float, default=0.999)
    parser.add_argument(
        "--allow-stop-without-shrink",
        action="store_true",
        help="Disable safeguard that requires V_t < V_(t-1) before stopping check.",
    )
    parser.add_argument(
        "--allow-no-truncation-next-round",
        action="store_true",
        help="Disable safeguard that aborts next round when previous volume is ~1.",
    )
    parser.add_argument("--show-progress-bars", action="store_true")
    parser.add_argument("--psd-dir", type=Path, default=Path("data/processed/psd/GW150914"))
    parser.add_argument(
        "--disable-varying-noise",
        action="store_true",
        help="Disable per-epoch unit-Gaussian noise augmentation in whitened space.",
    )
    parser.add_argument("--varying-noise-std", type=float, default=1.0)
    parser.add_argument("--varying-noise-apply-in-validation", action="store_true")
    parser.add_argument(
        "--disable-adaptive-truncation-relaxation",
        action="store_true",
        help="Disable automatic relaxation of TSNPE truncation when probe acceptance becomes too small.",
    )
    parser.add_argument("--min-probe-acceptance-rate-for-next-round", type=float, default=0.002)
    parser.add_argument(
        "--disable-auto-reduce-round-sims",
        action="store_true",
        help="Disable adaptive reduction of later-round simulation counts when rejection budget is infeasible.",
    )
    parser.add_argument("--rejection-acceptance-safety-factor", type=float, default=0.8)
    parser.add_argument("--min-round-sims-after-reduction", type=int, default=4_096)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=60.0)
    parser.add_argument("--status-path", type=Path, default=None)
    parser.add_argument("--heartbeat-log", type=Path, default=None)
    return parser.parse_args()


def _prior_bounds(mode_ids: list[tuple[int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    low = [20.0, 0.0]
    high = [300.0, 0.99]
    for l, m, n in mode_ids:
        low.append(0.1e-21 if (l, m, n) == (2, 2, 0) else 0.0)
        high.append(50.0e-21)
        low.append(0.0)
        high.append(2.0 * np.pi)
    return np.array(low, dtype=np.float32), np.array(high, dtype=np.float32)


def _theta_true(cfg: dict[str, Any], mode_ids: list[tuple[int, int, int]]) -> np.ndarray:
    out = [float(cfg["remnant"]["mass_msun"]), float(cfg["remnant"]["chi_f"])]
    mode_cfg = {(int(m["l"]), int(m["m"]), int(m["n"])): m for m in cfg["modes"]}
    for mode_id in mode_ids:
        m = mode_cfg[mode_id]
        out.extend([float(m["amplitude"]), float(m["phase"])])
    return np.array(out, dtype=np.float32)


def _load_detector_whiteners(psd_dir: Path, *, sample_rate_hz: float, n_samples: int) -> dict[str, dict[str, Any]]:
    manifest_path = (ROOT / psd_dir / "manifest_psd.json").resolve()
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    manifest_records = {
        str(item.get("detector")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict) and item.get("detector") is not None
    }
    whiteners: dict[str, dict[str, Any]] = {}
    for detector in ("H1", "L1"):
        psd_path = (ROOT / psd_dir / f"{detector}_psd_welch.npz").resolve()
        if not psd_path.exists():
            raise FileNotFoundError(f"Missing PSD file for {detector}: {psd_path}")
        frequency_hz_raw, psd_raw = load_psd_npz(str(psd_path))
        frequency_hz_target, psd_target = resample_psd_to_rfft_grid(
            frequency_hz=frequency_hz_raw,
            psd=psd_raw,
            sample_rate_hz=sample_rate_hz,
            n_samples=n_samples,
        )
        covariance = covariance_from_one_sided_psd(psd_target, sample_rate_hz=sample_rate_hz, n_samples=n_samples)
        cholesky_l = cholesky_lower_with_jitter(covariance)
        whiteners[detector] = {
            "psd_path": str(psd_path),
            "frequency_hz_raw": frequency_hz_raw,
            "psd_raw": psd_raw,
            "frequency_hz_target": frequency_hz_target,
            "psd_target": psd_target,
            "covariance": covariance,
            "cholesky_l": cholesky_l,
            "manifest_record": manifest_records.get(detector),
            "manifest_path": str(manifest_path) if manifest_path.exists() else None,
        }
    return whiteners


def _apply_whitener(strain: np.ndarray, cholesky_l: np.ndarray) -> np.ndarray:
    x = np.asarray(strain, dtype=np.float64)
    y = solve_triangular(cholesky_l, x, lower=True, check_finite=False)
    return np.asarray(y, dtype=np.float32)


def _maybe_bandpass_strain(
    strain: np.ndarray,
    *,
    sample_rate_hz: float,
    bandpass_min_hz: float | None,
    bandpass_max_hz: float | None,
    order: int = 4,
) -> np.ndarray:
    x = np.asarray(strain, dtype=np.float64)
    if bandpass_min_hz is None and bandpass_max_hz is None:
        return x
    if bandpass_min_hz is None or bandpass_max_hz is None:
        raise ValueError("bandpass_min_hz and bandpass_max_hz must be provided together")
    nyquist = 0.5 * float(sample_rate_hz)
    low = float(bandpass_min_hz) / nyquist
    high = float(bandpass_max_hz) / nyquist
    if not (0.0 < low < high < 1.0):
        raise ValueError(
            f"Invalid bandpass for sample_rate_hz={sample_rate_hz}: "
            f"min={bandpass_min_hz}, max={bandpass_max_hz}"
        )
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return np.asarray(sosfiltfilt(sos, x), dtype=np.float64)


def _clone_cfg_with_ablation_overrides(
    cfg: dict[str, Any],
    *,
    override_sample_rate_hz: float | None,
    override_duration_s: float | None,
) -> dict[str, Any]:
    cloned = json.loads(json.dumps(cfg))
    if override_sample_rate_hz is not None:
        cloned["data"]["sample_rate_hz"] = float(override_sample_rate_hz)
    if override_duration_s is not None:
        cloned["data"]["duration_s"] = float(override_duration_s)
    return cloned


def _build_simulator(
    cfg: dict[str, Any],
    mode_ids: list[tuple[int, int, int]],
    *,
    psd_dir: Path,
    theta_transform: UnitCubeBoxTransform,
    observation_timing_mode: str,
    bandpass_min_hz: float | None,
    bandpass_max_hz: float | None,
):
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
    fplus_h1, fcross_h1 = antenna_pattern(h1, ra_rad=ra_rad, dec_rad=dec_rad, psi_rad=psi_rad, gmst_rad=gmst_rad)
    fplus_l1, fcross_l1 = antenna_pattern(l1, ra_rad=ra_rad, dec_rad=dec_rad, psi_rad=psi_rad, gmst_rad=gmst_rad)
    t = build_time_array(sample_rate_hz=sample_rate_hz, duration_s=duration_s, start_time_s=0.0)
    whiteners = _load_detector_whiteners(psd_dir, sample_rate_hz=sample_rate_hz, n_samples=int(t.shape[0]))
    timing = build_detector_timing_context(
        detectors=["H1", "L1"],
        use_detector_time_delay=use_delay,
        reference_detector=str(cfg.get("reference_detector", "H1")),
        t_start_s=t_start_s,
        ra_rad=ra_rad,
        dec_rad=dec_rad,
        gmst_rad=gmst_rad,
    )
    if observation_timing_mode == "shared_window_anchor":
        observation_t_start = {
            det: float(start)
            for det, start in timing["t_start_detector_s"].items()
        }
        observation_timing = {
            "mode": observation_timing_mode,
            "detector_local_truncation_like_pyring": False,
            "t_start_detector_s": observation_t_start,
            "window_anchor_strategy": str(timing["window_anchor_strategy"]),
            "window_anchor_shift_s": float(timing["window_anchor_shift_s"]),
            "earliest_detector": str(timing["earliest_detector"]),
            "earliest_relative_delay_s": float(timing["earliest_relative_delay_s"]),
            "relative_delay_to_reference_s": {
                det: float(delay)
                for det, delay in timing["relative_delay_to_reference_s"].items()
            },
        }
    elif observation_timing_mode == "detector_local_truncation":
        observation_t_start = {"H1": float(t_start_s), "L1": float(t_start_s)}
        observation_timing = {
            "mode": observation_timing_mode,
            "detector_local_truncation_like_pyring": True,
            "t_start_detector_s": observation_t_start,
            "window_anchor_strategy": "detector_local_truncation_like_pyring",
            "window_anchor_shift_s": 0.0,
            "earliest_detector": None,
            "earliest_relative_delay_s": None,
            "relative_delay_to_reference_s": {
                det: float(delay)
                for det, delay in timing["relative_delay_to_reference_s"].items()
            },
        }
    else:
        raise ValueError(f"Unsupported observation_timing_mode={observation_timing_mode}")

    def project_strains(theta_physical_row: np.ndarray) -> dict[str, np.ndarray]:
        row = np.asarray(theta_physical_row, dtype=np.float64).reshape(-1)
        mass = float(row[0])
        chi = float(row[1])
        cursor = 2
        modes: list[RingdownMode] = []
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
        t_h1 = float(observation_t_start["H1"])
        t_l1 = float(observation_t_start["L1"])
        pol_h1 = generate_ringdown_polarizations(t, modes=modes, inclination_rad=inclination_rad, t_start_s=t_h1)
        pol_l1 = generate_ringdown_polarizations(t, modes=modes, inclination_rad=inclination_rad, t_start_s=t_l1)
        s1_raw = detector_strain(pol_h1.h_plus, pol_h1.h_cross, fplus_h1, fcross_h1)
        s2_raw = detector_strain(pol_l1.h_plus, pol_l1.h_cross, fplus_l1, fcross_l1)
        s1 = _maybe_bandpass_strain(
            s1_raw,
            sample_rate_hz=sample_rate_hz,
            bandpass_min_hz=bandpass_min_hz,
            bandpass_max_hz=bandpass_max_hz,
        )
        s2 = _maybe_bandpass_strain(
            s2_raw,
            sample_rate_hz=sample_rate_hz,
            bandpass_min_hz=bandpass_min_hz,
            bandpass_max_hz=bandpass_max_hz,
        )
        return {
            "H1": np.asarray(s1, dtype=np.float32),
            "L1": np.asarray(s2, dtype=np.float32),
            "H1_raw": np.asarray(s1_raw, dtype=np.float32),
            "L1_raw": np.asarray(s2_raw, dtype=np.float32),
            "H1_white": _apply_whitener(s1, whiteners["H1"]["cholesky_l"]),
            "L1_white": _apply_whitener(s2, whiteners["L1"]["cholesky_l"]),
        }

    def simulator(theta: torch.Tensor) -> torch.Tensor:
        arr_unit = theta.detach().cpu().numpy()
        out: list[np.ndarray] = []
        for row_unit in arr_unit:
            row_physical = theta_transform.inverse_numpy(row_unit)
            projected = project_strains(row_physical)
            out.append(np.concatenate([projected["H1_white"], projected["L1_white"]], axis=0).astype(np.float32))
        return torch.from_numpy(np.stack(out, axis=0))

    preprocessing = {
        "representation": "time_domain_dual_detector_concat",
        "input_dim": int(2 * t.shape[0]),
        "per_detector_bins": int(t.shape[0]),
        "sample_rate_hz": sample_rate_hz,
        "duration_s": duration_s,
        "whitening_method": "psd_to_acf_to_toeplitz_to_cholesky",
        "appendix_a_faithful": bool(
            abs(sample_rate_hz - 2048.0) < 1e-9
            and abs(duration_s - 0.1) < 1e-9
            and observation_timing_mode == "shared_window_anchor"
            and bandpass_min_hz is None
            and bandpass_max_hz is None
        ),
        "psd_sources": {det: meta["psd_path"] for det, meta in whiteners.items()},
        "psd_manifest_path": next((meta["manifest_path"] for meta in whiteners.values() if meta.get("manifest_path")), None),
        "psd_metadata": {
            det: meta.get("manifest_record")
            for det, meta in whiteners.items()
        },
        "x_standardization": {
            "applied": True,
            "method": "zscore_independent",
            "source": "sbi z_score_x=independent inside density estimator",
            "paper_faithful": True,
        },
        "theta_normalization": {
            "applied": True,
            "method": "external_minmax_unit_interval",
            "source": "physical BoxUniform -> [0,1]^d before TSNPE/SNPE training",
            "paper_expected": "minmax_0_1",
            "paper_faithful": True,
            "transform_low": theta_transform.low.astype(float).tolist(),
            "transform_high": theta_transform.high.astype(float).tolist(),
        },
        "window_anchor": {
            "strategy": str(observation_timing["window_anchor_strategy"]),
            "shift_s": float(observation_timing["window_anchor_shift_s"]),
            "reference_detector": str(timing["reference_detector"]),
            "earliest_detector": observation_timing["earliest_detector"],
            "earliest_relative_delay_s": observation_timing["earliest_relative_delay_s"],
            "t_start_detector_s": {
                det: float(start)
                for det, start in observation_timing["t_start_detector_s"].items()
            },
            "observation_timing_mode": observation_timing_mode,
            "detector_local_truncation_like_pyring": bool(
                observation_timing["detector_local_truncation_like_pyring"]
            ),
            "relative_delay_to_reference_s": {
                det: float(delay)
                for det, delay in observation_timing["relative_delay_to_reference_s"].items()
            },
        },
        "bandpass": {
            "enabled": bandpass_min_hz is not None and bandpass_max_hz is not None,
            "f_min_hz": bandpass_min_hz,
            "f_max_hz": bandpass_max_hz,
            "filter": "butterworth_zero_phase_order4" if bandpass_min_hz is not None and bandpass_max_hz is not None else None,
            "paper_faithful": bandpass_min_hz is None and bandpass_max_hz is None,
        },
    }
    return simulator, project_strains, preprocessing, whiteners


def _fixed_injection_context(cfg: dict[str, Any]) -> dict[str, Any]:
    src = cfg["source"]
    data = cfg["data"]
    timing = build_detector_timing_context(
        detectors=[str(x) for x in cfg.get("detectors", ["H1", "L1"])],
        use_detector_time_delay=bool(cfg.get("use_detector_time_delay", True)),
        reference_detector=str(cfg.get("reference_detector", "H1")),
        t_start_s=float(data["t_start_s"]),
        ra_rad=float(src["ra_rad"]),
        dec_rad=float(src["dec_rad"]),
        gmst_rad=gmst_from_gps(float(src["gps_h1"]), leap_seconds=int(src.get("leap_seconds", 18))),
    )
    return {
        "sample_rate_hz": float(data["sample_rate_hz"]),
        "duration_s": float(data["duration_s"]),
        "t_start_s": float(data["t_start_s"]),
        "ra_rad": float(src["ra_rad"]),
        "dec_rad": float(src["dec_rad"]),
        "psi_rad": float(src["psi_rad"]),
        "inclination_rad": float(src["inclination_rad"]),
        "gps_h1": float(src["gps_h1"]),
        "leap_seconds": int(src.get("leap_seconds", 18)),
        "use_detector_time_delay": bool(cfg.get("use_detector_time_delay", True)),
        "reference_detector": str(cfg.get("reference_detector", "H1")),
        "window_anchor": {
            "strategy": str(timing["window_anchor_strategy"]),
            "shift_s": float(timing["window_anchor_shift_s"]),
            "earliest_detector": str(timing["earliest_detector"]),
            "earliest_relative_delay_s": float(timing["earliest_relative_delay_s"]),
            "t_start_detector_s": {
                det: float(start)
                for det, start in timing["t_start_detector_s"].items()
            },
        },
    }


def _param_order(mode_ids: list[tuple[int, int, int]]) -> list[str]:
    order = ["M_f_msun", "chi_f"]
    for (l, m, n) in mode_ids:
        order.extend([f"A_{l}{m}{n}", f"phi_{l}{m}{n}"])
    return order


def _fixed_param_names() -> list[str]:
    return [
        "ra_rad",
        "dec_rad",
        "psi_rad",
        "inclination_rad",
        "gps_h1",
        "sample_rate_hz",
        "duration_s",
        "t_start_s",
        "use_detector_time_delay",
    ]


def _resolve_under(path: Path | None, out_dir: Path, default_name: str) -> Path:
    if path is None:
        return out_dir / default_name
    return path if path.is_absolute() else (ROOT / path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _append_heartbeat_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = str(payload.get("event", "heartbeat"))
    round_index = payload.get("round_index", "?")
    phase = payload.get("phase", "?")
    state = payload.get("state", "?")
    elapsed = payload.get("phase_elapsed_seconds")
    extra = []
    requested_num_simulations = payload.get("requested_num_simulations")
    if requested_num_simulations is not None and requested_num_simulations != payload.get("num_simulations"):
        extra.append(f"n_sim={payload['num_simulations']}/{requested_num_simulations}")
    if "truncated_prior_volume" in payload:
        extra.append(f"volume={payload['truncated_prior_volume']:.6f}")
    if "probe_acceptance_rate" in payload:
        extra.append(f"probe_accept={payload['probe_acceptance_rate']:.6f}")
    if payload.get("simulation_budget_adjusted"):
        extra.append("budget_adjusted=1")
    if payload.get("truncation_relaxed"):
        extra.append("trunc_relaxed=1")
    if "stop_reason" in payload:
        extra.append(f"reason={payload['stop_reason']}")
    elapsed_text = f"{float(elapsed):.1f}s" if elapsed is not None else "n/a"
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} event={event} state={state} round={round_index} phase={phase} elapsed={elapsed_text}"
    if extra:
        line += " " + " ".join(extra)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    case = next(c for c in CASE_SPECS if c.key == args.case)
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = _resolve_under(args.status_path, out_dir, f"{case.key}_status.json")
    heartbeat_log = _resolve_under(args.heartbeat_log, out_dir, f"{case.key}_heartbeat.log")

    def heartbeat_callback(payload: dict[str, Any]) -> None:
        _write_json_atomic(status_path, payload)
        _append_heartbeat_line(heartbeat_log, payload)

    cfg_base = load_yaml_config(ROOT / case.cfg_path)
    cfg_issues = validate_paper_case_config(cfg_base, case.key)
    if cfg_issues:
        raise ValueError(f"{case.key} config is not paper-case compliant: {cfg_issues}")
    cfg = _clone_cfg_with_ablation_overrides(
        cfg_base,
        override_sample_rate_hz=args.override_sample_rate_hz,
        override_duration_s=args.override_duration_s,
    )
    ablation_active = (
        args.override_sample_rate_hz is not None
        or args.override_duration_s is not None
        or args.bandpass_min_hz is not None
        or args.bandpass_max_hz is not None
        or args.observation_timing_mode != "shared_window_anchor"
    )
    cfg_effective_issues = validate_paper_case_config(cfg, case.key)
    low_phys, high_phys = _prior_bounds(case.mode_ids)
    theta_transform = UnitCubeBoxTransform(low_phys, high_phys)
    theta_true_physical = _theta_true(cfg, case.mode_ids)
    theta_true = theta_transform.forward_numpy(theta_true_physical).astype(np.float32)
    simulator, project_strains, preprocessing_meta, whiteners = _build_simulator(
        cfg,
        case.mode_ids,
        psd_dir=args.psd_dir,
        theta_transform=theta_transform,
        observation_timing_mode=args.observation_timing_mode,
        bandpass_min_hz=args.bandpass_min_hz,
        bandpass_max_hz=args.bandpass_max_hz,
    )

    low = np.zeros_like(low_phys, dtype=np.float32)
    high = np.ones_like(high_phys, dtype=np.float32)
    device_for_prior = "cuda:0" if str(args.device).startswith("cuda") else "cpu"
    prior = BoxUniform(
        low=torch.from_numpy(low).to(device_for_prior),
        high=torch.from_numpy(high).to(device_for_prior),
    )
    theta_true_projection = project_strains(theta_true_physical)
    x_observed = torch.from_numpy(
        np.concatenate([theta_true_projection["H1_white"], theta_true_projection["L1_white"]], axis=0).astype(np.float32)
    )
    target_network_snr = float(cfg.get("target_snr", np.nan))
    snr_result = compute_network_snr(
        strains={"H1": theta_true_projection["H1"], "L1": theta_true_projection["L1"]},
        sample_rate_hz=float(cfg["data"]["sample_rate_hz"]),
        psd_by_detector={
            det: (meta["frequency_hz_raw"], meta["psd_raw"])
            for det, meta in whiteners.items()
        },
    )
    snr_relative_error = (
        abs(float(snr_result.network_snr) - target_network_snr) / target_network_snr
        if np.isfinite(target_network_snr) and target_network_snr > 0.0
        else None
    )
    input_dim = int(preprocessing_meta["input_dim"])

    density_builder = build_nsf_density_estimator(
        embedding_config=EmbeddingConfig(input_dim=input_dim, num_hidden_layers=2, hidden_dim=150, output_dim=128),
        hidden_features=150,
        num_transforms=5,
        num_bins=10,
        num_blocks=2,
        batch_norm=True,
        z_score_theta=None,
        z_score_x="independent",
    )
    runner = TSNPERunner(
        prior=prior,
        simulator=simulator,
        x_observed=x_observed,
        density_estimator_builder=density_builder,
        heartbeat_callback=heartbeat_callback,
        heartbeat_interval_seconds=args.heartbeat_interval_seconds,
        config=TSNPEConfig(
            num_simulations_first_round=args.num_sim_first,
            num_simulations_per_round=args.num_sim_round,
            max_rounds=args.max_rounds,
            trunc_quantile=args.trunc_quantile,
            stopping_ratio=args.stopping_ratio,
            posterior_samples_for_hpd=8192,
            prior_volume_mc_samples=16384,
            rejection_candidate_batch=16384,
            rejection_max_batches=1024,
            training_batch_size=args.training_batch_size,
            learning_rate=args.learning_rate,
            validation_fraction=args.validation_fraction,
            stop_after_epochs=args.stop_after_epochs,
            max_num_epochs=args.max_num_epochs,
            show_train_summary=False,
            discard_round1_after_first_update=True,
            device=args.device,
            show_progress_bars=args.show_progress_bars,
            truncation_device=args.truncation_device,
            truncation_probe_samples=args.truncation_probe_samples,
            min_rounds_before_stopping=args.min_rounds_before_stopping,
            max_volume_for_stopping=args.max_volume_for_stopping,
            require_volume_shrink_for_stopping=(not args.allow_stop_without_shrink),
            fail_on_no_truncation_for_next_round=(not args.allow_no_truncation_next_round),
            no_truncation_volume_threshold=args.no_truncation_volume_threshold,
            varying_noise_enabled=(not args.disable_varying_noise),
            varying_noise_std=args.varying_noise_std,
            varying_noise_apply_in_validation=args.varying_noise_apply_in_validation,
            resume_training_state_across_rounds=False,
            adaptive_truncation_relaxation_enabled=(not args.disable_adaptive_truncation_relaxation),
            min_probe_acceptance_rate_for_next_round=args.min_probe_acceptance_rate_for_next_round,
            auto_reduce_round_simulations_on_low_acceptance=(not args.disable_auto_reduce_round_sims),
            rejection_acceptance_safety_factor=args.rejection_acceptance_safety_factor,
            min_simulations_per_round_after_reduction=args.min_round_sims_after_reduction,
        ),
    )

    t0 = time.time()
    print(f"Status file: {status_path}", flush=True)
    print(f"Heartbeat log: {heartbeat_log}", flush=True)
    try:
        posterior, diagnostics = runner.run()
        _ = posterior
        density_estimator = runner.last_density_estimator
        if density_estimator is None:
            raise RuntimeError("No density estimator after run")
        heartbeat_callback(
            {
                "state": "running",
                "event": "posterior_sampling_start",
                "round_index": len(diagnostics),
                "phase": "posterior_sampling",
                "num_simulations": 0,
                "phase_elapsed_seconds": 0.0,
            }
        )
        samples_unit = runner.sample_posterior(args.posterior_samples, cpu_safe=True)
        samples = torch.from_numpy(theta_transform.inverse_numpy(samples_unit.numpy()).astype(np.float32))
        heartbeat_callback(
            {
                "state": "running",
                "event": "posterior_sampling_end",
                "round_index": len(diagnostics),
                "phase": "posterior_sampling",
                "num_simulations": 0,
                "phase_elapsed_seconds": 0.0,
            }
        )
    except Exception as exc:  # noqa: BLE001
        failure_summary = {
            "case": case.key,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "status_path": str(status_path),
            "heartbeat_log": str(heartbeat_log),
            "latest_status": runner._snapshot_status(),
            "partial_diagnostics": [d.__dict__ for d in runner.last_diagnostics],
        }
        failure_path = out_dir / f"{case.key}_failure_summary.json"
        _write_json_atomic(failure_path, failure_summary)
        heartbeat_callback(
            {
                "state": "failed",
                "event": "error",
                "round_index": runner._snapshot_status().get("round_index", 0),
                "phase": runner._snapshot_status().get("phase", "unknown"),
                "num_simulations": runner._snapshot_status().get("num_simulations", 0),
                "phase_elapsed_seconds": runner._snapshot_status().get("phase_elapsed_seconds", 0.0),
                "error": str(exc),
                "failure_summary": str(failure_path),
            }
        )
        print(f"Saved failure summary: {failure_path}", flush=True)
        raise

    total_seconds = time.time() - t0
    npz_path = out_dir / f"{case.key}_sbi_posterior_20000.npz"
    np.savez_compressed(
        npz_path,
        samples=samples.detach().cpu().numpy().astype(np.float32),
        theta_true=theta_true_physical.astype(np.float32),
        theta_true_unit=theta_true.astype(np.float32),
        x_observed=x_observed.detach().cpu().numpy().astype(np.float32),
    )
    summary = {
        "case": case.key,
        "config": str(case.cfg_path),
        "paper_reference": "2404.11373v3",
        "prior": {
            "param_order": _param_order(case.mode_ids),
            "low": low_phys.astype(float).tolist(),
            "high": high_phys.astype(float).tolist(),
            "paper_table_ii_expected": {
                "M_f_msun": [20.0, 300.0],
                "chi_f": [0.0, 0.99],
            },
        },
        "parameterization": {
            "inferred_params": _param_order(case.mode_ids),
            "fixed_params": _fixed_param_names(),
            "dimension": len(_param_order(case.mode_ids)),
        },
        "preprocessing": preprocessing_meta,
        "waveform_model": {
            "kerr_mapping": {
                "backend": str(cfg.get("qnm", {}).get("method", "")),
                "paper_reference": "fits_[45]",
                "paper_faithful": str(cfg.get("qnm", {}).get("method", "")).strip().lower() == "fit",
            }
        },
        "training_contract": {
            "density_estimator": {
                "model": "nsf",
                "num_transforms": 5,
                "hidden_features": 150,
                "num_blocks": 2,
                "num_bins": 10,
                "batch_norm": True,
            },
            "embedding": {
                "input_dim": input_dim,
                "num_hidden_layers": 2,
                "hidden_dim": 150,
                "output_dim": 128,
            },
            "normalization": {
                "theta": "external_minmax_unit_interval",
                "x": "independent",
            },
            "varying_noise": {
                "enabled": (not args.disable_varying_noise),
                "noise_std": float(args.varying_noise_std),
                "apply_in_validation": bool(args.varying_noise_apply_in_validation),
                "implementation": "TSNPERunner -> NoiseResamplingSNPE in whitened space",
            },
        },
        "params": {
            "num_sim_first": args.num_sim_first,
            "num_sim_round": args.num_sim_round,
            "max_rounds": args.max_rounds,
            "trunc_quantile": args.trunc_quantile,
            "stopping_ratio": args.stopping_ratio,
            "posterior_samples": args.posterior_samples,
            "device": args.device,
            "truncation_device": args.truncation_device,
            "final_sampling_device": "cpu_safe_copy",
            "heartbeat_interval_seconds": args.heartbeat_interval_seconds,
            "observation_timing_mode": args.observation_timing_mode,
            "truncation_probe_samples": args.truncation_probe_samples,
            "min_rounds_before_stopping": args.min_rounds_before_stopping,
            "max_volume_for_stopping": args.max_volume_for_stopping,
            "require_volume_shrink_for_stopping": (not args.allow_stop_without_shrink),
            "fail_on_no_truncation_for_next_round": (not args.allow_no_truncation_next_round),
            "no_truncation_volume_threshold": args.no_truncation_volume_threshold,
            "resume_training_state_across_rounds": False,
            "adaptive_truncation_relaxation_enabled": (not args.disable_adaptive_truncation_relaxation),
            "min_probe_acceptance_rate_for_next_round": args.min_probe_acceptance_rate_for_next_round,
            "auto_reduce_round_simulations_on_low_acceptance": (not args.disable_auto_reduce_round_sims),
            "rejection_acceptance_safety_factor": args.rejection_acceptance_safety_factor,
            "min_simulations_per_round_after_reduction": args.min_round_sims_after_reduction,
        },
        "tsnpe_definition": {
            "truncation": "density-threshold HPD approximation",
            "criterion": "accept theta when log q_phi(theta|x_o) >= tau_epsilon",
            "tau_source": "posterior sample log_prob quantile at epsilon=trunc_quantile",
            "stopping_rule": "stop when V_t/V_(t-1) > stopping_ratio for t>1",
            "theta_space_for_training": "unit_cube",
            "sbi_loss_mode": "force_first_round_mle_on_explicit_truncated_prior_samples",
            "rejection_budget_fallback": {
                "enabled": (not args.disable_auto_reduce_round_sims),
                "mode": "adaptive_reduce_num_simulations_per_round",
                "safety_factor": float(args.rejection_acceptance_safety_factor),
                "min_simulations_after_reduction": int(args.min_round_sims_after_reduction),
            },
            "adaptive_truncation_relaxation": {
                "enabled": (not args.disable_adaptive_truncation_relaxation),
                "mode": "raise_truncated_prior_acceptance_floor",
                "min_probe_acceptance_rate_for_next_round": float(args.min_probe_acceptance_rate_for_next_round),
            },
        },
        "injection_context": _fixed_injection_context(cfg),
        "paper_case_validation": {
            "base_config_issues": cfg_issues,
            "base_config_passed": not cfg_issues,
            "effective_config_issues": cfg_effective_issues,
            "effective_config_passed": not cfg_effective_issues,
            "ablation_active": bool(ablation_active),
            "passed": (not cfg_issues) and (not ablation_active),
        },
        "snr": {
            "target_network_snr": target_network_snr,
            "measured_network_snr": float(snr_result.network_snr),
            "per_detector": {item.detector: float(item.snr) for item in snr_result.per_detector},
            "psd_source": {det: meta["psd_path"] for det, meta in whiteners.items()},
            "relative_error_to_target": snr_relative_error,
            "paper_faithful": bool(snr_relative_error is not None and snr_relative_error <= 0.05),
        },
        "coverage": {
            "appendix_b_run": False,
            "summary_path": None,
        },
        "total_seconds": total_seconds,
        "diagnostics": [d.__dict__ for d in diagnostics],
        "output_npz": str(npz_path),
        "status_path": str(status_path),
        "heartbeat_log": str(heartbeat_log),
    }
    summary_path = out_dir / f"{case.key}_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved posterior: {npz_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Total seconds: {total_seconds:.1f}")


if __name__ == "__main__":
    main()
