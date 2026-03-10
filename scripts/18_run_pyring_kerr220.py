"""Prepare, launch, and export a paper-like pyRing Kerr220 run from Windows via WSL.

This wrapper keeps the physical choices close to the public pyRing/GW150914 path:
4096 Hz, 0.2 s truncation, 100-500 Hz bandpass, detector-local pyRing timing,
and a zero-noise Kerr injection embedded in local GW150914 noise data.
"""

from __future__ import annotations

import argparse
import configparser
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "injections" / "kerr220.yaml"
DEFAULT_GW150914_DIR = ROOT / "external" / "pyRingGW-2.3.0" / "pyRing" / "data" / "Real_data" / "GW150914"
DEFAULT_PYRING_ROOT = ROOT / "reports" / "posteriors" / "pyring"
DEFAULT_REFERENCE_AMPLITUDE = 1.0e-21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper-like pyRing Kerr220 job from Windows via WSL")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--wsl-venv", type=str, default="~/.venvs/pyring312")
    parser.add_argument("--nlive", type=int, default=2048)
    parser.add_argument("--maxmcmc", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sampling-rate", type=float, default=4096.0)
    parser.add_argument("--analysis-duration", type=float, default=0.2)
    parser.add_argument("--signal-chunksize", type=float, default=4.0)
    parser.add_argument("--noise-chunksize", type=float, default=4.0)
    parser.add_argument("--f-min-bp", type=float, default=100.0)
    parser.add_argument("--f-max-bp", type=float, default=500.0)
    parser.add_argument("--disable-bandpass", action="store_true")
    parser.add_argument("--fix-t", type=float, default=0.0)
    parser.add_argument("--reference-amplitude", type=float, default=DEFAULT_REFERENCE_AMPLITUDE)
    parser.add_argument("--a2220-max", type=float, default=7.0)
    parser.add_argument("--target-samples", type=int, default=20000)
    parser.add_argument("--export-seed", type=int, default=240411373)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for a fast foreground smoke run (nlive=maxmcmc=32)")
    return parser.parse_args()


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    posix = resolved.as_posix()
    if len(posix) >= 3 and posix[1:3] == ":/":
        return f"/mnt/{posix[0].lower()}{posix[2:]}"
    return posix


def _load_kerr220_payload(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modes = payload.get("modes", [])
    if len(modes) != 1:
        raise ValueError(f"{config_path} must contain exactly one mode for Kerr220")
    mode = modes[0]
    if (int(mode["l"]), int(mode["m"]), int(mode["n"])) != (2, 2, 0):
        raise ValueError(f"{config_path} is not a pure (2,2,0) configuration")
    return payload


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    if args.run_tag:
        tag = args.run_tag
    else:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.smoke:
            tag = f"{tag}-smoke"
    return (DEFAULT_PYRING_ROOT / f"kerr220_{tag}").resolve()


def _build_config_text(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_dir: Path,
) -> str:
    data = payload["data"]
    source = payload["source"]
    remnant = payload["remnant"]
    mode = payload["modes"][0]

    direct_amp = float(mode["amplitude"])
    if args.reference_amplitude <= 0.0:
        raise ValueError("--reference-amplitude must be positive")
    a2220_injection = direct_amp / args.reference_amplitude

    h1_path = _to_wsl_path(DEFAULT_GW150914_DIR / "H-H1_GWOSC_4KHZ_R1-1126259447-32.txt")
    l1_path = _to_wsl_path(DEFAULT_GW150914_DIR / "L-L1_GWOSC_4KHZ_R1-1126259447-32.txt")
    imr_samples_path = _to_wsl_path(DEFAULT_GW150914_DIR / "GW150914_LAL_IMRPhenomP_O1_GWOSC_Mf_af_samples.txt")
    output_path = _to_wsl_path(run_dir)

    config = configparser.ConfigParser()
    config.optionxform = str
    config["input"] = {
        "run-type": "full",
        "pesummary": "0",
        "screen-output": "1",
        "output": output_path,
        "run-tag": run_dir.name,
        "data-H1": h1_path,
        "data-L1": l1_path,
        "trigtime": f"{float(source['gps_h1']):.8f}",
        "detectors": "H1,L1",
        "ref-det": str(payload.get("reference_detector", "H1")),
        "sky-frame": "equatorial",
        "template": "Kerr",
        "injection-approximant": "Kerr",
        "kerr-modes": "[(2,2,2,0)]",
        "reference-amplitude": f"{args.reference_amplitude:.8e}",
        "amp-non-prec-sym": "1",
        "sampling-rate": f"{args.sampling_rate:.1f}",
        "signal-chunksize": f"{args.signal_chunksize:.1f}",
        "noise-chunksize": f"{args.noise_chunksize:.1f}",
        "f-min-bp": f"{args.f_min_bp:.1f}",
        "f-max-bp": f"{args.f_max_bp:.1f}",
        "bandpassing": "0" if args.disable_bandpass else "1",
        "fft-acf": "1",
        "acf-simple-norm": "1",
        "truncate": "1",
        "analysis-duration": f"{args.analysis_duration:.3f}",
        "zero-noise": "1",
    }
    config["Sampler settings"] = {
        "nlive": str(args.nlive),
        "maxmcmc": str(args.maxmcmc),
        "seed": str(args.seed),
        "nthreads": "1",
        "verbose": "2",
    }
    config["Priors"] = {
        "mf-time-prior": f"{float(remnant['mass_msun']):.1f}",
        "fix-t": f"{args.fix_t:.6f}",
        "fix-ra": f"{float(source['ra_rad']):.8f}",
        "fix-dec": f"{float(source['dec_rad']):.8f}",
        "fix-psi": f"{float(source['psi_rad']):.8f}",
        "fix-phi": "0.0",
        "fix-cosiota": f"{np.cos(float(source['inclination_rad'])):.8f}",
        "Mf-min": "50.0",
        "Mf-max": "100.0",
        "af-min": "0.2",
        "af-max": "0.96",
        "A2220-min": "0.0",
        "A2220-max": f"{args.a2220_max:.3f}",
    }
    config["Injection"] = {
        "t0": f"{args.fix_t:.6f}",
        "ra": f"{float(source['ra_rad']):.8f}",
        "dec": f"{float(source['dec_rad']):.8f}",
        "psi": f"{float(source['psi_rad']):.8f}",
        "Mf": f"{float(remnant['mass_msun']):.8f}",
        "af": f"{float(remnant['chi_f']):.8f}",
        "logdistance": "6.0857",
        "cosiota": f"{np.cos(float(source['inclination_rad'])):.8f}",
        "phi": "0.0",
        "kerr-amplitudes": json.dumps({"2220": a2220_injection}),
        "kerr-phases": json.dumps({"2220": float(mode["phase"])}),
    }
    config["Plot"] = {"imr-samples": imr_samples_path}

    lines: list[str] = []
    for section in config.sections():
        lines.append(f"[{section}]")
        lines.extend(f"{key}={value}" for key, value in config[section].items())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _build_run_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    wsl_config = _to_wsl_path(config_path)
    command = f"source {args.wsl_venv}/bin/activate && pyRing --config-file '{wsl_config}'"
    return ["wsl", "bash", "-lc", command]


def _write_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _launch_run(args: argparse.Namespace, run_dir: Path, config_path: Path) -> int | None:
    stdout_path = run_dir / "pyring_stdout.log"
    stderr_path = run_dir / "pyring_stderr.log"
    command = _build_run_command(args, config_path)

    if args.dry_run:
        print("Dry run command:")
        print(" ".join(command))
        return None

    if args.wait:
        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
            proc = subprocess.run(command, stdout=stdout_file, stderr=stderr_file, check=False)
        return proc.returncode

    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
        proc = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file, creationflags=creationflags)
    return proc.pid


def _load_structured_posterior(run_dir: Path) -> np.ndarray:
    posterior_dat = run_dir / "Nested_sampler" / "posterior.dat"
    if posterior_dat.exists():
        data = np.genfromtxt(posterior_dat, names=True, deletechars="")
        return np.atleast_1d(data)

    posterior_h5 = run_dir / "Nested_sampler" / "cpnest.h5"
    if posterior_h5.exists():
        with h5py.File(posterior_h5, "r") as handle:
            data = np.array(handle["combined"]["posterior_samples"])
        return np.atleast_1d(data)

    raise FileNotFoundError(f"No posterior samples found in {run_dir}")


def _posterior_exists(run_dir: Path) -> bool:
    return (run_dir / "Nested_sampler" / "posterior.dat").exists() or (run_dir / "Nested_sampler" / "cpnest.h5").exists()


def _resample_indices(count: int, target: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = count < target
    return rng.choice(count, size=target, replace=replace)


def _export_posterior(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    data = _load_structured_posterior(run_dir)
    names = list(data.dtype.names or ())
    if "Mf" not in names or "af" not in names:
        raise ValueError(f"{run_dir} posterior missing Mf/af columns; names={names}")

    raw_count = int(data.shape[0])
    export_count = raw_count if args.target_samples <= 0 else int(args.target_samples)
    if export_count == raw_count:
        indices = np.arange(raw_count, dtype=int)
    else:
        indices = _resample_indices(raw_count, export_count, args.export_seed)
    selected = data[indices]

    export_dir = DEFAULT_PYRING_ROOT.resolve()
    export_dir.mkdir(parents=True, exist_ok=True)
    export_run_npz = run_dir / "kerr220_pyring.npz"
    export_stable_npz = export_dir / "kerr220_pyring.npz"

    payload: dict[str, Any] = {
        "Mf_msun": np.asarray(selected["Mf"], dtype=np.float64),
        "chi_f": np.asarray(selected["af"], dtype=np.float64),
        "raw_sample_count": np.array(raw_count, dtype=np.int64),
        "export_sample_count": np.array(export_count, dtype=np.int64),
    }
    for optional_key in ("A2220", "phi2220", "logL", "logPrior", "logPost"):
        if optional_key in names:
            payload[optional_key] = np.asarray(selected[optional_key], dtype=np.float64)

    np.savez(export_run_npz, **payload)
    np.savez(export_stable_npz, **payload)

    manifest_path = export_dir / "manifest_pyring.json"
    manifest: dict[str, Any]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"cases": {}}
    manifest.update(
        {
            "pyring_version": "2.3.0",
            "cpnest_live_points": int(args.nlive),
            "cpnest_max_mcmc_steps": int(args.maxmcmc),
            "export_target_samples": int(args.target_samples),
            "cases": {
                **manifest.get("cases", {}),
                "kerr220": {
                    "run_dir": str(run_dir),
                    "run_npz": str(export_run_npz),
                    "stable_npz": str(export_stable_npz),
                    "raw_sample_count": raw_count,
                    "export_sample_count": export_count,
                },
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "run_dir": str(run_dir),
        "posterior_columns": names,
        "raw_sample_count": raw_count,
        "export_sample_count": export_count,
        "run_npz": str(export_run_npz),
        "stable_npz": str(export_stable_npz),
        "manifest_path": str(manifest_path),
    }
    (run_dir / "pyring_export_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.nlive = 32
        args.maxmcmc = 32
        args.wait = True
        args.background = False

    if args.background and args.wait:
        raise ValueError("Use either --background or --wait, not both")

    run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_kerr220_payload(args.config.resolve())
    config_text = _build_config_text(payload, args=args, run_dir=run_dir)
    config_path = run_dir / "kerr220_pyring.ini"
    config_path.write_text(config_text, encoding="utf-8")

    meta = {
        "case": "kerr220",
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "stdout_log": str(run_dir / "pyring_stdout.log"),
        "stderr_log": str(run_dir / "pyring_stderr.log"),
        "nlive": int(args.nlive),
        "maxmcmc": int(args.maxmcmc),
        "sampling_rate": float(args.sampling_rate),
        "analysis_duration": float(args.analysis_duration),
        "signal_chunksize": float(args.signal_chunksize),
        "noise_chunksize": float(args.noise_chunksize),
        "bandpass_hz": None if args.disable_bandpass else [float(args.f_min_bp), float(args.f_max_bp)],
        "bandpass_enabled": not args.disable_bandpass,
        "fix_t": float(args.fix_t),
        "reference_amplitude": float(args.reference_amplitude),
        "mode_amplitude_direct": float(payload["modes"][0]["amplitude"]),
        "mode_amplitude_relative": float(payload["modes"][0]["amplitude"]) / float(args.reference_amplitude),
        "mode_phase": float(payload["modes"][0]["phase"]),
        "zero_noise_injection": True,
        "paper_like_pyring_path": {
            "sampling_rate_hz": 4096.0,
            "analysis_duration_s": 0.2,
            "bandpass_hz": [100.0, 500.0],
            "truncate": True,
            "fft_acf": True,
            "acf_simple_norm": True,
        },
    }
    _write_run_meta(run_dir, meta)

    if args.export_only:
        summary = _export_posterior(args, run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"Prepared pyRing config: {config_path}")
    print(f"Run directory: {run_dir}")
    if not args.wait and not args.background:
        print("Launch mode: prepare-only")
        return

    launch_mode = "background" if args.background else "foreground"
    print(f"Launch mode: {launch_mode}")

    exit_or_pid = _launch_run(args, run_dir, config_path)
    if args.dry_run:
        return

    if args.wait:
        print(f"pyRing exit code: {exit_or_pid}")
        if int(exit_or_pid) != 0 and not _posterior_exists(run_dir):
            raise SystemExit(int(exit_or_pid))
        if int(exit_or_pid) != 0:
            print("pyRing returned a non-zero exit code after producing posterior samples; proceeding with export.")
        summary = _export_posterior(args, run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    print(f"Launched pyRing background process with pid={exit_or_pid}")


if __name__ == "__main__":
    main()
