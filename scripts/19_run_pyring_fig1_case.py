"""Prepare, launch, and export a pyRing Fig.1 case from Windows via WSL.

Supports the three Fig.1 injections:
- kerr220 : (2,2,0)
- kerr221 : (2,2,0) + (2,2,1)
- kerr330 : (2,2,0) + (3,3,0)

The default execution path uses the locally aligned preprocessing that matches
the current SBI comparison chain:
- 2048 Hz
- 0.1 s analysis duration
- no bandpass
- detector-local pyRing truncation semantics
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
CONFIG_MAP = {
    "kerr220": ROOT / "configs" / "injections" / "kerr220.yaml",
    "kerr221": ROOT / "configs" / "injections" / "kerr221.yaml",
    "kerr330": ROOT / "configs" / "injections" / "kerr330.yaml",
}
DEFAULT_GW150914_DIR = ROOT / "external" / "pyRingGW-2.3.0" / "pyRing" / "data" / "Real_data" / "GW150914"
DEFAULT_PYRING_ROOT = ROOT / "reports" / "posteriors" / "pyring"
DEFAULT_REFERENCE_AMPLITUDE = 1.0e-21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Fig.1 pyRing case from Windows via WSL")
    parser.add_argument("--case", type=str, choices=sorted(CONFIG_MAP.keys()), required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--wsl-venv", type=str, default="~/.venvs/pyring312")
    parser.add_argument("--nlive", type=int, default=2048)
    parser.add_argument("--maxmcmc", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sampling-rate", type=float, default=2048.0)
    parser.add_argument("--analysis-duration", type=float, default=0.1)
    parser.add_argument("--signal-chunksize", type=float, default=4.0)
    parser.add_argument("--noise-chunksize", type=float, default=4.0)
    parser.add_argument("--f-min-bp", type=float, default=100.0)
    parser.add_argument("--f-max-bp", type=float, default=500.0)
    parser.add_argument("--disable-bandpass", action="store_true")
    parser.add_argument("--fix-t", type=float, default=0.0)
    parser.add_argument("--reference-amplitude", type=float, default=DEFAULT_REFERENCE_AMPLITUDE)
    parser.add_argument("--target-samples", type=int, default=20000)
    parser.add_argument("--export-seed", type=int, default=240411373)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Shortcut for a fast foreground smoke run")
    return parser.parse_args()


def _to_wsl_path(path: Path) -> str:
    resolved = path.resolve()
    posix = resolved.as_posix()
    if len(posix) >= 3 and posix[1:3] == ":/":
        return f"/mnt/{posix[0].lower()}{posix[2:]}"
    return posix


def _load_case_payload(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    modes = payload.get("modes", [])
    if not modes:
        raise ValueError(f"{config_path} has no modes")
    return payload


def _resolve_config_path(args: argparse.Namespace) -> Path:
    if args.config is not None:
        return args.config.resolve()
    return CONFIG_MAP[args.case].resolve()


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    if args.run_tag:
        tag = args.run_tag
    else:
        tag = datetime.now().strftime("%Y%m%d-%H%M%S")
        if args.smoke:
            tag = f"{tag}-smoke"
    return (DEFAULT_PYRING_ROOT / f"{args.case}_{tag}").resolve()


def _pyring_mode_string(mode: dict[str, Any]) -> str:
    l = int(mode["l"])
    m = int(mode["m"])
    n = int(mode["n"])
    return f"(2,{l},{m},{n})"


def _mode_tag(mode: dict[str, Any]) -> str:
    l = int(mode["l"])
    m = int(mode["m"])
    n = int(mode["n"])
    return f"2{l}{m}{n}"


def _amp_prior_max(relative_amp: float) -> float:
    return float(max(7.0, relative_amp * 1.5, relative_amp + 5.0))


def _build_config_text(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    run_dir: Path,
) -> str:
    data = payload["data"]
    source = payload["source"]
    remnant = payload["remnant"]
    modes = payload["modes"]

    if args.reference_amplitude <= 0.0:
        raise ValueError("--reference-amplitude must be positive")

    h1_path = _to_wsl_path(DEFAULT_GW150914_DIR / "H-H1_GWOSC_4KHZ_R1-1126259447-32.txt")
    l1_path = _to_wsl_path(DEFAULT_GW150914_DIR / "L-L1_GWOSC_4KHZ_R1-1126259447-32.txt")
    imr_samples_path = _to_wsl_path(DEFAULT_GW150914_DIR / "GW150914_LAL_IMRPhenomP_O1_GWOSC_Mf_af_samples.txt")
    output_path = _to_wsl_path(run_dir)

    kerr_mode_entries = [_pyring_mode_string(mode) for mode in modes]
    kerr_modes = "[" + ", ".join(kerr_mode_entries) + "]"
    kerr_amplitudes = {
        _mode_tag(mode): float(mode["amplitude"]) / float(args.reference_amplitude)
        for mode in modes
    }
    kerr_phases = {
        _mode_tag(mode): float(mode["phase"])
        for mode in modes
    }

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
        "kerr-modes": kerr_modes,
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
    }
    for mode in modes:
        tag = _mode_tag(mode)
        rel_amp = float(mode["amplitude"]) / float(args.reference_amplitude)
        config["Priors"][f"A{tag}-min"] = "0.0"
        config["Priors"][f"A{tag}-max"] = f"{_amp_prior_max(rel_amp):.3f}"

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
        "kerr-amplitudes": json.dumps(kerr_amplitudes),
        "kerr-phases": json.dumps(kerr_phases),
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
    export_run_npz = run_dir / f"{args.case}_pyring.npz"
    export_stable_npz = export_dir / f"{args.case}_pyring.npz"
    raw_run_npz = run_dir / f"{args.case}_pyring_raw.npz"
    raw_stable_npz = export_dir / f"{args.case}_pyring_raw.npz"

    raw_payload: dict[str, Any] = {
        "Mf_msun": np.asarray(data["Mf"], dtype=np.float64),
        "chi_f": np.asarray(data["af"], dtype=np.float64),
        "raw_sample_count": np.array(raw_count, dtype=np.int64),
        "export_sample_count": np.array(raw_count, dtype=np.int64),
        "sample_source": np.array("raw", dtype="<U16"),
    }
    comparison_payload: dict[str, Any] = {
        "Mf_msun": np.asarray(selected["Mf"], dtype=np.float64),
        "chi_f": np.asarray(selected["af"], dtype=np.float64),
        "raw_sample_count": np.array(raw_count, dtype=np.int64),
        "export_sample_count": np.array(export_count, dtype=np.int64),
        "sample_source": np.array("resampled" if export_count != raw_count else "raw", dtype="<U16"),
    }
    for name in names:
        try:
            raw_payload[name] = np.asarray(data[name], dtype=np.float64)
            comparison_payload[name] = np.asarray(selected[name], dtype=np.float64)
        except Exception:
            continue

    np.savez(raw_run_npz, **raw_payload)
    np.savez(raw_stable_npz, **raw_payload)
    np.savez(export_run_npz, **comparison_payload)
    np.savez(export_stable_npz, **comparison_payload)

    manifest_path = export_dir / "manifest_pyring.json"
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
                args.case: {
                    "run_dir": str(run_dir),
                    "run_npz": str(export_run_npz),
                    "stable_npz": str(export_stable_npz),
                    "raw_run_npz": str(raw_run_npz),
                    "raw_stable_npz": str(raw_stable_npz),
                    "raw_sample_count": raw_count,
                    "export_sample_count": export_count,
                },
            },
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "case": args.case,
        "run_dir": str(run_dir),
        "posterior_columns": names,
        "raw_sample_count": raw_count,
        "export_sample_count": export_count,
        "run_npz": str(export_run_npz),
        "stable_npz": str(export_stable_npz),
        "raw_run_npz": str(raw_run_npz),
        "raw_stable_npz": str(raw_stable_npz),
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

    config_path = _resolve_config_path(args)
    run_dir = _resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.export_only:
        if not _posterior_exists(run_dir):
            raise FileNotFoundError(f"Run has no posterior to export: {run_dir}")
        summary = _export_posterior(args, run_dir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    payload = _load_case_payload(config_path)
    config_text = _build_config_text(payload, args=args, run_dir=run_dir)
    pyring_ini = run_dir / f"{args.case}_pyring.ini"
    pyring_ini.write_text(config_text, encoding="utf-8")

    mode_amplitudes = {
        _mode_tag(mode): float(mode["amplitude"])
        for mode in payload["modes"]
    }
    meta = {
        "case": args.case,
        "config_path": str(config_path),
        "pyring_ini": str(pyring_ini),
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
        "mode_amplitudes_direct": mode_amplitudes,
        "mode_amplitudes_relative": {
            key: value / float(args.reference_amplitude)
            for key, value in mode_amplitudes.items()
        },
        "zero_noise_injection": True,
        "locallike_path": {
            "sampling_rate_hz": 2048.0,
            "analysis_duration_s": 0.1,
            "bandpass_enabled": False,
            "truncate": True,
            "fft_acf": True,
        },
    }
    _write_run_meta(run_dir, meta)

    result = _launch_run(args, run_dir, pyring_ini)
    if args.wait:
        if result not in (0, None) and not _posterior_exists(run_dir):
            raise RuntimeError(f"pyRing failed with exit code {result}")
        if _posterior_exists(run_dir):
            summary = _export_posterior(args, run_dir)
            print(json.dumps(summary, indent=2, sort_keys=True))
        else:
            print(f"Run finished but no posterior found yet: {run_dir}")
        return

    print(f"started case={args.case}")
    print(f"run_dir={run_dir}")
    print(f"pyring_ini={pyring_ini}")
    print(f"result={result}")


if __name__ == "__main__":
    main()
