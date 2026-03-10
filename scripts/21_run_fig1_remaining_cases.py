"""Run the remaining Fig.1 cases and build the final corner overlay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ("kerr221", "kerr330")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run remaining Fig.1 cases and produce the final corner plot")
    parser.add_argument("--cases", type=str, default="kerr221,kerr330")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--sbi-root-output", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--pyring-dir", type=Path, default=Path("reports/posteriors/pyring"))
    parser.add_argument("--kerr220-sbi-dir", type=Path, default=Path("reports/posteriors/fig1_paper_precision/kerr220_20260310-fig1220-detlocal"))
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=60.0)
    parser.add_argument("--skip-pyring", action="store_true")
    parser.add_argument("--skip-corner", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _run_and_log(cmd: list[str], log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            handle.write(line)
            handle.flush()
        ret = proc.wait()
        handle.write(f"[exit={ret}]\n\n")
        handle.flush()
        if ret != 0:
            raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    selected = [case.strip().lower() for case in args.cases.split(",") if case.strip()]
    selected = [case for case in selected if case in CASES]
    if not selected:
        raise ValueError(f"No valid case selected; choices={CASES}")

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    sbi_root_output = _resolve(args.sbi_root_output)
    pyring_dir = _resolve(args.pyring_dir)
    kerr220_sbi_dir = _resolve(args.kerr220_sbi_dir)
    log_path = _resolve(args.log_path or (sbi_root_output / f"run_{run_tag}_remaining_cases.log"))

    sbi_case_dirs: dict[str, Path] = {"kerr220": kerr220_sbi_dir}

    meta = {
        "run_tag": run_tag,
        "device": args.device,
        "selected_cases": selected,
        "log_path": str(log_path),
        "pyring_dir": str(pyring_dir),
        "kerr220_sbi_dir": str(kerr220_sbi_dir),
    }
    meta_path = sbi_root_output / f"run_{run_tag}_remaining_cases_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    if not args.skip_pyring:
        for case in selected:
            _run_and_log(
                [
                    sys.executable,
                    "scripts/19_run_pyring_fig1_case.py",
                    "--case",
                    case,
                    "--run-tag",
                    f"{run_tag}-locallike-main",
                    "--wait",
                    "--sampling-rate",
                    "2048",
                    "--analysis-duration",
                    "0.1",
                    "--disable-bandpass",
                    "--nlive",
                    "2048",
                    "--maxmcmc",
                    "2048",
                ],
                log_path,
            )

    for case in selected:
        out_dir = sbi_root_output / f"{case}_{run_tag}-detlocal"
        out_dir.mkdir(parents=True, exist_ok=True)
        sbi_case_dirs[case] = out_dir
        _run_and_log(
            [
                sys.executable,
                "-u",
                "scripts/08_run_fig1_paper_precision.py",
                "--case",
                case,
                "--device",
                args.device,
                "--output-dir",
                str(out_dir),
                "--status-path",
                str(out_dir / f"{case}_status.json"),
                "--heartbeat-log",
                str(out_dir / f"{case}_heartbeat.log"),
                "--heartbeat-interval-seconds",
                str(args.heartbeat_interval_seconds),
                "--observation-timing-mode",
                "detector_local_truncation",
            ],
            log_path,
        )

    if not args.skip_corner:
        sbi_case_dirs_arg = ",".join(f"{case}={path}" for case, path in sbi_case_dirs.items())
        _run_and_log(
            [
                sys.executable,
                "scripts/20_plot_fig1_corner_overlay.py",
                "--cases",
                "kerr220,kerr221,kerr330",
                "--sbi-case-dirs",
                sbi_case_dirs_arg,
                "--pyring-dir",
                str(pyring_dir),
                "--style",
                "paper",
                "--output-figure",
                str(ROOT / "reports" / "figures" / f"fig1_corner_overlay_paper_{run_tag}.png"),
                "--output-summary",
                str(ROOT / "reports" / "figures" / f"fig1_corner_overlay_paper_{run_tag}.json"),
            ],
            log_path,
        )
        _run_and_log(
            [
                sys.executable,
                "scripts/20_plot_fig1_corner_overlay.py",
                "--cases",
                "kerr220,kerr221,kerr330",
                "--sbi-case-dirs",
                sbi_case_dirs_arg,
                "--pyring-dir",
                str(pyring_dir),
                "--style",
                "audit",
                "--disable-raw-pyring",
                "--output-figure",
                str(ROOT / "reports" / "figures" / f"fig1_corner_overlay_audit_{run_tag}.png"),
                "--output-summary",
                str(ROOT / "reports" / "figures" / f"fig1_corner_overlay_audit_{run_tag}.json"),
            ],
            log_path,
        )

    print(f"log_path={log_path}")
    print(f"meta_path={meta_path}")
    for case, path in sbi_case_dirs.items():
        print(f"{case}_sbi_dir={path}")


if __name__ == "__main__":
    main()
