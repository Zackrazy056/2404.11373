"""Run monitored CPU paper-budget reproduction for Kerr220 and auto-plot on success."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run monitored Kerr220 reproduction")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--root-output", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=60.0)
    parser.add_argument(
        "--observation-timing-mode",
        type=str,
        choices=["shared_window_anchor", "detector_local_truncation"],
        default=None,
    )
    parser.add_argument("--override-sample-rate-hz", type=float, default=None)
    parser.add_argument("--override-duration-s", type=float, default=None)
    parser.add_argument("--bandpass-min-hz", type=float, default=None)
    parser.add_argument("--bandpass-max-hz", type=float, default=None)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _run_and_log(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = _resolve(args.root_output) / f"kerr220_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    status_path = out_dir / "kerr220_status.json"
    heartbeat_log = out_dir / "kerr220_heartbeat.log"
    case_log = out_dir / "kerr220.log"
    plot_log = out_dir / "kerr220_plot.log"
    meta_path = out_dir / "run_meta.json"

    meta = {
        "case": "kerr220",
        "device": args.device,
        "run_tag": run_tag,
        "out_dir": str(out_dir),
        "status_path": str(status_path),
        "heartbeat_log": str(heartbeat_log),
        "case_log": str(case_log),
        "observation_timing_mode": args.observation_timing_mode,
        "override_sample_rate_hz": args.override_sample_rate_hz,
        "override_duration_s": args.override_duration_s,
        "bandpass_min_hz": args.bandpass_min_hz,
        "bandpass_max_hz": args.bandpass_max_hz,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Run directory: {out_dir}")
    print(f"Status file: {status_path}")
    print(f"Heartbeat log: {heartbeat_log}")
    print(f"Case log: {case_log}")
    print(f"Meta: {meta_path}")

    train_cmd = [
        sys.executable,
        "-u",
        "scripts/08_run_fig1_paper_precision.py",
        "--case",
        "kerr220",
        "--device",
        args.device,
        "--output-dir",
        str(out_dir),
        "--status-path",
        str(status_path),
        "--heartbeat-log",
        str(heartbeat_log),
        "--heartbeat-interval-seconds",
        str(args.heartbeat_interval_seconds),
    ]
    if args.observation_timing_mode is not None:
        train_cmd.extend(["--observation-timing-mode", args.observation_timing_mode])
    if args.override_sample_rate_hz is not None:
        train_cmd.extend(["--override-sample-rate-hz", str(args.override_sample_rate_hz)])
    if args.override_duration_s is not None:
        train_cmd.extend(["--override-duration-s", str(args.override_duration_s)])
    if args.bandpass_min_hz is not None:
        train_cmd.extend(["--bandpass-min-hz", str(args.bandpass_min_hz)])
    if args.bandpass_max_hz is not None:
        train_cmd.extend(["--bandpass-max-hz", str(args.bandpass_max_hz)])

    _run_and_log(train_cmd, case_log)

    fig_path = ROOT / "reports" / "figures" / f"fig1_kerr220_paper_grade_{run_tag}.png"
    fig_summary = ROOT / "reports" / "figures" / f"fig1_kerr220_paper_grade_{run_tag}.json"
    _run_and_log(
        [
            sys.executable,
            "scripts/15_plot_fig1_case_paper_grade.py",
            "--case",
            "kerr220",
            "--run-dir",
            str(out_dir),
            "--output-figure",
            str(fig_path),
            "--output-summary",
            str(fig_summary),
        ],
        plot_log,
    )

    print(f"Figure: {fig_path}")
    print(f"Figure summary: {fig_summary}")


if __name__ == "__main__":
    main()
