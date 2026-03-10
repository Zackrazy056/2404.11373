"""Run stage-2 Kerr220 preprocessing ablations and summarize round-1 quality.

This keeps the detector-local truncation semantics fixed and probes whether
moving toward pyRing-like preprocessing helps round-1 TSNPE contraction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = ROOT / "scripts" / "08_run_fig1_paper_precision.py"


@dataclass(frozen=True)
class AblationSpec:
    label: str
    sample_rate_hz: float
    duration_s: float
    bandpass_min_hz: float | None = None
    bandpass_max_hz: float | None = None


ABLATONS = [
    AblationSpec("detector_local_2048_0p1_nobp", 2048.0, 0.1),
    AblationSpec("detector_local_4096_0p1_nobp", 4096.0, 0.1),
    AblationSpec("detector_local_4096_0p2_nobp", 4096.0, 0.2),
    AblationSpec("detector_local_4096_0p2_bp100_500", 4096.0, 0.2, 100.0, 500.0),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage-2 Kerr220 preprocessing ablations")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=240411373)
    parser.add_argument("--num-sim-first", type=int, default=50_000)
    parser.add_argument("--training-batch-size", type=int, default=512)
    parser.add_argument("--output-root", type=Path, default=Path("reports/posteriors/kerr220_preprocessing_ablation_stage2"))
    parser.add_argument("--summary-json", type=Path, default=Path("reports/audits/kerr220_preprocessing_ablation_stage2.json"))
    parser.add_argument("--summary-md", type=Path, default=Path("reports/audits/kerr220_preprocessing_ablation_stage2.md"))
    parser.add_argument("--stop-after-epochs", type=int, default=20)
    parser.add_argument("--max-num-epochs", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=0)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_payload(run_dir: Path) -> tuple[str, dict[str, Any]]:
    summary_path = run_dir / "kerr220_run_summary.json"
    failure_path = run_dir / "kerr220_failure_summary.json"
    if summary_path.exists():
        return "completed", _load_json(summary_path)
    if failure_path.exists():
        return "failed", _load_json(failure_path)
    raise FileNotFoundError(f"No summary/failure payload under {run_dir}")


def _round1_summary(state: str, payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics") if state == "completed" else payload.get("partial_diagnostics", [])
    round1 = diagnostics[0] if diagnostics else {}
    preprocessing = payload.get("preprocessing", {})
    snr = payload.get("snr", {})
    return {
        "state": state,
        "round1_volume": round1.get("truncated_prior_volume"),
        "round1_probe_accept": round1.get("probe_acceptance_rate"),
        "round1_epochs": round1.get("epochs_trained_this_round"),
        "round1_training_seconds": round1.get("training_seconds"),
        "round1_total_seconds": round1.get("round_total_seconds"),
        "contraction_ok": bool(
            round1.get("truncated_prior_volume") is not None
            and round1.get("probe_acceptance_rate") is not None
            and float(round1["truncated_prior_volume"]) < 1.0
            and float(round1["probe_acceptance_rate"]) < 1.0
        ),
        "measured_network_snr": snr.get("measured_network_snr"),
        "per_detector_snr": snr.get("per_detector"),
        "preprocessing": {
            "sample_rate_hz": preprocessing.get("sample_rate_hz"),
            "duration_s": preprocessing.get("duration_s"),
            "input_dim": preprocessing.get("input_dim"),
            "per_detector_bins": preprocessing.get("per_detector_bins"),
            "bandpass": preprocessing.get("bandpass"),
            "window_anchor": preprocessing.get("window_anchor"),
        },
        "error": payload.get("error"),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _run_one(args: argparse.Namespace, spec: AblationSpec) -> dict[str, Any]:
    run_dir = _resolve(args.output_root / spec.label)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(RUN_SCRIPT),
        "--case",
        "kerr220",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--max-rounds",
        "1",
        "--num-sim-first",
        str(args.num_sim_first),
        "--posterior-samples",
        "2000",
        "--training-batch-size",
        str(args.training_batch_size),
        "--stop-after-epochs",
        str(args.stop_after_epochs),
        "--max-num-epochs",
        str(args.max_num_epochs),
        "--truncation-device",
        "cpu",
        "--observation-timing-mode",
        "detector_local_truncation",
        "--override-sample-rate-hz",
        str(spec.sample_rate_hz),
        "--override-duration-s",
        str(spec.duration_s),
        "--output-dir",
        str(run_dir),
    ]
    if spec.bandpass_min_hz is not None and spec.bandpass_max_hz is not None:
        cmd.extend(["--bandpass-min-hz", str(spec.bandpass_min_hz), "--bandpass-max-hz", str(spec.bandpass_max_hz)])

    attempt = 0
    while True:
        attempt += 1
        print(f"[stage2-ablation] start {spec.label} attempt={attempt}", flush=True)
        started = time.time()
        result = subprocess.run(cmd, cwd=str(ROOT))
        elapsed = time.time() - started
        try:
            state, payload = _latest_payload(run_dir)
            summary = _round1_summary(state, payload)
            summary["exit_code"] = int(result.returncode)
            summary["wall_seconds"] = float(elapsed)
            summary["run_dir"] = str(run_dir)
            summary["label"] = spec.label
            summary["requested"] = {
                "sample_rate_hz": spec.sample_rate_hz,
                "duration_s": spec.duration_s,
                "bandpass_min_hz": spec.bandpass_min_hz,
                "bandpass_max_hz": spec.bandpass_max_hz,
            }
            return summary
        except FileNotFoundError:
            if attempt > args.retries:
                return {
                    "label": spec.label,
                    "run_dir": str(run_dir),
                    "requested": {
                        "sample_rate_hz": spec.sample_rate_hz,
                        "duration_s": spec.duration_s,
                        "bandpass_min_hz": spec.bandpass_min_hz,
                        "bandpass_max_hz": spec.bandpass_max_hz,
                    },
                    "state": "missing_artifacts",
                    "exit_code": int(result.returncode),
                    "wall_seconds": float(elapsed),
                    "error": "no_summary_or_failure_payload",
                }
            print(f"[stage2-ablation] missing artifacts for {spec.label}, retrying", flush=True)


def main() -> None:
    args = parse_args()
    summary_json = _resolve(args.summary_json)
    summary_md = _resolve(args.summary_md)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_md.parent.mkdir(parents=True, exist_ok=True)

    results = [_run_one(args, spec) for spec in ABLATONS]
    report = {
        "case": "kerr220",
        "timing_mode": "detector_local_truncation",
        "runs": results,
    }
    summary_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for item in results:
        bandpass = item.get("requested", {})
        bp_text = (
            f"{bandpass.get('bandpass_min_hz'):.0f}-{bandpass.get('bandpass_max_hz'):.0f}"
            if bandpass.get("bandpass_min_hz") is not None and bandpass.get("bandpass_max_hz") is not None
            else "none"
        )
        rows.append(
            "| {label} | {state} | {sr} | {dur} | {bp} | {snr} | {vol} | {acc} | {contract} | {epochs} |".format(
                label=item.get("label"),
                state=item.get("state"),
                sr=_fmt(item.get("requested", {}).get("sample_rate_hz")),
                dur=_fmt(item.get("requested", {}).get("duration_s")),
                bp=bp_text,
                snr=_fmt(item.get("measured_network_snr")),
                vol=_fmt(item.get("round1_volume")),
                acc=_fmt(item.get("round1_probe_accept")),
                contract=_fmt(item.get("contraction_ok")),
                epochs=_fmt(item.get("round1_epochs")),
            )
        )
    md = "\n".join(
        [
            "# Kerr220 Preprocessing Ablation Stage-2",
            "",
            "| label | state | sample_rate_hz | duration_s | bandpass_hz | network_snr | round1_volume | round1_probe_accept | contraction_ok | epochs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Raw results",
            json.dumps(report, indent=2, ensure_ascii=False),
            "",
        ]
    )
    summary_md.write_text(md, encoding="utf-8")
    print(f"Saved JSON: {summary_json}")
    print(f"Saved Markdown: {summary_md}")


if __name__ == "__main__":
    main()
