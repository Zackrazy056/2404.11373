"""Summarize round-1 Kerr220 preprocessing ablation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Kerr220 preprocessing ablation runs")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/audits/kerr220_preprocessing_ablation_round1.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/audits/kerr220_preprocessing_ablation_round1.md"),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_payload(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "kerr220_run_summary.json"
    failure_path = run_dir / "kerr220_failure_summary.json"
    if summary_path.exists():
        payload = _load_json(summary_path)
        diagnostics = payload.get("diagnostics", [])
        round1 = diagnostics[0] if diagnostics else {}
        state = "completed"
    elif failure_path.exists():
        payload = _load_json(failure_path)
        diagnostics = payload.get("partial_diagnostics", [])
        round1 = diagnostics[0] if diagnostics else {}
        state = "failed"
    else:
        raise FileNotFoundError(f"No run summary or failure summary in {run_dir}")

    preprocessing = payload.get("preprocessing", {})
    if not preprocessing and "latest_status" in payload:
        preprocessing = {}
    snr = payload.get("snr", {})
    mode = (
        preprocessing.get("window_anchor", {}).get("observation_timing_mode")
        or payload.get("params", {}).get("observation_timing_mode")
        or "unknown"
    )
    return {
        "run_dir": str(run_dir),
        "state": state,
        "observation_timing_mode": mode,
        "window_anchor": preprocessing.get("window_anchor"),
        "snr": snr,
        "round1": {
            "truncated_prior_volume": round1.get("truncated_prior_volume"),
            "probe_acceptance_rate": round1.get("probe_acceptance_rate"),
            "epochs_trained_this_round": round1.get("epochs_trained_this_round"),
            "training_seconds": round1.get("training_seconds"),
            "simulation_seconds": round1.get("simulation_seconds"),
            "round_total_seconds": round1.get("round_total_seconds"),
            "hpd_log_prob_threshold": round1.get("hpd_log_prob_threshold"),
            "contraction_ok": bool(
                round1.get("truncated_prior_volume") is not None
                and round1.get("probe_acceptance_rate") is not None
                and float(round1["truncated_prior_volume"]) < 1.0
                and float(round1["probe_acceptance_rate"]) < 1.0
            ),
        },
        "error": payload.get("error"),
    }


def _fmt(value: Any, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    args = parse_args()
    run_a = _resolve(args.run_a)
    run_b = _resolve(args.run_b)
    payload_a = _run_payload(run_a)
    payload_b = _run_payload(run_b)

    report = {
        "case": "kerr220",
        "comparison": [payload_a, payload_b],
    }

    output_json = _resolve(args.output_json)
    output_md = _resolve(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for item in (payload_a, payload_b):
        rows.append(
            "| {mode} | {state} | {snr} | {volume} | {accept} | {contract} | {epochs} |".format(
                mode=item["observation_timing_mode"],
                state=item["state"],
                snr=_fmt(item["snr"].get("measured_network_snr")),
                volume=_fmt(item["round1"]["truncated_prior_volume"]),
                accept=_fmt(item["round1"]["probe_acceptance_rate"]),
                contract=_fmt(item["round1"]["contraction_ok"]),
                epochs=_fmt(item["round1"]["epochs_trained_this_round"]),
            )
        )

    md = "\n".join(
        [
            "# Kerr220 Preprocessing Ablation Round-1 Summary",
            "",
            "| observation_timing_mode | state | network_snr | round1_volume | round1_probe_accept | contraction_ok | epochs |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## Run A",
            json.dumps(payload_a, indent=2, ensure_ascii=False),
            "",
            "## Run B",
            json.dumps(payload_b, indent=2, ensure_ascii=False),
            "",
        ]
    )
    output_md.write_text(md, encoding="utf-8")
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
