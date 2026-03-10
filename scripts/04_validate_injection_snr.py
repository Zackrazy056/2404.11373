"""Validate injection SNRs against target values in configs.

Usage:
    python scripts/04_validate_injection_snr.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import load_yaml_config  # noqa: E402
from rd_sbi.eval.snr import compute_network_snr, load_psd_npz  # noqa: E402
from rd_sbi.simulator.injection import generate_injection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SNR consistency for injection configs")
    parser.add_argument(
        "--configs-glob",
        type=str,
        default="configs/injections/kerr*.yaml",
        help="Glob pattern for injection configs",
    )
    parser.add_argument(
        "--psd-dir",
        type=Path,
        default=Path("data/processed/psd/GW150914"),
        help="Directory containing detector PSD npz files",
    )
    parser.add_argument(
        "--rel-tol",
        type=float,
        default=0.05,
        help="Relative tolerance for target SNR comparison",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["network", "rss_mean"],
        default="network",
        help="Which SNR metric to compare against target",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/tables/snr_validation_report.json"),
        help="Output JSON report path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = sorted(ROOT.glob(args.configs_glob))
    if not config_paths:
        raise FileNotFoundError(f"No configs found for {args.configs_glob}")

    psd_by_detector = {
        "H1": load_psd_npz(str((ROOT / args.psd_dir / "H1_psd_welch.npz").resolve())),
        "L1": load_psd_npz(str((ROOT / args.psd_dir / "L1_psd_welch.npz").resolve())),
    }

    tmp_dir = ROOT / "data/cache/snr_validation"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    all_pass = True

    for cfg_path in config_paths:
        cfg = load_yaml_config(cfg_path)
        name = str(cfg.get("name", cfg_path.stem))
        target = float(cfg.get("target_snr", np.nan))

        tmp_npz = tmp_dir / f"{cfg_path.stem}__tmp.npz"
        generate_injection(cfg=cfg, output_path=tmp_npz)

        payload = np.load(tmp_npz)
        sample_rate_hz = float(payload["sample_rate_hz"])
        strains = {k.replace("strain_", ""): payload[k] for k in payload.files if k.startswith("strain_")}
        snr = compute_network_snr(strains=strains, sample_rate_hz=sample_rate_hz, psd_by_detector=psd_by_detector)

        metric_value = snr.network_snr if args.metric == "network" else snr.rss_mean_snr
        if np.isfinite(target):
            rel_err = abs(metric_value - target) / max(target, 1e-12)
            passed = bool(rel_err <= args.rel_tol)
        else:
            rel_err = float("nan")
            passed = True

        all_pass = all_pass and passed
        rows.append(
            {
                "config": str(cfg_path),
                "name": name,
                "target_snr": target,
                "metric": args.metric,
                "metric_value": metric_value,
                "network_snr": snr.network_snr,
                "rss_mean_snr": snr.rss_mean_snr,
                "per_detector": {x.detector: x.snr for x in snr.per_detector},
                "relative_error": rel_err,
                "pass": passed,
            }
        )

        print(
            f"{name}: network={snr.network_snr:.3f}, rss_mean={snr.rss_mean_snr:.3f}, "
            f"target={target:.3f}, pass={passed}"
        )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metric": args.metric,
        "relative_tolerance": args.rel_tol,
        "all_pass": all_pass,
        "results": rows,
    }
    (ROOT / args.report).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Report: {ROOT / args.report}")
    if not all_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
