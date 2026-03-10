"""Generate ringdown injections from YAML config.

Usage:
    python scripts/03_generate_injection.py --config configs/injections/kerr220.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import load_yaml_config  # noqa: E402
from rd_sbi.simulator.injection import generate_injection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one ringdown injection from config")
    parser.add_argument("--config", type=Path, required=True, help="Path to injection YAML config")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output .npz override (otherwise uses config output.path)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    output_path = args.output
    if output_path is None:
        output_path = ROOT / str(cfg["output"]["path"])
    elif not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path, meta_path = generate_injection(cfg=cfg, output_path=output_path)

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["config"] = str(args.config)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Generated injection: {output_path}")
    print(f"Metadata: {meta_path}")
    print("Bins per detector: 204")


if __name__ == "__main__":
    main()
