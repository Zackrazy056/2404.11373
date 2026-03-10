"""Batch-generate injections and build train/val cache index.

Usage:
    python scripts/05_build_injection_cache.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import load_yaml_config  # noqa: E402
from rd_sbi.simulator.injection import generate_injection  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build batch injection cache with train/val split")
    parser.add_argument("--configs-glob", type=str, default="configs/injections/kerr*.yaml")
    parser.add_argument("--samples-per-config", type=int, default=100)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=240411373)
    parser.add_argument("--noise-std", type=float, default=None, help="Optional global noise std override")
    parser.add_argument("--output-dir", type=Path, default=Path("data/cache/injections"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing sample files")
    return parser.parse_args()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg_paths = sorted(ROOT.glob(args.configs_glob))
    if not cfg_paths:
        raise FileNotFoundError(f"No config files found for {args.configs_glob}")
    if args.samples_per_config <= 0:
        raise ValueError("samples-per-config must be > 0")
    if not (0.0 <= args.val_fraction < 1.0):
        raise ValueError("val-fraction must satisfy 0 <= val_fraction < 1")

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    row_index = 0
    for cfg_path in cfg_paths:
        cfg = load_yaml_config(cfg_path)
        cfg_name = str(cfg.get("name", cfg_path.stem))
        cfg_seed = int(cfg.get("seed", 0))
        cfg_dir = out_dir / cfg_path.stem
        cfg_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.samples_per_config):
            sample_seed = cfg_seed + i
            sample_id = f"{cfg_path.stem}_{i:05d}"
            npz_path = cfg_dir / f"{sample_id}.npz"

            if npz_path.exists() and not args.overwrite:
                meta_path = npz_path.with_suffix(".json")
            else:
                npz_path, meta_path = generate_injection(
                    cfg=cfg,
                    output_path=npz_path,
                    override_seed=sample_seed,
                    override_noise_std=args.noise_std,
                )

            rows.append(
                {
                    "row_index": row_index,
                    "sample_id": sample_id,
                    "config_path": str(cfg_path),
                    "config_name": cfg_name,
                    "seed": sample_seed,
                    "npz_path": str(npz_path),
                    "meta_path": str(meta_path),
                    "split": "",
                }
            )
            row_index += 1

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(rows))
    n_val = int(round(len(rows) * args.val_fraction))
    val_set = set(int(x) for x in perm[:n_val])
    for idx, row in enumerate(rows):
        row["split"] = "val" if idx in val_set else "train"

    index_jsonl = out_dir / "index.jsonl"
    index_csv = out_dir / "index.csv"
    _write_jsonl(index_jsonl, rows)
    _write_csv(index_csv, rows)

    manifest = {
        "configs_glob": args.configs_glob,
        "samples_per_config": args.samples_per_config,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "noise_std_override": args.noise_std,
        "total_samples": len(rows),
        "train_samples": int(sum(1 for r in rows if r["split"] == "train")),
        "val_samples": int(sum(1 for r in rows if r["split"] == "val")),
        "index_jsonl": str(index_jsonl),
        "index_csv": str(index_csv),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Cache built at: {out_dir}")
    print(f"Total samples: {manifest['total_samples']}")
    print(f"Train/Val: {manifest['train_samples']}/{manifest['val_samples']}")
    print(f"Index: {index_jsonl}")


if __name__ == "__main__":
    main()
