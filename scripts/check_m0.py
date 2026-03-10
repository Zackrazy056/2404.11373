"""M0 bootstrap checker.

Usage:
    python scripts/check_m0.py
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.config import load_yaml_config  # noqa: E402
from rd_sbi.utils import set_global_seed  # noqa: E402


REQUIRED_PATHS = [
    "configs/defaults.yaml",
    "docs/ARTIFACT_POLICY.md",
    "src/rd_sbi/config.py",
    "src/rd_sbi/utils/seed.py",
    "src/rd_sbi/io/artifacts.py",
    "reports/figures",
    "reports/tables",
    "reports/logs",
]


def main() -> None:
    root = ROOT
    missing = [p for p in REQUIRED_PATHS if not (root / p).exists()]
    if missing:
        raise FileNotFoundError(f"M0 check failed, missing paths: {missing}")

    cfg = load_yaml_config(root / "configs/defaults.yaml")
    seed = int(cfg["runtime"]["seed"])
    applied = set_global_seed(seed)

    print("M0 check passed")
    print(f"project={cfg['project']['name']}")
    print(f"seed={applied}")


if __name__ == "__main__":
    main()
