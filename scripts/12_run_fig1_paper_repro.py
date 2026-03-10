"""Orchestrate paper-precision Fig.1 reproduction run.

Flow:
1) preflight audit (must pass)
2) run 08_run_fig1_paper_precision.py for selected cases
3) optional pyRing overlay + artifact-level audit
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ("kerr220", "kerr221", "kerr330")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Fig.1 paper-precision SBI reproduction")
    parser.add_argument("--cases", type=str, default="kerr220,kerr221,kerr330")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--root-output", type=Path, default=Path("reports/posteriors/fig1_paper_precision"))
    parser.add_argument("--pyring-dir", type=Path, default=Path("reports/posteriors/pyring"))
    parser.add_argument("--with-overlay", action="store_true", help="Run pyRing overlay step if files exist")
    parser.add_argument("--with-artifact-audit", action="store_true", help="Run artifact-level strict audit at end")
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
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed ({ret}): {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    selected = [c.strip().lower() for c in args.cases.split(",") if c.strip()]
    selected = [c for c in selected if c in CASES]
    if not selected:
        raise ValueError(f"No valid case in --cases. Choices: {','.join(CASES)}")

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = _resolve(args.root_output) / f"run_{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    preflight_log = out_dir / "preflight_audit.log"
    _run_and_log(
        [
            sys.executable,
            "scripts/11_audit_fig1_paper_spec.py",
            "--output-json",
            str(out_dir / "preflight_audit.json"),
        ],
        preflight_log,
    )

    for case in selected:
        case_log = out_dir / f"{case}.log"
        _run_and_log(
            [
                sys.executable,
                "scripts/08_run_fig1_paper_precision.py",
                "--case",
                case,
                "--device",
                args.device,
                "--output-dir",
                str(out_dir),
            ],
            case_log,
        )

    if args.with_overlay:
        overlay_log = out_dir / "overlay.log"
        _run_and_log(
            [
                sys.executable,
                "scripts/10_overlay_fig1_pyring.py",
                "--cases",
                ",".join(selected),
                "--sbi-dir",
                str(out_dir),
                "--pyring-dir",
                str(_resolve(args.pyring_dir)),
                "--output-figure",
                str(ROOT / "reports/figures" / f"fig1_overlay_sbi_pyring_{run_tag}.png"),
                "--summary-path",
                str(out_dir / "overlay_summary.json"),
            ],
            overlay_log,
        )

    if args.with_artifact_audit:
        artifact_log = out_dir / "artifact_audit.log"
        _run_and_log(
            [
                sys.executable,
                "scripts/11_audit_fig1_paper_spec.py",
                "--check-artifacts",
                "--run-root",
                str(out_dir),
                "--pyring-dir",
                str(_resolve(args.pyring_dir)),
                "--overlay-summary",
                str(out_dir / "overlay_summary.json"),
                "--output-json",
                str(out_dir / "artifact_audit.json"),
            ],
            artifact_log,
        )

    print(f"Run directory: {out_dir}")


if __name__ == "__main__":
    main()
