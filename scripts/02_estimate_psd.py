"""Estimate one-sided PSD from GWOSC strain files.

Usage:
    python scripts/02_estimate_psd.py
    python scripts/02_estimate_psd.py --input-glob "data/raw/gwosc/GW150914/*-32.hdf5"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rd_sbi.noise.psd import estimate_psd_welch, load_gwosc_strain_hdf5  # noqa: E402


@dataclass(frozen=True)
class PSDFileRecord:
    source_hdf5: str
    output_npz: str
    detector: str | None
    sample_rate_hz: float
    n_frequency_bins: int
    nperseg: int
    noverlap: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate PSDs for GWOSC HDF5 strain files")
    parser.add_argument(
        "--input-glob",
        type=str,
        default="data/raw/gwosc/GW150914/*-32.hdf5",
        help="Glob pattern to input HDF5 files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/psd/GW150914"),
        help="Directory where PSD npz files and manifest are saved",
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=16384,
        help="Welch segment length (default: 16384 samples = 4s at 4096Hz)",
    )
    parser.add_argument(
        "--noverlap",
        type=int,
        default=8192,
        help="Welch overlap length (default: 50% overlap for nperseg=16384)",
    )
    return parser.parse_args()


def detector_tag(file_path: Path, fallback: str | None = None) -> str:
    name = file_path.name.upper()
    if "H-H1_" in name:
        return "H1"
    if "L-L1_" in name:
        return "L1"
    return fallback or file_path.stem


def main() -> None:
    args = parse_args()
    input_files = sorted(ROOT.glob(args.input_glob))
    if not input_files:
        raise FileNotFoundError(f"No files found for input_glob={args.input_glob!r}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[PSDFileRecord] = []
    for file_path in input_files:
        try:
            series = load_gwosc_strain_hdf5(file_path)
        except OSError as exc:
            print(f"Skip unreadable HDF5 file: {file_path} ({exc})")
            continue
        psd_result = estimate_psd_welch(
            series.strain,
            series.sample_rate_hz,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
        )

        tag = detector_tag(file_path, series.detector)
        output_path = args.output_dir / f"{tag}_psd_welch.npz"
        np.savez_compressed(
            output_path,
            frequency_hz=psd_result.frequency_hz,
            psd=psd_result.psd,
            sample_rate_hz=np.array(psd_result.sample_rate_hz),
            nperseg=np.array(psd_result.nperseg),
            noverlap=np.array(psd_result.noverlap),
            source_hdf5=np.array(str(file_path)),
            detector=np.array(tag),
        )

        records.append(
            PSDFileRecord(
                source_hdf5=str(file_path),
                output_npz=str(output_path),
                detector=tag,
                sample_rate_hz=psd_result.sample_rate_hz,
                n_frequency_bins=int(psd_result.frequency_hz.shape[0]),
                nperseg=psd_result.nperseg,
                noverlap=psd_result.noverlap,
            )
        )
        print(f"Saved PSD: {output_path}")

    if not records:
        raise RuntimeError("No valid HDF5 files were processed.")

    manifest_path = args.output_dir / "manifest_psd.json"
    manifest = {
        "event": "GW150914",
        "input_glob": args.input_glob,
        "file_count": len(records),
        "files": [asdict(record) for record in records],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
