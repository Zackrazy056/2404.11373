"""PSD estimation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class StrainSeries:
    """Time-domain strain series with metadata."""

    strain: np.ndarray
    sample_rate_hz: float
    gps_start: float | None
    detector: str | None
    source: str


@dataclass(frozen=True)
class PSDResult:
    """One-sided PSD estimate."""

    frequency_hz: np.ndarray
    psd: np.ndarray
    sample_rate_hz: float
    nperseg: int
    noverlap: int
    window: str


def load_gwosc_strain_hdf5(path: str | Path) -> StrainSeries:
    """Load GWOSC strain data from an event HDF5 file.

    Expected dataset path:
    - `strain/Strain`

    Sample rate is inferred from `Xspacing` attribute.
    """
    source_path = Path(path)
    with h5py.File(source_path, "r") as handle:
        data = np.asarray(handle["strain/Strain"][()], dtype=np.float64)
        xspacing = float(handle["strain/Strain"].attrs["Xspacing"])
        sample_rate_hz = 1.0 / xspacing

        gps_start = None
        detector = None
        if "meta/GPSstart" in handle:
            gps_start = float(handle["meta/GPSstart"][()])
        if "meta/Detector" in handle:
            detector_raw = handle["meta/Detector"][()]
            detector = detector_raw.decode("utf-8") if isinstance(detector_raw, bytes) else str(detector_raw)

    return StrainSeries(
        strain=data,
        sample_rate_hz=sample_rate_hz,
        gps_start=gps_start,
        detector=detector,
        source=str(source_path),
    )


def estimate_psd_welch(
    strain: np.ndarray,
    sample_rate_hz: float,
    *,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str = "constant",
) -> PSDResult:
    """Estimate one-sided PSD with Welch's method."""
    if strain.ndim != 1:
        raise ValueError(f"strain must be 1D, got shape={strain.shape}")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    n = strain.shape[0]
    if nperseg is None:
        # 4-second segments by default, clipped by available length.
        nperseg = int(min(n, max(256, round(sample_rate_hz * 4.0))))
    nperseg = int(min(nperseg, n))
    if nperseg < 8:
        raise ValueError("nperseg is too small")

    if noverlap is None:
        noverlap = nperseg // 2
    noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError("noverlap must be smaller than nperseg")

    frequency_hz, psd = welch(
        strain,
        fs=sample_rate_hz,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=detrend,
        return_onesided=True,
        scaling="density",
    )

    return PSDResult(
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        psd=np.asarray(psd, dtype=np.float64),
        sample_rate_hz=float(sample_rate_hz),
        nperseg=nperseg,
        noverlap=noverlap,
        window=window,
    )
