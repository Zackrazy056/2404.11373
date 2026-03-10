"""SNR utilities for ringdown injections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorSNR:
    detector: str
    snr: float


@dataclass(frozen=True)
class NetworkSNRResult:
    per_detector: list[DetectorSNR]
    network_snr: float
    rss_mean_snr: float


def load_psd_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load frequency and one-sided PSD arrays from npz file."""
    data = np.load(path)
    return np.asarray(data["frequency_hz"], dtype=np.float64), np.asarray(data["psd"], dtype=np.float64)


def resample_psd_to_rfft_grid(
    frequency_hz: np.ndarray,
    psd: np.ndarray,
    sample_rate_hz: float,
    n_samples: int,
    *,
    floor: float = 1e-60,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one-sided PSD to the rFFT grid of a target strain."""
    f_target = np.fft.rfftfreq(n_samples, d=1.0 / float(sample_rate_hz))
    psd_target = np.interp(f_target, frequency_hz, psd, left=psd[0], right=psd[-1])
    psd_target = np.maximum(psd_target, floor)
    return f_target.astype(np.float64), psd_target.astype(np.float64)


def compute_detector_snr(strain: np.ndarray, sample_rate_hz: float, frequency_hz: np.ndarray, psd: np.ndarray) -> float:
    """Compute one-detector optimal SNR from one-sided PSD."""
    h = np.asarray(strain, dtype=np.float64)
    n = h.shape[0]
    dt = 1.0 / float(sample_rate_hz)
    h_f = np.fft.rfft(h) * dt
    f, s = resample_psd_to_rfft_grid(
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        psd=np.asarray(psd, dtype=np.float64),
        sample_rate_hz=sample_rate_hz,
        n_samples=n,
    )
    df = f[1] - f[0]

    if h_f.shape[0] > 2:
        rho2 = 4.0 * df * np.sum((np.abs(h_f[1:-1]) ** 2) / s[1:-1])
        rho2 += 2.0 * df * ((np.abs(h_f[0]) ** 2) / s[0] + (np.abs(h_f[-1]) ** 2) / s[-1])
    else:
        rho2 = 4.0 * df * np.sum((np.abs(h_f) ** 2) / s)
    return float(np.sqrt(max(float(rho2), 0.0)))


def compute_network_snr(
    strains: dict[str, np.ndarray],
    sample_rate_hz: float,
    psd_by_detector: dict[str, tuple[np.ndarray, np.ndarray]],
) -> NetworkSNRResult:
    """Compute per-detector and combined network SNR."""
    per: list[DetectorSNR] = []
    sum_sq = 0.0
    for det, strain in strains.items():
        if det not in psd_by_detector:
            raise KeyError(f"Missing PSD for detector {det}")
        f, p = psd_by_detector[det]
        rho = compute_detector_snr(strain, sample_rate_hz, f, p)
        per.append(DetectorSNR(detector=det, snr=rho))
        sum_sq += rho * rho

    network = float(np.sqrt(sum_sq))
    rss_mean = float(network / np.sqrt(max(len(per), 1)))
    return NetworkSNRResult(per_detector=per, network_snr=network, rss_mean_snr=rss_mean)
