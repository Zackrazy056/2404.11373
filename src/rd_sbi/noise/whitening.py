"""Time-domain whitening based on PSD -> ACF -> Toeplitz -> Cholesky."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular, toeplitz


@dataclass(frozen=True)
class WhiteningResult:
    """Container for whitening outputs."""

    whitened: np.ndarray
    acf: np.ndarray
    covariance: np.ndarray
    cholesky_l: np.ndarray


def _one_sided_to_two_sided_density(psd_one_sided: np.ndarray) -> np.ndarray:
    """Convert one-sided PSD bins to the positive branch of a two-sided PSD.

    Welch-style one-sided PSD doubles interior frequencies. We undo that before
    inverse FFT to obtain a valid autocovariance sequence.
    """
    psd = np.asarray(psd_one_sided, dtype=np.float64)
    if psd.ndim != 1 or psd.shape[0] < 2:
        raise ValueError("psd_one_sided must be a 1D array with at least 2 bins")

    two_sided_pos = psd.copy()
    if two_sided_pos.shape[0] > 2:
        two_sided_pos[1:-1] *= 0.5
    return two_sided_pos


def acf_from_one_sided_psd(psd_one_sided: np.ndarray, sample_rate_hz: float, n_lags: int) -> np.ndarray:
    """Compute autocovariance from one-sided PSD via inverse real FFT.

    Parameters
    ----------
    psd_one_sided:
        One-sided PSD values from `scipy.signal.welch(..., return_onesided=True)`.
    sample_rate_hz:
        Sampling frequency.
    n_lags:
        Number of ACF lags to keep (starting at lag 0).
    """
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if n_lags <= 0:
        raise ValueError("n_lags must be positive")

    two_sided_pos = _one_sided_to_two_sided_density(psd_one_sided)
    n_fft_even = 2 * (two_sided_pos.shape[0] - 1)
    n_fft_odd = n_fft_even + 1
    if n_lags <= n_fft_even:
        n_fft = n_fft_even
    elif n_lags == n_fft_odd:
        # For odd-length time-domain segments, rFFT still has floor(n/2)+1 bins.
        # Recover the intended inverse-transform length from the requested lag count.
        n_fft = n_fft_odd
    else:
        raise ValueError(f"n_lags={n_lags} cannot exceed supported n_fft in {{{n_fft_even}, {n_fft_odd}}}")

    acf_full = np.fft.irfft(two_sided_pos, n=n_fft) * float(sample_rate_hz)
    return np.asarray(acf_full[:n_lags], dtype=np.float64)


def covariance_from_acf(acf: np.ndarray, n_samples: int) -> np.ndarray:
    """Build Toeplitz covariance matrix from ACF."""
    acf = np.asarray(acf, dtype=np.float64)
    if acf.ndim != 1:
        raise ValueError("acf must be 1D")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if acf.shape[0] < n_samples:
        raise ValueError("acf length must be >= n_samples")

    first_col = acf[:n_samples]
    return np.asarray(toeplitz(first_col), dtype=np.float64)


def covariance_from_one_sided_psd(psd_one_sided: np.ndarray, sample_rate_hz: float, n_samples: int) -> np.ndarray:
    """Convenience wrapper: PSD -> ACF -> Toeplitz covariance."""
    acf = acf_from_one_sided_psd(psd_one_sided, sample_rate_hz, n_samples)
    return covariance_from_acf(acf, n_samples)


def cholesky_lower_with_jitter(
    covariance: np.ndarray,
    *,
    relative_jitter: float = 1e-12,
    max_tries: int = 8,
) -> np.ndarray:
    """Compute stable lower-triangular Cholesky factor."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")

    diag_scale = float(np.max(np.abs(np.diag(cov))))
    if not np.isfinite(diag_scale) or diag_scale <= 0.0:
        diag_scale = float(np.finfo(np.float64).eps)

    jitter = float(relative_jitter) * diag_scale
    eye = np.eye(cov.shape[0], dtype=np.float64)
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(cov + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("Cholesky decomposition failed even after jitter escalation")


def whiten_strain_from_covariance(strain: np.ndarray, covariance: np.ndarray) -> WhiteningResult:
    """Whiten strain using Eq. (A1): h_white = L^{-1} h."""
    x = np.asarray(strain, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"strain must be 1D, got shape={x.shape}")

    n = x.shape[0]
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (n, n):
        raise ValueError(f"covariance shape {cov.shape} must match (n,n)=({n},{n})")

    l_factor = cholesky_lower_with_jitter(cov)
    whitened = solve_triangular(l_factor, x, lower=True, check_finite=False)

    return WhiteningResult(
        whitened=np.asarray(whitened, dtype=np.float64),
        acf=np.asarray(cov[0], dtype=np.float64),
        covariance=cov,
        cholesky_l=l_factor,
    )


def whiten_strain_from_psd(strain: np.ndarray, psd_one_sided: np.ndarray, sample_rate_hz: float) -> WhiteningResult:
    """Whiten strain from one-sided PSD following Eq. (A1)."""
    x = np.asarray(strain, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"strain must be 1D, got shape={x.shape}")

    acf = acf_from_one_sided_psd(psd_one_sided, sample_rate_hz, n_lags=x.shape[0])
    covariance = covariance_from_acf(acf, n_samples=x.shape[0])
    result = whiten_strain_from_covariance(x, covariance)

    return WhiteningResult(
        whitened=result.whitened,
        acf=acf,
        covariance=covariance,
        cholesky_l=result.cholesky_l,
    )
