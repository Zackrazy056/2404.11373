"""Time-domain black-hole ringdown waveform utilities.

Implements paper equations:
- Eq. (1): detector projection h = F_plus h_plus + F_cross h_cross
- Eq. (2a, 2b): damped-sinusoid mode superposition in time domain
- Eq. (3): phase model Phi_lmn
- Eq. (5): Y_plus / Y_cross from spin-weighted spherical harmonics
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class QNMBias:
    """Optional fractional deviations from GR QNM frequency components.

    The GR complex QNM is:
        omega_tilde = 2*pi*f + i/tau

    With bias parameters:
        Re(omega_tilde) -> Re(omega_tilde) * (1 + alpha_r)
        Im(omega_tilde) -> Im(omega_tilde) * (1 + alpha_i)
    """

    alpha_r: float = 0.0
    alpha_i: float = 0.0


@dataclass(frozen=True)
class RingdownMode:
    """Single excited QNM mode."""

    l: int
    m: int
    n: int
    amplitude: float
    phase: float
    frequency_hz: float
    damping_time_s: float
    bias: QNMBias | None = None


@dataclass(frozen=True)
class RingdownPolarizations:
    """Plus and cross polarizations for one source."""

    time_s: np.ndarray
    h_plus: np.ndarray
    h_cross: np.ndarray


def build_time_array(
    sample_rate_hz: float = 2048.0,
    duration_s: float = 0.1,
    start_time_s: float = 0.0,
) -> np.ndarray:
    """Build the ringdown time grid.

    Paper setup uses:
    - sample_rate_hz = 2048
    - duration_s = 0.1
    which yields int(2048*0.1)=204 bins.
    """
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")

    n_bins = int(sample_rate_hz * duration_s)
    if n_bins <= 0:
        raise ValueError("empty time grid, adjust sample_rate_hz or duration_s")

    return start_time_s + np.arange(n_bins, dtype=np.float64) / float(sample_rate_hz)


def qnm_complex_frequency(frequency_hz: float, damping_time_s: float, bias: QNMBias | None = None) -> complex:
    """Return complex QNM frequency omega_tilde = 2*pi*f + i/tau with optional alpha bias."""
    if frequency_hz <= 0.0:
        raise ValueError("frequency_hz must be positive")
    if damping_time_s <= 0.0:
        raise ValueError("damping_time_s must be positive")

    real_part = 2.0 * np.pi * float(frequency_hz)
    imag_part = 1.0 / float(damping_time_s)

    if bias is not None:
        real_part *= 1.0 + float(bias.alpha_r)
        imag_part *= 1.0 + float(bias.alpha_i)

    return complex(real_part, imag_part)


def _wigner_small_d(l: int, m: int, mp: int, theta: float) -> float:
    """Compute Wigner small-d matrix element d^l_{m,mp}(theta)."""
    if l < 0:
        raise ValueError("l must be >= 0")
    if abs(m) > l or abs(mp) > l:
        return 0.0

    prefactor = np.sqrt(
        factorial(l + m) * factorial(l - m) * factorial(l + mp) * factorial(l - mp)
    )
    cos_half = np.cos(0.5 * theta)
    sin_half = np.sin(0.5 * theta)

    k_min = max(0, m - mp)
    k_max = min(l + m, l - mp)

    total = 0.0
    for k in range(k_min, k_max + 1):
        denom = (
            factorial(l + m - k)
            * factorial(k)
            * factorial(mp - m + k)
            * factorial(l - mp - k)
        )
        sign = -1.0 if ((k - m + mp) % 2) else 1.0
        pow_cos = 2 * l + m - mp - 2 * k
        pow_sin = mp - m + 2 * k
        total += sign * prefactor / denom * (cos_half ** pow_cos) * (sin_half ** pow_sin)
    return float(total)


def spin_weighted_spherical_harmonic(s: int, l: int, m: int, theta: float, phi: float = 0.0) -> complex:
    """Compute spin-weighted spherical harmonic {}_sY_{lm}(theta, phi)."""
    if l < 0:
        raise ValueError("l must be >= 0")
    if abs(s) > l:
        raise ValueError("require |s| <= l")
    if abs(m) > l:
        raise ValueError("require |m| <= l")

    normalization = np.sqrt((2.0 * l + 1.0) / (4.0 * np.pi))
    d_elem = _wigner_small_d(l=l, m=m, mp=-s, theta=theta)
    phase = np.exp(1j * m * phi)
    return ((-1) ** s) * normalization * d_elem * phase


def y_plus_y_cross(l: int, m: int, inclination_rad: float) -> tuple[float, float]:
    """Eq. (5) approximation using spin-weighted spherical harmonics."""
    y_lm = spin_weighted_spherical_harmonic(-2, l, m, inclination_rad, phi=0.0)
    y_l_minus_m = spin_weighted_spherical_harmonic(-2, l, -m, inclination_rad, phi=0.0)

    parity = (-1) ** l
    y_plus = y_lm + parity * y_l_minus_m
    y_cross = y_lm - parity * y_l_minus_m

    # For phi=0 this should be real in the adopted convention.
    return float(np.real_if_close(y_plus)), float(np.real_if_close(y_cross))


def generate_ringdown_polarizations(
    time_s: np.ndarray,
    modes: Sequence[RingdownMode],
    inclination_rad: float,
    t_start_s: float = 0.0,
) -> RingdownPolarizations:
    """Generate h_plus and h_cross from multi-mode damped sinusoids."""
    t = np.asarray(time_s, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"time_s must be 1D, got shape={t.shape}")
    if len(modes) == 0:
        raise ValueError("modes cannot be empty")

    dt = t - float(t_start_s)
    gate = (dt >= 0.0).astype(np.float64)

    h_plus = np.zeros_like(t, dtype=np.float64)
    h_cross = np.zeros_like(t, dtype=np.float64)

    for mode in modes:
        omega = qnm_complex_frequency(
            frequency_hz=mode.frequency_hz,
            damping_time_s=mode.damping_time_s,
            bias=mode.bias,
        )
        omega_r = float(np.real(omega))
        omega_i = float(np.imag(omega))

        phase_t = omega_r * dt + float(mode.phase)
        envelope = np.exp(-omega_i * dt) * gate
        y_plus, y_cross = y_plus_y_cross(mode.l, mode.m, inclination_rad)

        h_plus += float(mode.amplitude) * envelope * np.cos(phase_t) * y_plus
        h_cross += float(mode.amplitude) * envelope * np.sin(phase_t) * y_cross

    return RingdownPolarizations(time_s=t, h_plus=h_plus, h_cross=h_cross)
