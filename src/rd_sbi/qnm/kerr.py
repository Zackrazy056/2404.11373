"""Kerr QNM mapping: (Mf, chi_f) -> (f_lmn, tau_lmn)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

MTSUN_SI = 4.925490947e-6  # G * M_sun / c^3 in seconds


@dataclass(frozen=True)
class KerrMode:
    """QNM mode label."""

    l: int
    m: int
    n: int


@dataclass(frozen=True)
class KerrQNMPhysical:
    """Physical-frequency representation of one Kerr QNM."""

    mode: KerrMode
    mass_msun: float
    chi_f: float
    omega_dimensionless: complex
    frequency_hz: float
    damping_time_s: float


def mass_seconds_from_msun(mass_msun: float) -> float:
    """Convert detector-frame mass in solar masses to geometric time (seconds)."""
    if mass_msun <= 0.0:
        raise ValueError("mass_msun must be positive")
    return float(mass_msun) * MTSUN_SI


def _validate_spin(chi_f: float) -> float:
    chi = float(chi_f)
    if not (0.0 <= chi < 1.0):
        raise ValueError("chi_f must satisfy 0 <= chi_f < 1")
    return chi


def _omega_from_qnm_package(mode: KerrMode, chi_f: float, spin_weight: int = -2) -> complex:
    """Get dimensionless complex frequency using the `qnm` package."""
    import qnm

    seq = _qnm_mode_sequence(spin_weight=spin_weight, l=mode.l, m=mode.m, n=mode.n)
    omega = complex(seq(a=chi_f)[0])
    if np.imag(omega) >= 0.0:
        # enforce damping sign convention exp(-|Im(omega)| t / M)
        omega = complex(np.real(omega), -abs(np.imag(omega)))
    return omega


@lru_cache(maxsize=64)
def _qnm_mode_sequence(spin_weight: int, l: int, m: int, n: int):
    """Cached `qnm.modes_cache` sequence object for repeated evaluations."""
    import qnm

    return qnm.modes_cache(s=spin_weight, l=l, m=m, n=n)


_BERTI_FIT_COEFFS: dict[tuple[int, int, int], tuple[float, float, float, float, float, float]] = {
    # (l,m,n): (f1,f2,f3,q1,q2,q3)
    # Phenomenological fits from Berti et al. [45], Tables VIII-IX.
    (2, 2, 0): (1.5251, -1.1568, 0.1292, 0.7000, 1.4187, -0.4990),
    (2, 2, 1): (1.3673, -1.0260, 0.1628, 0.1000, 0.5436, -0.4731),
    (3, 3, 0): (1.8956, -1.3043, 0.1818, 0.9000, 2.3430, -0.4810),
}


def _omega_from_berti_fit(mode: KerrMode, chi_f: float) -> complex:
    key = (mode.l, mode.m, mode.n)
    if key not in _BERTI_FIT_COEFFS:
        raise ValueError(f"No fallback fit coefficients for mode={key}")
    f1, f2, f3, q1, q2, q3 = _BERTI_FIT_COEFFS[key]
    x = 1.0 - chi_f
    omega_r = f1 + f2 * (x**f3)
    quality = q1 + q2 * (x**q3)
    omega_i = omega_r / (2.0 * quality)
    return complex(omega_r, -abs(omega_i))


def kerr_qnm_dimensionless_omega(
    l: int,
    m: int,
    n: int,
    chi_f: float,
    *,
    method: str = "auto",
    spin_weight: int = -2,
) -> complex:
    """Return dimensionless Kerr QNM omega*M."""
    mode = KerrMode(int(l), int(m), int(n))
    chi = _validate_spin(chi_f)

    if method not in {"auto", "qnm", "fit"}:
        raise ValueError("method must be one of {'auto','qnm','fit'}")

    if method in {"auto", "qnm"}:
        try:
            return _omega_from_qnm_package(mode, chi, spin_weight=spin_weight)
        except Exception:  # noqa: BLE001
            if method == "qnm":
                raise

    return _omega_from_berti_fit(mode, chi)


def kerr_qnm_physical(
    l: int,
    m: int,
    n: int,
    mass_msun: float,
    chi_f: float,
    *,
    alpha_r: float = 0.0,
    alpha_i: float = 0.0,
    method: str = "auto",
    spin_weight: int = -2,
) -> KerrQNMPhysical:
    """Map remnant mass and spin to physical QNM frequency and damping time."""
    mode = KerrMode(int(l), int(m), int(n))
    m_sec = mass_seconds_from_msun(mass_msun)
    omega = kerr_qnm_dimensionless_omega(
        mode.l,
        mode.m,
        mode.n,
        chi_f=chi_f,
        method=method,
        spin_weight=spin_weight,
    )

    omega_r = float(np.real(omega)) * (1.0 + float(alpha_r))
    omega_i_abs = abs(float(np.imag(omega))) * (1.0 + float(alpha_i))
    if omega_r <= 0.0:
        raise ValueError("biased real(omega) must stay positive")
    if omega_i_abs <= 0.0:
        raise ValueError("biased imag(omega) magnitude must stay positive")

    frequency_hz = omega_r / (2.0 * np.pi * m_sec)
    damping_time_s = m_sec / omega_i_abs

    return KerrQNMPhysical(
        mode=mode,
        mass_msun=float(mass_msun),
        chi_f=float(chi_f),
        omega_dimensionless=complex(omega_r, -omega_i_abs),
        frequency_hz=float(frequency_hz),
        damping_time_s=float(damping_time_s),
    )


def map_modes_to_qnms(
    modes: Iterable[KerrMode | tuple[int, int, int]],
    mass_msun: float,
    chi_f: float,
    *,
    alpha_r: float = 0.0,
    alpha_i: float = 0.0,
    method: str = "auto",
    spin_weight: int = -2,
) -> list[KerrQNMPhysical]:
    """Batch map modes to physical QNMs."""
    out: list[KerrQNMPhysical] = []
    for mode_in in modes:
        mode = mode_in if isinstance(mode_in, KerrMode) else KerrMode(*mode_in)
        out.append(
            kerr_qnm_physical(
                mode.l,
                mode.m,
                mode.n,
                mass_msun=mass_msun,
                chi_f=chi_f,
                alpha_r=alpha_r,
                alpha_i=alpha_i,
                method=method,
                spin_weight=spin_weight,
            )
        )
    return out
