"""Detector antenna pattern and projection utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorGeometry:
    """Interferometer geometry in Earth-fixed Cartesian coordinates."""

    name: str
    x_arm: np.ndarray
    y_arm: np.ndarray
    location_m: np.ndarray

    def detector_tensor(self) -> np.ndarray:
        x = np.asarray(self.x_arm, dtype=np.float64)
        y = np.asarray(self.y_arm, dtype=np.float64)
        return 0.5 * (np.outer(x, x) - np.outer(y, y))


def _normalize(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("cannot normalize zero-length vector")
    return v / norm


def h1_geometry() -> DetectorGeometry:
    """LIGO Hanford (H1) arm unit vectors in ECEF coordinates."""
    return DetectorGeometry(
        name="H1",
        x_arm=_normalize(np.array([-0.22389266154, 0.79983062746, 0.55690487831], dtype=np.float64)),
        y_arm=_normalize(np.array([-0.91397818574, 0.02609403989, -0.40492342125], dtype=np.float64)),
        location_m=np.array([-2161414.92636, -3834695.17889, 4600350.22664], dtype=np.float64),
    )


def l1_geometry() -> DetectorGeometry:
    """LIGO Livingston (L1) arm unit vectors in ECEF coordinates."""
    return DetectorGeometry(
        name="L1",
        x_arm=_normalize(np.array([-0.95457412153, -0.14158077340, -0.26218911324], dtype=np.float64)),
        y_arm=_normalize(np.array([0.29774156894, -0.48791033647, -0.82054461286], dtype=np.float64)),
        location_m=np.array([-74276.0447238, -5496283.71971, 3224257.01744], dtype=np.float64),
    )


def gmst_from_gps(gps_time_s: float, leap_seconds: int = 18) -> float:
    """Approximate Greenwich mean sidereal time in radians.

    The default GPS-UTC offset is 18 seconds for the relevant O1/O2/O3 epochs.
    """
    unix_time_s = float(gps_time_s) + 315964800.0 - float(leap_seconds)
    jd = unix_time_s / 86400.0 + 2440587.5
    t_ut1 = (jd - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t_ut1
        + 0.093104 * (t_ut1**2)
        - 6.2e-6 * (t_ut1**3)
    )
    gmst_wrapped = np.mod(gmst_sec, 86400.0)
    return 2.0 * np.pi * (gmst_wrapped / 86400.0)


def _source_basis(ra_rad: float, dec_rad: float, psi_rad: float, gmst_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Construct polarization basis vectors p and q in ECEF frame."""
    gha = float(gmst_rad) - float(ra_rad)
    sin_g, cos_g = np.sin(gha), np.cos(gha)
    sin_d, cos_d = np.sin(dec_rad), np.cos(dec_rad)

    # Tangential basis aligned with the detector-response convention used by
    # standard GW toolkits. Keep the propagation/time-delay convention
    # unchanged; only the polarization basis lives here.
    e_east = np.array([-sin_g, -cos_g, 0.0], dtype=np.float64)
    e_north = np.array([-sin_d * cos_g, sin_d * sin_g, cos_d], dtype=np.float64)

    e_east = _normalize(e_east)
    e_north = _normalize(e_north)

    cos_p, sin_p = np.cos(psi_rad), np.sin(psi_rad)
    p_vec = cos_p * e_east + sin_p * e_north
    q_vec = -sin_p * e_east + cos_p * e_north
    return _normalize(p_vec), _normalize(q_vec)


def source_unit_vector_ecef(ra_rad: float, dec_rad: float, gmst_rad: float) -> np.ndarray:
    """Unit vector from Earth center toward source in ECEF frame."""
    gha = float(gmst_rad) - float(ra_rad)
    cos_d, sin_d = np.cos(dec_rad), np.sin(dec_rad)
    cos_g, sin_g = np.cos(gha), np.sin(gha)
    vec = np.array([cos_d * cos_g, -cos_d * sin_g, sin_d], dtype=np.float64)
    return _normalize(vec)


def time_delay_from_geocenter_s(detector: DetectorGeometry, ra_rad: float, dec_rad: float, gmst_rad: float) -> float:
    """Arrival time delay relative to geocenter for a plane wave."""
    c_m_s = 299792458.0
    n_hat = source_unit_vector_ecef(ra_rad=ra_rad, dec_rad=dec_rad, gmst_rad=gmst_rad)
    r_det = np.asarray(detector.location_m, dtype=np.float64)
    return float(-np.dot(n_hat, r_det) / c_m_s)


def antenna_pattern(detector: DetectorGeometry, ra_rad: float, dec_rad: float, psi_rad: float, gmst_rad: float) -> tuple[float, float]:
    """Compute (F_plus, F_cross) for one detector."""
    p_vec, q_vec = _source_basis(ra_rad=ra_rad, dec_rad=dec_rad, psi_rad=psi_rad, gmst_rad=gmst_rad)
    e_plus = np.outer(p_vec, p_vec) - np.outer(q_vec, q_vec)
    e_cross = np.outer(p_vec, q_vec) + np.outer(q_vec, p_vec)

    d = detector.detector_tensor()
    f_plus = float(np.tensordot(d, e_plus, axes=2))
    f_cross = float(np.tensordot(d, e_cross, axes=2))
    return f_plus, f_cross


def detector_strain(h_plus: np.ndarray, h_cross: np.ndarray, f_plus: float, f_cross: float) -> np.ndarray:
    """Eq. (1) projection to detector strain."""
    hp = np.asarray(h_plus, dtype=np.float64)
    hx = np.asarray(h_cross, dtype=np.float64)
    if hp.shape != hx.shape:
        raise ValueError(f"h_plus shape {hp.shape} must match h_cross shape {hx.shape}")
    return float(f_plus) * hp + float(f_cross) * hx
