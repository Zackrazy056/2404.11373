import numpy as np

from rd_sbi.detector.patterns import (
    _source_basis,
    antenna_pattern,
    detector_strain,
    gmst_from_gps,
    h1_geometry,
    l1_geometry,
    time_delay_from_geocenter_s,
)
from rd_sbi.simulator.injection import build_detector_timing_context
from rd_sbi.waveforms.ringdown import (
    QNMBias,
    RingdownMode,
    build_time_array,
    generate_ringdown_polarizations,
    qnm_complex_frequency,
)


def test_detector_geometry_and_tensor_sanity() -> None:
    for geometry in (h1_geometry(), l1_geometry()):
        assert np.isclose(np.linalg.norm(geometry.x_arm), 1.0)
        assert np.isclose(np.linalg.norm(geometry.y_arm), 1.0)
        assert abs(np.dot(geometry.x_arm, geometry.y_arm)) < 1e-5
        assert 6.0e6 < np.linalg.norm(geometry.location_m) < 6.5e6

        tensor = geometry.detector_tensor()
        assert np.allclose(tensor, tensor.T)
        assert abs(np.trace(tensor)) < 1e-12


def test_source_basis_is_orthonormal_even_near_poles() -> None:
    gmst = gmst_from_gps(1126259462.42323)
    for dec in (-0.5 * np.pi + 1e-8, -1.27, 0.0, 1.1, 0.5 * np.pi - 1e-8):
        p_vec, q_vec = _source_basis(ra_rad=1.95, dec_rad=dec, psi_rad=0.82, gmst_rad=gmst)
        assert np.isclose(np.linalg.norm(p_vec), 1.0)
        assert np.isclose(np.linalg.norm(q_vec), 1.0)
        assert abs(np.dot(p_vec, q_vec)) < 1e-12


def test_build_time_array_matches_paper_bins() -> None:
    t = build_time_array(sample_rate_hz=2048.0, duration_s=0.1)
    assert t.shape == (204,)
    assert np.isclose(t[1] - t[0], 1.0 / 2048.0)


def test_qnm_complex_frequency_supports_alpha_bias() -> None:
    base = qnm_complex_frequency(250.0, 0.004)
    biased = qnm_complex_frequency(250.0, 0.004, bias=QNMBias(alpha_r=0.1, alpha_i=-0.1))
    assert np.isclose(np.real(biased), 1.1 * np.real(base))
    assert np.isclose(np.imag(biased), 0.9 * np.imag(base))


def test_multimode_superposition_is_linear() -> None:
    t = build_time_array()
    mode_a = RingdownMode(l=2, m=2, n=0, amplitude=1e-21, phase=0.1, frequency_hz=250.0, damping_time_s=0.004)
    mode_b = RingdownMode(l=3, m=3, n=0, amplitude=0.3e-21, phase=1.2, frequency_hz=360.0, damping_time_s=0.003)
    inc = np.pi / 3

    ab = generate_ringdown_polarizations(t, [mode_a, mode_b], inc)
    only_a = generate_ringdown_polarizations(t, [mode_a], inc)
    only_b = generate_ringdown_polarizations(t, [mode_b], inc)

    assert np.allclose(ab.h_plus, only_a.h_plus + only_b.h_plus)
    assert np.allclose(ab.h_cross, only_a.h_cross + only_b.h_cross)


def test_h1_l1_antenna_and_projection() -> None:
    gmst = gmst_from_gps(1126259462.42323)
    ra = 1.95
    dec = -1.27
    psi = 0.82

    fph, fch = antenna_pattern(h1_geometry(), ra, dec, psi, gmst)
    fpl, fcl = antenna_pattern(l1_geometry(), ra, dec, psi, gmst)

    assert np.isfinite([fph, fch, fpl, fcl]).all()
    assert abs(fph) <= 1.0 and abs(fch) <= 1.0
    assert abs(fpl) <= 1.0 and abs(fcl) <= 1.0

    hp = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    hx = np.array([0.5, 0.5, 0.5], dtype=np.float64)
    proj = detector_strain(hp, hx, fph, fch)
    assert np.allclose(proj, fph * hp + fch * hx)


def test_h1_l1_geocenter_delay_difference_is_reasonable() -> None:
    gmst = gmst_from_gps(1126259462.42323)
    ra = 1.95
    dec = -1.27
    dt_h1 = time_delay_from_geocenter_s(h1_geometry(), ra, dec, gmst)
    dt_l1 = time_delay_from_geocenter_s(l1_geometry(), ra, dec, gmst)
    delta = dt_l1 - dt_h1
    assert abs(delta) < 0.02


def test_detector_time_delay_sign_chain_is_self_consistent() -> None:
    gmst = gmst_from_gps(1126259462.42323)
    ra = 1.95
    dec = -1.27
    dt_h1 = time_delay_from_geocenter_s(h1_geometry(), ra, dec, gmst)
    dt_l1 = time_delay_from_geocenter_s(l1_geometry(), ra, dec, gmst)
    delta = dt_l1 - dt_h1

    time_s = build_time_array(sample_rate_hz=2048.0, duration_s=0.1, start_time_s=0.0)
    mode = RingdownMode(l=2, m=2, n=0, amplitude=1e-21, phase=0.0, frequency_hz=250.0, damping_time_s=0.01)
    t_ref = 0.02
    hp_h1 = generate_ringdown_polarizations(time_s, [mode], inclination_rad=np.pi / 3, t_start_s=t_ref).h_plus
    hp_l1 = generate_ringdown_polarizations(time_s, [mode], inclination_rad=np.pi / 3, t_start_s=t_ref + delta).h_plus

    i_h1 = int(np.flatnonzero(np.abs(hp_h1) > 0.0)[0])
    i_l1 = int(np.flatnonzero(np.abs(hp_l1) > 0.0)[0])
    observed_delta = time_s[i_l1] - time_s[i_h1]
    sample_dt = time_s[1] - time_s[0]
    assert abs(observed_delta - delta) <= sample_dt


def test_window_anchor_preserves_earliest_detector_arrival() -> None:
    gmst = gmst_from_gps(1126259462.42323)
    timing = build_detector_timing_context(
        detectors=["H1", "L1"],
        use_detector_time_delay=True,
        reference_detector="H1",
        t_start_s=0.0,
        ra_rad=1.95,
        dec_rad=-1.27,
        gmst_rad=gmst,
    )

    assert timing["window_anchor_strategy"] == "preserve_earliest_arrival_in_common_window"
    assert timing["earliest_detector"] == "L1"
    assert timing["relative_delay_to_reference_s"]["L1"] < 0.0
    assert np.isclose(timing["t_start_detector_s"]["L1"], 0.0)
    assert timing["t_start_detector_s"]["H1"] > timing["t_start_detector_s"]["L1"]
    assert np.isclose(
        timing["t_start_detector_s"]["H1"] - timing["t_start_detector_s"]["L1"],
        -timing["relative_delay_to_reference_s"]["L1"],
    )
