import numpy as np

from rd_sbi.qnm.kerr import kerr_qnm_physical, map_modes_to_qnms


def test_kerr_qnm_220_physical_range() -> None:
    q = kerr_qnm_physical(l=2, m=2, n=0, mass_msun=67.0, chi_f=0.67)
    assert 200.0 < q.frequency_hz < 350.0
    assert 0.002 < q.damping_time_s < 0.01


def test_kerr_221_damps_faster_than_220() -> None:
    q0 = kerr_qnm_physical(l=2, m=2, n=0, mass_msun=67.0, chi_f=0.67)
    q1 = kerr_qnm_physical(l=2, m=2, n=1, mass_msun=67.0, chi_f=0.67)
    assert q1.damping_time_s < q0.damping_time_s


def test_map_modes_batch() -> None:
    out = map_modes_to_qnms([(2, 2, 0), (3, 3, 0)], mass_msun=67.0, chi_f=0.67)
    assert len(out) == 2


def test_fit_method_supports_all_fig1_modes() -> None:
    for mode in [(2, 2, 0), (2, 2, 1), (3, 3, 0)]:
        q = kerr_qnm_physical(*mode, mass_msun=67.0, chi_f=0.67, method="fit")
        assert q.frequency_hz > 0.0
        assert q.damping_time_s > 0.0


def test_fit_method_stays_close_to_qnm_backend_for_fig1_modes() -> None:
    for mode in [(2, 2, 0), (2, 2, 1), (3, 3, 0)]:
        q_fit = kerr_qnm_physical(*mode, mass_msun=67.0, chi_f=0.67, method="fit")
        q_ref = kerr_qnm_physical(*mode, mass_msun=67.0, chi_f=0.67, method="qnm")
        assert np.isclose(q_fit.frequency_hz, q_ref.frequency_hz, rtol=0.05)
        assert np.isclose(q_fit.damping_time_s, q_ref.damping_time_s, rtol=0.10)
