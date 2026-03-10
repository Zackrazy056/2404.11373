import numpy as np

from rd_sbi.eval.snr import compute_detector_snr, compute_network_snr


def test_detector_snr_scales_linearly_with_strain_amplitude() -> None:
    fs = 2048.0
    n = 204
    t = np.arange(n) / fs
    h = 1e-21 * np.sin(2 * np.pi * 250.0 * t)
    f_psd = np.linspace(0.0, fs / 2.0, 4097)
    psd = np.full_like(f_psd, 1e-40)

    rho1 = compute_detector_snr(h, fs, f_psd, psd)
    rho2 = compute_detector_snr(2.0 * h, fs, f_psd, psd)
    assert np.isclose(rho2 / rho1, 2.0, rtol=1e-6)


def test_network_snr_combines_detectors_by_rss() -> None:
    fs = 2048.0
    n = 204
    t = np.arange(n) / fs
    h1 = 1e-21 * np.sin(2 * np.pi * 250.0 * t)
    h2 = 0.5e-21 * np.sin(2 * np.pi * 250.0 * t)
    f_psd = np.linspace(0.0, fs / 2.0, 4097)
    psd = np.full_like(f_psd, 1e-40)

    result = compute_network_snr(
        strains={"H1": h1, "L1": h2},
        sample_rate_hz=fs,
        psd_by_detector={"H1": (f_psd, psd), "L1": (f_psd, psd)},
    )
    rho_h1 = compute_detector_snr(h1, fs, f_psd, psd)
    rho_l1 = compute_detector_snr(h2, fs, f_psd, psd)
    expected = np.sqrt(rho_h1**2 + rho_l1**2)
    assert np.isclose(result.network_snr, expected, rtol=1e-10)
