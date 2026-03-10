from __future__ import annotations

import numpy as np

from rd_sbi.noise.whitening import acf_from_one_sided_psd, covariance_from_one_sided_psd


def test_acf_from_one_sided_psd_supports_odd_length_segments() -> None:
    psd = np.ones(205, dtype=np.float64)
    acf = acf_from_one_sided_psd(psd, sample_rate_hz=4096.0, n_lags=409)
    assert acf.shape == (409,)
    assert np.isfinite(acf).all()


def test_covariance_from_one_sided_psd_supports_odd_length_segments() -> None:
    psd = np.ones(410, dtype=np.float64)
    covariance = covariance_from_one_sided_psd(psd, sample_rate_hz=4096.0, n_samples=819)
    assert covariance.shape == (819, 819)
    assert np.isfinite(covariance).all()
