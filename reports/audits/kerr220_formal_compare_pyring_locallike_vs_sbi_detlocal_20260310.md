# kerr220 formal compare: locallike pyRing vs detlocal SBI

Date: 2026-03-10

## Inputs

- pyRing baseline: `reports/posteriors/pyring/kerr220_20260310-locallike-main`
- pyRing stable export: `reports/posteriors/pyring/kerr220_pyring.npz`
- pyRing manifest: `reports/posteriors/pyring/manifest_pyring.json`
- SBI run: `reports/posteriors/fig1_paper_precision/kerr220_20260310-fig1220-detlocal`
- SBI posterior: `reports/posteriors/fig1_paper_precision/kerr220_20260310-fig1220-detlocal/kerr220_sbi_posterior_20000.npz`
- SBI summary: `reports/posteriors/fig1_paper_precision/kerr220_20260310-fig1220-detlocal/kerr220_run_summary.json`

## Output artifacts

- Overlay figure: `reports/figures/fig1_overlay_kerr220_locallike_main_vs_sbi_20260310-fig1220-detlocal.png`
- Overlay summary: `reports/posteriors/fig1_overlay_kerr220_locallike_main_vs_sbi_20260310-fig1220-detlocal.json`

## Quantitative agreement

- `wasserstein_Mf = 1.0390`
- `wasserstein_chi = 0.02426`
- `ks_Mf_D = 0.06965`
- `ks_chi_D = 0.06630`
- `area_ratio_90_sbi_over_pyring = 0.9980`

Interpretation:

- The 90% credible-region areas are effectively matched.
- The 2D contours overlap strongly across the full banana-shaped posterior.
- This is materially better than the previous comparison against the older `20260308-111922` SBI run.

## Posterior location check

True parameters:

- `Mf_true = 67.0`
- `chi_true = 0.67`

SBI posterior:

- `Mf = 63.218 [56.907, 69.069]`
- `chi = 0.5850 [0.3960, 0.7109]`

pyRing posterior:

- `Mf = 64.172 [57.831, 70.140]`
- `chi = 0.6091 [0.4212, 0.7256]`

Residual offset:

- pyRing remains slightly higher in both `Mf` and `chi`, but the medians differ only by about `0.95 Msun` and `0.024`.
- The overlap width is now comparable, unlike the earlier `paperlike pyRing` comparison.

## Comparison to earlier overlays

Previous overlay against old SBI (`locallike pyRing` vs `kerr220_20260308-111922`):

- `wasserstein_Mf = 1.6027`
- `wasserstein_chi = 0.05243`
- `area_ratio_90_sbi_over_pyring = 1.3447`

Earlier overlay against old SBI with `paperlike pyRing`:

- `wasserstein_Mf = 4.2758`
- `wasserstein_chi = 0.1273`
- `area_ratio_90_sbi_over_pyring = 4.1773`

Interpretation:

- Switching the pyRing baseline from `4096/0.2/bandpass` to `2048/0.1/no-bandpass` removed the dominant mismatch.
- Updating the SBI run to the `detector_local_truncation` path improved the posterior overlap further.

## Remaining mismatch sources

- SBI run summary reports `measured_network_snr = 15.7209`, while the current locallike pyRing baseline is `network optimal TD SNR = 14.0885`.
- The posterior agreement is now good enough that the remaining gap looks second-order and likely comes from:
  - differing whitening / likelihood implementation (`pyRing` TD covariance vs SBI PSD->ACF->Toeplitz->Cholesky path)
  - TSNPE truncation relaxation and simulation-budget fallback in rounds 2-4
  - finite-sample and sampler-shape differences

## Current recommendation

For `kerr220`, the current formal comparison baseline should be:

- pyRing: `locallike-main`
- SBI: `kerr220_20260310-fig1220-detlocal`

This is the first pair in the repo where the mechanism-level alignment is strong enough that overlay disagreement is no longer the primary blocker.
