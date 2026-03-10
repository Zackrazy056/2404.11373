# Engineering Status (2026-03-05)

## Overall

Project has reached a usable mid-stage:
- Core simulator, detector projection, QNM mapping, PSD/whitening, TSNPE loop are implemented.
- Batch injection cache and SNR validation are operational.
- A start-version Fig.1 pipeline exists and runs end-to-end.

Not yet paper-complete:
- No pyRing baseline posterior integration for Fig.1 overlays.
- TSNPE training scale is currently reduced (quick settings).
- Coverage test pipeline (Appendix B style production run) is not completed yet.

## Completed blocks

- `M0` bootstrap:
  - environment pinning (`pyproject.toml`)
  - config + seed management
  - artifact/log policy and checks
- Data/noise:
  - GWOSC fetch script
  - Welch PSD estimation
  - time-domain whitening path (PSD -> ACF -> Toeplitz -> Cholesky)
- Physics/model:
  - ringdown waveform superposition
  - spin-weighted spherical-harmonic approximation
  - H1/L1 antenna projection and detector time-delay support
  - Kerr QNM mapping `(M_f, chi_f) -> (f_lmn, tau_lmn)`
- Injection pipeline:
  - Kerr220/221/330 configs
  - single generation + batch cache with train/val index
  - SNR validation report generation
- Inference:
  - embedding FC network + NSF builder
  - TSNPE truncation + stopping-ratio loop
  - SNPE loss wrapper with dynamic noise resampling
  - Kerr220 training runner (cache-index driven)

## New this round

- TSNPE internal speed fix:
  - truncation diagnostics now use density-estimator log-prob
    (avoids very slow direct-posterior rejection sampling inside rounds).
- Fig.1 start script added:
  - `scripts/07_reproduce_fig1_start.py`
  - writes:
    - `reports/figures/fig1_start_sbi_quick.png`
    - `reports/posteriors/fig1_start/*.npz`
    - `reports/posteriors/fig1_start/fig1_start_summary.json`

## Current quality level of Fig.1

Current figure is a **start version**:
- TSNPE trained with very small simulation budget for runtime.
- SBI contours are therefore not converged to paper quality.
- pyRing contours are not overlaid yet.

It is suitable for pipeline validation, not final manuscript-level comparison.

## Highest-priority gaps to close

1. Increase TSNPE simulation budget per case and rerun Fig.1.
2. Add pyRing posterior import/overlay on the same axes.
3. Add quantitative agreement metrics (median shifts, CI overlap).
4. Freeze a reproducible config for "paper-like" run and save all seeds.

## Audit Fields (Paper-Alignment)

1. Prior by case:
- Source of truth: `scripts/08_run_fig1_paper_precision.py::_prior_bounds`.
- Current setting:
  - `M_f [20,300] Msun`, `chi_f [0,0.99]`
  - For each mode: `A in [0, 50e-21]` (with `(2,2,0)` using `A_low=0.1e-21`), `phi in [0,2pi]`.
- Table II alignment: mass/spin ranges aligned. Any deviation must be logged in run summary `prior` block.

2. TSNPE truncation definition:
- Source of truth: `src/rd_sbi/inference/tsnpe_runner.py`.
- Current setting: density-threshold HPD approximation.
  - `tau = quantile(log q_phi(theta|x_o), epsilon)`
  - accept region: `log q_phi(theta|x_o) >= tau`
- Explicitly not box-quantile truncation in current runner.

3. Round budgets and stopping:
- Paper-like defaults:
  - round1: `50k`
  - later rounds: `100k`
  - `epsilon=1e-4`
  - `stopping_ratio=0.8`
  - `batch_size=512`
- Serialized in `kerr*_run_summary.json -> params` and `diagnostics`.

4. Fixed injection/context parameters:
- Source of truth: `configs/injections/kerr220.yaml`, `kerr221.yaml`, `kerr330.yaml`.
- All cases use fixed sky/time context (`ra`, `dec`, `psi`, `gps_h1`, detector delay setting).
- Kerr330 uses different inclination to excite `(3,3,0)`.
- Runtime summaries now include `injection_context`.

5. SNR definition/report:
- Current report file: `reports/tables/snr_validation_report.json`.
- Stored fields include `network_snr`, per-detector SNR, target SNR, and tolerance check.
- This is the audit anchor for Fig.1 injection consistency.

6. pyRing-vs-SBI quantitative agreement:
- Overlay script: `scripts/10_overlay_fig1_pyring.py`.
- Summary artifact: `reports/posteriors/fig1_overlay_summary.json` (or run-local `overlay_summary.json`).
- Metrics per case:
  - `wasserstein_Mf`, `wasserstein_chi`
  - `ks_Mf_D`, `ks_chi_D`
  - `area_ratio_90_sbi_over_pyring`
