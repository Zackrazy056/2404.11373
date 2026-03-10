# pyRing 2.3.0 Static Audit for Kerr220

Date: 2026-03-08

## Scope

This note compares three evidence layers against the current `kerr220` SBI pipeline:

1. `pyRingGW-2.3.0` source tree downloaded from Zenodo
2. public GW150914 pyRing configuration/log artifacts that are still accessible
3. the current local `kerr220` engineering pipeline

The goal is not to claim that the public pyRing configs are exactly the same files used for Fig.1 in 2404.11373. The goal is to isolate implementation choices around `t0 / fix-t / trigtime / PSD / ACF / truncation window` that are likely responsible for the current bottleneck: matching `SNR ~= 14` while preserving TSNPE round-1 contraction.

## Evidence Base

### pyRing 2.3.0 source

- `external/pyRingGW-2.3.0/pyRing/initialise.py`
- `external/pyRingGW-2.3.0/pyRing/inject_signal.py`
- `external/pyRingGW-2.3.0/pyRing/waveform.pyx`
- `external/pyRingGW-2.3.0/pyRing/noise.py`
- `external/pyRingGW-2.3.0/pyRing/likelihood.pyx`
- `external/pyRingGW-2.3.0/pyRing/config_files/config_gw150914_production.ini`
- `external/pyRingGW-2.3.0/pyRing/config_files/Quickstart_configs/quick_gw150914_Kerr_220.ini`

### Public pyRing GW150914 artifacts

- `external/pyring_public_kerr220_config.html`
- `external/pyring_public_kerr221_stdout.txt`

### Current local pipeline

- `configs/injections/kerr220.yaml`
- `scripts/08_run_fig1_paper_precision.py`
- `src/rd_sbi/simulator/injection.py`
- `scripts/02_estimate_psd.py`
- `reports/tables/kerr220_snr_validation_report.json`

## Key pyRing Findings

### 1. pyRing time reference is detector-local, not a shared fixed 204-bin window

`pyRing` defines `trigtime` as the reference time in the reference detector, default `H1`:

- `initialise.py:134-166`
- `initialise.py:884-915`

In the likelihood, pyRing does this per detector:

- compute detector-dependent `dt = time_delay[ref_det -> d]`
- define detector-local time axis as `detector.time - (tevent + dt)`
- if truncating, crop with `time_array_raw >= t_start` and keep `duration_n` samples

This is explicit in `likelihood.pyx:424-436`.

This is materially different from the current SBI input representation. Our pipeline currently builds a shared 204-bin dual-detector tensor first and concatenates it later:

- `scripts/08_run_fig1_paper_precision.py:216-278`
- `src/rd_sbi/simulator/injection.py:38-104`

That difference is high risk. It means pyRing effectively performs detector-local alignment before truncation, while our current input representation still exposes the network to a fixed-bin onset-location problem.

### 2. pyRing fixed start time is discretised at sample level

The core damped sinusoid starts at the nearest discrete sample to `t0`:

- `waveform.pyx:506-516`

Public pyRing output also warns:

- when `fix-t` is used, pyRing selects the discrete sample immediately after the requested start time
- the mismatch can be as large as `1 / sampling_rate`

This warning appears in `external/pyring_public_kerr221_stdout.txt`.

At `4096 Hz`, that discretisation is about `0.000244 s`. At `2048 Hz`, the same effect doubles to about `0.000488 s`.

This makes our current `2048 Hz` choice more fragile than pyRing for any start-time-sensitive comparison.

### 3. pyRing public GW150914 production path is 4096 Hz, bandpassed, and truncated to 0.2 s

The public GW150914 stdout shows:

- `sampling-rate = 4096.0`
- `f-min-bp = 100.0`
- `f-max-bp = 500.0`
- `bandpassing = 1`
- `truncate = 1`
- `analysis-duration = 0.2`

See `external/pyring_public_kerr221_stdout.txt`.

This is not a cosmetic difference. It changes:

- start-time discretisation scale
- effective signal bandwidth
- PSD/ACF estimate
- number of retained samples in the likelihood

### 4. pyRing noise model is explicitly ACF -> Toeplitz covariance, with multiple inversion methods

Relevant source path:

- `noise.py:616-858`
- `noise.py:1818-1880`
- `likelihood.pyx:469-483`

Important details:

- if no ACF file is provided, pyRing estimates the ACF from data chunks avoiding the trigger time
- the ACF is averaged over chunks
- the covariance for the truncated signal is built as `toeplitz(ACF_signal)`
- pyRing stores both `inverse_covariance` and `cholesky`
- the likelihood can use:
  - `direct-inversion`
  - `cholesky-solve-triangular`
  - `toeplitz-inversion`

The public stdout also shows:

- Welch PSD is computed
- a standard ACF is estimated for comparison
- “Whitened plots using the FD PSD are illustrative and do not correspond to the full TD treatment applied in the analysis.”

This is significant because our current pipeline also uses a Toeplitz/Cholesky whitening path, but not the same acquisition and preprocessing path as public pyRing GW150914 runs.

### 5. pyRing Kerr amplitudes are parameterised with `reference-amplitude`

Source:

- `initialise.py:969-970`
- `inject_signal.py:201-245`
- `waveform.pyx:940-1004`

Public configs use:

- `reference-amplitude = 1E-21`

while amplitudes are sampled in dimensionless-ish units relative to that reference scale.

This is different from our current local config, where `A_220 = 5e-21` is injected directly as the waveform amplitude in physical units:

- `configs/injections/kerr220.yaml:29-39`

This amplitude convention mismatch is medium risk for overlay-level agreement, but it is not the most likely explanation for the current “SNR fixed but TSNPE no longer contracts” failure.

## Public Config Line-of-Sight

### Public production-like Kerr220 config page

Observed fields:

- `trigtime = 1126259462.4237232`
- `kerr-modes = [(2,2,2,0)]`
- `reference-amplitude = 1E-21`
- `nlive = 2048`
- `maxmcmc = 2048`
- `fix-t = 0.0`

Source: `external/pyring_public_kerr220_config.html`

### Public Kerr221 stdout

Observed fields:

- `trigtime = 1126259462.423235`
- `ref-det = H1`
- `sampling-rate = 4096.0`
- `f-min-bp = 100.0`
- `f-max-bp = 500.0`
- `bandpassing = 1`
- `fft-acf = 1`
- `acf-simple-norm = 1`
- `truncate = 1`
- `analysis-duration = 0.2`
- `reference-amplitude = 1e-21`
- `nlive = 2048`
- `maxmcmc = 2048`

Source: `external/pyring_public_kerr221_stdout.txt`

### pyRing source bundled configs

`config_gw150914_production.ini`:

- `trigtime = 1126259462.4232266`
- `reference-amplitude = 1E-21`
- `fix-t = 0.0`
- `noise-chunksize = 4.0`
- `signal-chunksize = 4.0`

`quick_gw150914_Kerr_220.ini`:

- `trigtime = 1126259462.4232266`
- `reference-amplitude = 1E-21`
- `fix-t = 0.00335`
- `fix-cosiota = -0.9`

These public/source configs are not fully consistent with one another. That is itself a useful signal: we should treat them as implementation clues, not as a single authoritative Fig.1 configuration.

## Current Local Pipeline Snapshot

Current `kerr220` local settings:

- `sample_rate_hz = 2048.0`
- `duration_s = 0.1`
- `t_start_s = 0.0`
- `gps_h1 = 1126259462.42323`
- `use_detector_time_delay = true`
- `reference_detector = H1`
- direct injected mode amplitude `5.0e-21`

Source: `configs/injections/kerr220.yaml`

Current `kerr220` calibrated SNR after the window-anchor experiment:

- `network_snr = 14.27205026991145`
- `H1 = 10.521104294497556`
- `L1 = 9.643535831383868`

Source: `reports/tables/kerr220_snr_validation_report.json`

Current SBI training path:

- 408-dim concatenated whitened input
- `204` bins per detector
- exact PSD -> ACF -> Toeplitz -> Cholesky whitening
- MLP embedding, not a time-shift-invariant representation

Source: `scripts/08_run_fig1_paper_precision.py`

## Difference Table

| Topic | Current local pipeline | pyRing 2.3.0 / public GW150914 clues | Risk | Likely impact |
|---|---|---|---|---|
| Time-axis construction | Shared 204-bin detector arrays are built first, then concatenated | Detector-local time axis uses `detector.time - (tevent + dt)` and truncates after detector delay | High | Strong candidate root cause for “SNR fixed but round-1 contraction lost” |
| Start-time discretisation | 2048 Hz | 4096 Hz in public GW150914 runs | High | Sample-level `t0` error is 2x worse locally |
| Truncation segment length | `0.1 s` | public GW150914 stdout uses `0.2 s` | High | Changes retained ringdown content and covariance size |
| Bandpass | none in current SBI preprocessing path | public pyRing GW150914 uses `100-500 Hz` | High | Changes both SNR distribution and effective waveform morphology |
| PSD source duration | local PSD built from 32 s GWOSC files | public pyRing downloads 4096 s data and computes Welch + ACF | High | Noise estimate likely not matched closely enough |
| ACF estimation | local whitening is from saved PSD resampled to 204 bins | pyRing estimates ACF from data chunks, optionally compares with Welch PSD | High | Noise model mismatch |
| ACF normalization | local path does not mirror pyRing `acf-simple-norm=1` chunk estimator | public pyRing uses `acf-simple-norm = 1` | Medium | May shift covariance tails and TD weighting |
| Likelihood weighting | local SBI uses whitened observations for amortized inference | pyRing likelihood supports direct inverse, Cholesky, Toeplitz solve | Medium | Not directly comparable at training time, but affects overlay agreement |
| Amplitude convention | injected `A_220 = 5e-21` directly | public pyRing uses `reference-amplitude = 1e-21` plus sampled amplitudes | Medium | Important for exact overlay calibration, but not first suspect for current contraction failure |
| Sky/time reference | `gps_h1` with `reference_detector = H1` | `trigtime` is reference time in `ref-det = H1` | Low | Broadly aligned |
| QNM backend | local `fit` backend for required Kerr modes | pyRing `qnm_fit = 1` by default | Low | Broadly aligned in intent |

## Highest-Value Conclusions

1. The most important gap is no longer raw SNR calibration.

   We already brought `kerr220` to `network SNR ~= 14.27`. The remaining bottleneck is that our training representation is still not equivalent to the way pyRing applies detector delay and truncation.

2. The highest-risk mismatch is detector-local truncation versus shared-window representation.

   pyRing truncates after moving each detector into its own local arrival-time frame. Our current anchored shared-window input preserves SNR better, but it also moved the onset structure in a way that made round-1 TSNPE contraction collapse.

3. The second highest-risk mismatch is `4096 Hz + 0.2 s + 100-500 Hz bandpass` versus `2048 Hz + 0.1 s + no explicit bandpass`.

   This is likely large enough that even a numerically correct SBI implementation can still fail to match pyRing behavior or the paper’s Fig.1 shape.

4. The public pyRing configs are internally inconsistent on `fix-t`.

   Some public/source configs show `fix-t = 0.0`, while quickstart Kerr220 shows `fix-t = 0.00335`. This means we should not overfit to one public config file. The right next step is to match the implementation semantics, not to hardcode one public number prematurely.

## Recommended Next Steps

### Priority 1

Rebuild the SBI observation representation to mimic pyRing detector-local truncation semantics:

- keep detector delays
- define detector-local time arrays relative to `trigtime + delay`
- truncate each detector after the requested start time
- only then whiten and concatenate

This is the single most promising experiment to restore round-1 contraction without sacrificing the newly recovered `SNR ~= 14`.

### Priority 2

Run a strict `kerr220` ablation on three preprocessing variants only:

1. current `2048 Hz / 0.1 s / no bandpass`
2. `4096 Hz / 0.1 s / 100-500 Hz bandpass`
3. `4096 Hz / 0.2 s / 100-500 Hz bandpass`

Keep everything else fixed and compare:

- network SNR
- round-1 `truncated_prior_volume`
- round-1 `probe_acceptance_rate`

### Priority 3

Do not spend the next iteration on amplitude renormalisation first.

The amplitude convention difference is real, but the evidence gathered here says the time/truncation/noise path is more likely to explain the current failure mode.
