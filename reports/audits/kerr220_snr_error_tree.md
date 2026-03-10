# Kerr220 SNR Error Tree Report

Date: 2026-03-10

## Scope

This note synthesizes existing local artifacts for `kerr220` and answers one narrow question:

What is actually responsible for the current SNR mismatch relative to the paper target `network SNR = 14.0`?

This report does not introduce a new training run. It consolidates:

- `reports/tables/kerr220_snr_decomposition_report.json`
- `reports/tables/kerr220_window_anchor_experiment.json`
- `reports/tables/kerr220_snr_validation_report.json`
- `reports/audits/kerr220_preprocessing_ablation_round1.md`
- `reports/audits/kerr220_preprocessing_ablation_stage2.md`
- `reports/audits/pyring_2_3_0_static_audit_kerr220.md`

## Executive Answer

The dominant SNR problem is not a missing "better SNR formula". The dominant problem is that different observation-construction semantics produce materially different signals before the same SNR calculator is even applied.

The strongest evidence is:

- the legacy fixed-window delay path gives `12.551`
- anchoring the shared window gives `14.272`
- detector-local truncation gives `15.721`

All three numbers are produced under the same local waveform/PSD machinery, but with different time-window semantics.

That means the main issue is upstream of the SNR metric itself.

## Baseline Matrix

| Branch | H1 SNR | L1 SNR | Network SNR | Delta vs target 14.0 | Interpretation | Primary source |
|---|---:|---:|---:|---:|---|---|
| Legacy fixed-window delay baseline | 12.416 | 1.840 | 12.551 | -1.449 | Matches the old formal `fig1`-side SNR deficit; L1 is heavily cropped by early arrival | `kerr220_snr_decomposition_report.json`, `kerr220_20260308-111922/kerr220_run_summary.json` |
| Anchored shared window | 10.521 | 9.644 | 14.272 | +0.272 | Shift common window so earliest detector starts at `t=0`; nearly recovers paper target | `kerr220_window_anchor_experiment.json`, `kerr220_snr_validation_report.json` |
| No-delay same-start / detector-local 2048-0.1-nobp | 12.416 | 9.644 | 15.721 | +1.721 | Removes the L1 cropping loss; also the branch with good round-1 contraction | `kerr220_snr_decomposition_report.json`, `kerr220_preprocessing_ablation_round1.md`, `kerr220_preprocessing_ablation_stage2.md` |
| Detector-local 4096-0.1-nobp | 28.470 | 21.497 | 35.674 | +21.674 | Moving toward public pyRing sample rate without other calibration strongly increases SNR | `kerr220_preprocessing_ablation_stage2.md` |
| Detector-local 4096-0.2-nobp | 29.125 | 22.870 | 37.031 | +23.031 | Longer duration further increases retained energy under current implementation | `kerr220_preprocessing_ablation_stage2.md` |
| Detector-local 4096-0.2-bp100-500 | n/a | n/a | n/a | n/a | Run failed before saving SNR | `kerr220_preprocessing_ablation_stage2.md` |

## Error Tree

```text
paper target: network SNR = 14.000

legacy fixed-window delay branch
  network SNR = 12.551
  delta vs target = -1.449
  meaning:
    H1 keeps its full 0.1 s window
    L1 arrives earlier by 0.006984 s
    L1 loses about 14.30 samples at 2048 Hz and is heavily cropped

  branch A: remove the delay-cropping effect
    no-delay same-start / detector-local 2048-0.1-nobp = 15.721
    delta vs legacy baseline = +3.170
    delta vs target = +1.721
    dominant change:
      L1 SNR recovers from 1.840 to 9.644

  branch B: anchor the shared window to earliest arrival
    anchored shared window = 14.272
    delta vs legacy baseline = +1.721
    delta vs target = +0.272
    dominant changes:
      H1 SNR drops from 12.416 to 10.521
      L1 SNR recovers from 1.840 to 9.644

pyRing/public-like preprocessing branch under current local implementation
  detector-local 4096-0.1-nobp = 35.674
  detector-local 4096-0.2-nobp = 37.031
  meaning:
    these changes do not naturally move the local pipeline toward target 14
    they move it far above target unless some other convention also changes
```

## Quantitative Attribution by Factor

### 1. Legacy detector-delay windowing is the largest confirmed loss

From `kerr220_snr_decomposition_report.json`:

- no-delay same-start SNR: `15.720896`
- legacy fixed-window delay SNR: `12.551302`
- network SNR loss from this effect alone: `3.169594`
- fractional loss relative to no-delay branch: `20.16%`
- L1 SNR loss: `9.643536 -> 1.840216`, a drop of `7.803320`

This is the single largest measured SNR loss in the current evidence base.

### 2. Shared-window anchoring almost closes the paper gap

From `kerr220_window_anchor_experiment.json`:

- legacy fixed-window delay SNR: `12.551302`
- anchored shared-window SNR: `14.272050`
- recovered network SNR: `+1.720748`
- residual offset to target after anchoring: `+0.272050`

This is why the current evidence says the main problem is not first-order amplitude normalization.

### 3. Anchoring is not the same as detector-local truncation

The anchored shared-window branch and detector-local branch are numerically different:

- anchored shared window: `14.272050`
- detector-local 2048-0.1-nobp: `15.720895`
- difference: `1.448845`

The reason is simple:

- anchored shared window recovers L1, but it also shifts H1 later in the common 0.1 s window
- detector-local truncation lets both detectors start at `t_start = 0` in their own local frames

This is the core semantic mismatch:

- one branch is closer to the paper SNR target
- the other branch is closer to pyRing-like detector-local truncation and gives better TSNPE round-1 contraction

### 4. Sample rate and duration changes are not an "easy SNR optimization"

Stage-2 ablation shows:

- `2048 / 0.1 / no bandpass / detector-local` -> `15.721`
- `4096 / 0.1 / no bandpass / detector-local` -> `35.674`
- `4096 / 0.2 / no bandpass / detector-local` -> `37.031`

So moving toward public pyRing-like sample rate and duration, without fully matching the rest of the inference semantics, does not gently improve the target mismatch. It blows the local SNR scale upward.

### 5. Bandpass remains unresolved in the local matrix

The intended `4096 / 0.2 / 100-500 Hz` run failed before emitting an SNR summary.

Therefore:

- bandpass is still a high-priority missing measurement
- but it cannot currently be used as evidence that "advanced SNR processing" solves the main problem

### 6. Amplitude-only fixing is inconsistent across branches

If one insists on hitting `14.0` by amplitude scaling alone, the required scale factor depends on which branch is treated as the reference:

| Branch treated as truth | Current SNR | Required scale to hit 14.0 |
|---|---:|---:|
| Legacy fixed-window delay | 12.551302 | `1.115422` |
| Anchored shared window | 14.272050 | `0.980938` |
| Detector-local 2048-0.1-nobp | 15.720896 | `0.890534` |

These factors are mutually inconsistent.

That means amplitude-only calibration is not a stable root-cause fix. It only hides whichever upstream semantic mismatch produced the chosen baseline.

## What Is Probably Not the Root Cause

Based on the existing evidence, the following are not the leading explanation for the current `kerr220` SNR mismatch:

- QNM fit backend selection by itself
- a small global amplitude underscaling by itself
- replacing the SNR metric with a more sophisticated metric, while leaving the signal-construction path unchanged

The reason is structural:

- the same SNR routine already produces `12.551`, `14.272`, and `15.721`
- therefore the main sensitivity sits in the input signal definition, not in the final SNR formula

## What Is Still Not Fully Aligned With the Paper / pyRing

The static pyRing audit says the remaining high-risk mismatches are:

- detector-local truncation semantics versus shared 204-bin common window
- `4096 Hz` public pyRing clue versus local `2048 Hz`
- `0.2 s` public pyRing clue versus local `0.1 s`
- public `100-500 Hz` bandpass versus local no-bandpass path
- pyRing ACF/Toeplitz covariance acquisition path versus local PSD-resample path
- possible `reference-amplitude` convention differences

The important point is that these are pipeline-alignment problems, not just SNR-estimator problems.

## Bottom Line

The current `kerr220` SNR mismatch is best understood as a semantic alignment problem with three competing branches:

1. Legacy fixed-window delay handling gives the low SNR `12.551`.
2. Anchored shared-window handling gives the near-target SNR `14.272`.
3. Detector-local truncation gives the contraction-friendly but too-high SNR `15.721`.

So the real question is not "how do we compute a more advanced SNR?".

The real question is:

Which observation semantics should be treated as the paper-faithful object before any further SNR calibration is attempted?

## Recommended Next Actions

1. Add one explicit bandpass row to the matrix by fixing the failed `4096 / 0.2 / 100-500 Hz` run.
2. Freeze one reference observation semantic for `kerr220`:
   `anchored shared window` or `detector-local truncation`.
3. Only after step 2, decide whether amplitude calibration is still necessary.
4. If deeper SNR diagnostics are still wanted, add a pyRing-like TD covariance-weighted SNR as a diagnostic, not as the first-line optimization target.
