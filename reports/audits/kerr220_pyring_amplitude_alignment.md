# Kerr220 pyRing Amplitude Alignment Audit

Date: 2026-03-10

## Scope

This note audits whether the current `kerr220` local injection amplitude
convention is semantically aligned with the vendored `pyRing 2.3.0` Kerr
waveform implementation.

The practical question is:

- if local SBI uses `A_220 = 5e-21`
- and pyRing uses `reference-amplitude = 1e-21` with `A2220 = 5`

do these define the same source-frame ringdown waveform, or is there an
amplitude-convention mismatch large enough to explain why the paper-like
pyRing path yields `network SNR ~ 30` instead of `~14`?

## Code Paths Reviewed

- `src/rd_sbi/waveforms/ringdown.py`
- `src/rd_sbi/simulator/injection.py`
- `external/pyRingGW-2.3.0/pyRing/waveform.pyx`
- `external/pyRingGW-2.3.0/pyRing/inject_signal.py`
- `external/pyRingGW-2.3.0/pyRing/pyRing.py`

## Source-Level Convention Comparison

### Local SBI waveform

Local ringdown polarizations are built directly from a physical strain-scale
mode amplitude:

- `h_plus += amplitude * envelope * cos(phase_t) * y_plus`
- `h_cross += amplitude * envelope * sin(phase_t) * y_cross`

So in the local path, `mode.amplitude` is already a physical strain amplitude.

Current `kerr220` config:

- `A_220 = 5e-21`
- `phase = 1.047`
- `inclination = pi`

### pyRing Kerr waveform

pyRing first builds a complex mode coefficient `A2220 * exp(i phi2220)` and
then multiplies the full waveform by a global prefactor:

- if `reference_amplitude != 0`, `prefactor = reference_amplitude`
- else `prefactor = (Mf / r) * mass_dist_units_conversion`

So with `reference-amplitude = 1e-21`, the physical scale of the `(2,2,0)`
mode is:

- `A_phys = A2220 * 1e-21`

For the single non-precessing `220` case, that matches the local convention
exactly if:

- `A2220 = A_local / reference_amplitude`

For `A_local = 5e-21` and `reference_amplitude = 1e-21`, this gives:

- `A2220 = 5`

## Exact Waveform Equivalence Test

I ran a direct waveform-level comparison using:

- `Mf = 67`
- `af = 0.67`
- `phase = 1.047`
- `inclination = pi`
- `sample_rate = 4096 Hz`
- `duration = 0.2 s`
- local amplitude `A = 5e-21`
- pyRing `reference-amplitude = 1e-21`, `A2220 = 5`

The resulting `h_plus` and `h_cross` agree to numerical precision.

Key results:

- `h_plus(t=0)` ratio `pyRing / local = 1.0`
- `h_cross(t=0)` ratio `pyRing / local = 1.0`
- RMS ratio for both polarizations `~ 1.00000000006`
- maximum absolute difference `~ 1e-30`

This is strong evidence that the waveform-level amplitude convention is already
aligned.

## Distance-Prefactor Sanity Check

If pyRing does **not** use `reference-amplitude` and instead uses the default
`Mf/r` prefactor with `logdistance = 6.0857`, then:

- `distance ~= 439.53 Mpc`
- `(Mf/r) * mass_dist_units_conversion ~= 7.29e-21`

So in the distance-based path:

- `A2220 = 1` would already correspond to a physical prefactor of `7.29e-21`

This matters conceptually, but it is **not** what the current paper-like pyRing
run is doing. The current run uses `reference-amplitude = 1e-21`, so this
distance prefactor is bypassed.

## What This Means For The `SNR ~ 30` Result

The current paper-like pyRing run `kerr220_20260310-paperlike-main` gives:

- network optimal TD SNR `= 30.435`
- network matched-filter TD SNR `= 30.579`

Because the waveform-level amplitude mapping is already exact, this `~30`
cannot be blamed on a source-waveform amplitude-convention mismatch.

The remaining dominant causes are upstream/downstream of that mapping:

- `4096 Hz` instead of `2048 Hz`
- `0.2 s` truncation instead of `0.1 s`
- explicit `100-500 Hz` bandpass
- pyRing TD covariance / ACF weighting path
- detector-local truncation semantics

In other words:

- amplitude convention is aligned
- SNR measurement domain is not aligned

## Direct Preprocessing Counter-Test

I then ran pyRing again with the **same** amplitude mapping:

- `reference-amplitude = 1e-21`
- `A2220 = 5`

but switched preprocessing to a local-like path:

- `2048 Hz`
- `0.1 s`
- `no bandpass`

This run produced:

- network matched-filter TD SNR `= 14.319`
- network optimal TD SNR `= 14.146`
- logB-implied SNR `= 14.381`

This is the key alignment result of the audit.

It shows that:

1. the amplitude convention does **not** need a semantic fix
2. the same amplitude mapping already yields `SNR ~ 14`
3. the jump from `~14` to `~30` is driven primarily by preprocessing /
   covariance choices, not by how the mode amplitude is parameterized

## Overlay Sanity Check

Using the current locallike pyRing export as the reference posterior and
overlaying it against the existing SBI posterior
`kerr220_20260308-111922` gives substantially better agreement than the
previous paper-like pyRing (`4096/0.2/bandpassed`) baseline.

Temporary overlay metrics:

- Wasserstein `Mf`: `1.60`
- Wasserstein `chi`: `0.052`
- KS `Mf`: `0.101`
- KS `chi`: `0.110`
- 90% area ratio `SBI / pyRing`: `1.345`

For comparison, the earlier `4096/0.2/bandpassed` pyRing baseline gave much
larger discrepancies:

- Wasserstein `Mf`: `4.28`
- Wasserstein `chi`: `0.127`
- KS `Mf`: `0.395`
- KS `chi`: `0.397`

This strongly suggests that for the current SBI reproduction path, the
`2048/0.1/no-bandpass` pyRing baseline is the right object to compare against.

## If One Forced pyRing To Hit 14

Pure amplitude rescaling from the current pyRing result would require roughly:

- scale factor `14 / 30.435 ~= 0.460`
- equivalent `A2220 ~= 2.30` if `reference-amplitude = 1e-21` is kept fixed

That is a valid calibration exercise, but it would be a **numerical retune**,
not evidence that the current amplitude convention is wrong.

## Audit Conclusion

The current best interpretation is:

1. `A_220 = 5e-21` in the local SBI path is already correctly represented in
   pyRing as `reference-amplitude = 1e-21` with `A2220 = 5`.
2. The source-waveform amplitude convention is not the root cause of the
   current `pyRing SNR ~ 30`.
3. The real remaining mismatch is that the local `SNR ~ 14` target and the
   paper-like pyRing `SNR ~ 30` are being evaluated under different
   preprocessing / covariance semantics.

## Recommended Next Checks

1. Compare the pyRing `2048/0.1/no-bandpass` run against the local
   `14.27` anchored-window validation and the detector-local SBI run.
2. Decide which of these is the authoritative paper target:
   `A_220 = 5e-21` in a 2048/0.1 local validation domain, or
   `A2220 = 5` in a pyRing 4096/0.2/bandpassed domain.
3. Only after that decision, perform any amplitude calibration.
