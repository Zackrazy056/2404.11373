# kerr220 SNR Decomposition Report

## Summary

- target network SNR: `14.000`
- final network SNR: `12.551`
- gap: `1.449` (10.35%)

## Decomposition

- waveform-only unit-PSD proxy RSS: `2.117394e-22`
- after projection unit-PSD proxy (no delay): `1.377239e-22`
- after projection unit-PSD proxy (with delay): `1.121200e-22`
- after PSD choice, no delay: `15.720896`
- final network SNR: `12.551302`

## Dominant effect

The current detector-delay handling inside the fixed 0.1 s window is the dominant SNR loss:

- no-delay network SNR: `15.720896`
- final network SNR: `12.551302`
- network SNR loss from delay/windowing: `3.169594`
- loss fraction relative to no-delay: `20.16%`
- L1 SNR no-delay: `9.643536`
- L1 SNR with delay: `1.840216`

## Projection and geometry

- H1 `(F+, Fx)`: `(0.578689, -0.451035)`
- L1 `(F+, Fx)`: `(-0.527410, 0.205288)`
- H1 delay from geocenter: `0.014686 s`
- L1 delay from geocenter: `0.007701 s`
- L1 - H1: `-0.006984 s`
- L1 lead samples at 2048 Hz: `14.30`

## Waveform factors

- QNM frequency: `252.075488 Hz`
- damping time: `0.003999 s`
- `Y_plus_220`: `0.630783`
- `Y_cross_220`: `-0.630783`

## Amplitude calibration check

If amplitude were the only issue, the current implementation would need:

- scale factor: `1.115422`
- equivalent `A220`: `5.577111e-21`

But the no-delay SNR is already above target, so amplitude-only rescaling is not the first suspect.
