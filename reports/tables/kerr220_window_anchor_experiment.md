# kerr220 Window Anchor Experiment

## Summary

- current network SNR: `12.551302`
- anchored network SNR: `14.272050`
- target network SNR: `14.000000`
- gain from anchor: `1.720748`
- gap to target after anchor: `-0.272050`

## Key result

This single timing/window change raises `kerr220` from `12.55` to `14.27`, without changing:

- amplitude
- QNM mapping
- antenna pattern
- PSD choice

## Interpretation

The dominant SNR deficit comes from the current fixed-window anchor convention:

- H1 is anchored at `t_start_s = 0`
- L1 arrives earlier by `-0.006984 s`
- that means L1 starts before the sampled window and gets cropped

The anchored experiment shifts the common window by `0.006984 s` so the earliest detector starts at `t = 0`.
