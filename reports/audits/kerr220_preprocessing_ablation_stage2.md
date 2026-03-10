# Kerr220 Preprocessing Ablation Stage-2

| label | state | sample_rate_hz | duration_s | bandpass_hz | network_snr | round1_volume | round1_probe_accept | contraction_ok | epochs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| detector_local_2048_0p1_nobp | completed | 2048.000000 | 0.100000 | none | 15.720895 | 0.018799 | 0.018460 | true | 88 |
| detector_local_4096_0p1_nobp | completed | 4096.000000 | 0.100000 | none | 35.674237 | 0.010193 | 0.009340 | true | 74 |
| detector_local_4096_0p2_nobp | completed | 4096.000000 | 0.200000 | none | 37.031459 | 0.010132 | 0.010540 | true | 76 |
| detector_local_4096_0p2_bp100_500 | failed | 4096.000000 | 0.200000 | 100-500 | n/a | 0.001953 | 0.002340 | true | 172 |

## Raw results
{
  "case": "kerr220",
  "timing_mode": "detector_local_truncation",
  "runs": [
    {
      "state": "completed",
      "round1_volume": 0.018798828125,
      "round1_probe_accept": 0.018459999933838844,
      "round1_epochs": 88,
      "round1_training_seconds": 385.8559901714325,
      "round1_total_seconds": 391.31640005111694,
      "contraction_ok": true,
      "measured_network_snr": 15.720895091764794,
      "per_detector_snr": {
        "H1": 12.415666102113503,
        "L1": 9.643535540822363
      },
      "preprocessing": {
        "sample_rate_hz": 2048.0,
        "duration_s": 0.1,
        "input_dim": 408,
        "per_detector_bins": 204,
        "bandpass": {
          "enabled": false,
          "f_max_hz": null,
          "f_min_hz": null,
          "filter": null,
          "paper_faithful": true
        },
        "window_anchor": {
          "detector_local_truncation_like_pyring": true,
          "earliest_detector": null,
          "earliest_relative_delay_s": null,
          "observation_timing_mode": "detector_local_truncation",
          "reference_detector": "H1",
          "relative_delay_to_reference_s": {
            "H1": 0.0,
            "L1": -0.006984385696649813
          },
          "shift_s": 0.0,
          "strategy": "detector_local_truncation_like_pyring",
          "t_start_detector_s": {
            "H1": 0.0,
            "L1": 0.0
          }
        }
      },
      "error": null,
      "exit_code": 0,
      "wall_seconds": 396.2917215824127,
      "run_dir": "C:\\Users\\97747\\GW QNM EMRI PROJECT\\2404.11373v3\\2404.11373\\reports\\posteriors\\kerr220_preprocessing_ablation_stage2\\detector_local_2048_0p1_nobp",
      "label": "detector_local_2048_0p1_nobp",
      "requested": {
        "sample_rate_hz": 2048.0,
        "duration_s": 0.1,
        "bandpass_min_hz": null,
        "bandpass_max_hz": null
      }
    },
    {
      "state": "completed",
      "round1_volume": 0.01019287109375,
      "round1_probe_accept": 0.009340000338852406,
      "round1_epochs": 74,
      "round1_training_seconds": 330.07253217697144,
      "round1_total_seconds": 337.0459544658661,
      "contraction_ok": true,
      "measured_network_snr": 35.674236762718614,
      "per_detector_snr": {
        "H1": 28.469900959237314,
        "L1": 21.496881354599378
      },
      "preprocessing": {
        "sample_rate_hz": 4096.0,
        "duration_s": 0.1,
        "input_dim": 818,
        "per_detector_bins": 409,
        "bandpass": {
          "enabled": false,
          "f_max_hz": null,
          "f_min_hz": null,
          "filter": null,
          "paper_faithful": true
        },
        "window_anchor": {
          "detector_local_truncation_like_pyring": true,
          "earliest_detector": null,
          "earliest_relative_delay_s": null,
          "observation_timing_mode": "detector_local_truncation",
          "reference_detector": "H1",
          "relative_delay_to_reference_s": {
            "H1": 0.0,
            "L1": -0.006984385696649813
          },
          "shift_s": 0.0,
          "strategy": "detector_local_truncation_like_pyring",
          "t_start_detector_s": {
            "H1": 0.0,
            "L1": 0.0
          }
        }
      },
      "error": null,
      "exit_code": 0,
      "wall_seconds": 342.1583466529846,
      "run_dir": "C:\\Users\\97747\\GW QNM EMRI PROJECT\\2404.11373v3\\2404.11373\\reports\\posteriors\\kerr220_preprocessing_ablation_stage2\\detector_local_4096_0p1_nobp",
      "label": "detector_local_4096_0p1_nobp",
      "requested": {
        "sample_rate_hz": 4096.0,
        "duration_s": 0.1,
        "bandpass_min_hz": null,
        "bandpass_max_hz": null
      }
    },
    {
      "state": "completed",
      "round1_volume": 0.0101318359375,
      "round1_probe_accept": 0.010540000163018703,
      "round1_epochs": 76,
      "round1_training_seconds": 343.8114948272705,
      "round1_total_seconds": 356.56402015686035,
      "contraction_ok": true,
      "measured_network_snr": 37.0314589711607,
      "per_detector_snr": {
        "H1": 29.125110240376497,
        "L1": 22.87043740330898
      },
      "preprocessing": {
        "sample_rate_hz": 4096.0,
        "duration_s": 0.2,
        "input_dim": 1638,
        "per_detector_bins": 819,
        "bandpass": {
          "enabled": false,
          "f_max_hz": null,
          "f_min_hz": null,
          "filter": null,
          "paper_faithful": true
        },
        "window_anchor": {
          "detector_local_truncation_like_pyring": true,
          "earliest_detector": null,
          "earliest_relative_delay_s": null,
          "observation_timing_mode": "detector_local_truncation",
          "reference_detector": "H1",
          "relative_delay_to_reference_s": {
            "H1": 0.0,
            "L1": -0.006984385696649813
          },
          "shift_s": 0.0,
          "strategy": "detector_local_truncation_like_pyring",
          "t_start_detector_s": {
            "H1": 0.0,
            "L1": 0.0
          }
        }
      },
      "error": null,
      "exit_code": 0,
      "wall_seconds": 361.64640164375305,
      "run_dir": "C:\\Users\\97747\\GW QNM EMRI PROJECT\\2404.11373v3\\2404.11373\\reports\\posteriors\\kerr220_preprocessing_ablation_stage2\\detector_local_4096_0p2_nobp",
      "label": "detector_local_4096_0p2_nobp",
      "requested": {
        "sample_rate_hz": 4096.0,
        "duration_s": 0.2,
        "bandpass_min_hz": null,
        "bandpass_max_hz": null
      }
    },
    {
      "state": "failed",
      "round1_volume": 0.001953125,
      "round1_probe_accept": 0.0023399998899549246,
      "round1_epochs": 172,
      "round1_training_seconds": 1019.2612204551697,
      "round1_total_seconds": 1076.8647210597992,
      "contraction_ok": true,
      "measured_network_snr": null,
      "per_detector_snr": null,
      "preprocessing": {
        "sample_rate_hz": null,
        "duration_s": null,
        "input_dim": null,
        "per_detector_bins": null,
        "bandpass": null,
        "window_anchor": null
      },
      "error": "[Errno 22] Invalid argument",
      "exit_code": 120,
      "wall_seconds": 1082.168881893158,
      "run_dir": "C:\\Users\\97747\\GW QNM EMRI PROJECT\\2404.11373v3\\2404.11373\\reports\\posteriors\\kerr220_preprocessing_ablation_stage2\\detector_local_4096_0p2_bp100_500",
      "label": "detector_local_4096_0p2_bp100_500",
      "requested": {
        "sample_rate_hz": 4096.0,
        "duration_s": 0.2,
        "bandpass_min_hz": 100.0,
        "bandpass_max_hz": 500.0
      }
    }
  ]
}
