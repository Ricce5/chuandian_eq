# Oracle, RTPP v2, and ETAS Cross-Experiment Analysis

## Scope

This note compares the following experiments after aligning the four datasets'
catalog thresholds and train/validation/test boundaries:

- RTPP v2: `experiments/rtpp_v2_multi_bg_split_0.7`
- ETAS: `experiments/etas_multi_ds_bg_split_0.7`
- Oracle: `experiments/oracle_multi_dataset_0.7`

All comparisons below use `best_model_1.pth` test metrics and the saved
sliding-window evaluation outputs. RTPP v2 and ETAS each evaluate seven
background variants over three seeds, while Oracle has one configuration per
dataset over three seeds. Therefore, the main operational comparison selects
the lowest test-start RMSE background variant for RTPP v2 and ETAS per
dataset, then compares it with Oracle.

The test-start sliding-window metrics are based on the windows satisfying
`t_forecast >= test_start_t`. Lower RMSE, MAE, and CRPS are better. The
sliding prediction interval uses the 2.5th and 97.5th percentiles, so coverage
should be close to 0.95.

## Sources

- [RTPP v2 test metrics](../../experiments/rtpp_v2_multi_bg_split_0.7/reports/rtpp_v2_grid_metrics_group_mean_std.csv)
- [ETAS test metrics](../../experiments/etas_multi_ds_bg_split_0.7/reports/etas_grid_metrics_group_mean_std.csv)
- [Oracle test metrics](../../experiments/oracle_multi_dataset_0.7/reports/oracle_grid_metrics_group_mean_std.csv)
- [RTPP v2 test-start sliding metrics](../../experiments/rtpp_v2_multi_bg_split_0.7/reports/sliding_window_eval_metrics_by_split_start_group_mean_std.csv)
- [ETAS test-start sliding metrics](../../experiments/etas_multi_ds_bg_split_0.7/reports/sliding_window_eval_metrics_by_split_start_group_mean_std.csv)
- [Oracle test-start sliding metrics](../../experiments/oracle_multi_dataset_0.7/reports/sliding_window_eval_metrics_by_split_start_group_mean_std.csv)

All three experiments have complete evaluation coverage for this comparison:

| Experiment | Runs | Valid best test metrics | Valid sliding metrics | Test-start rows |
| --- | ---: | ---: | ---: | ---: |
| RTPP v2 | 84 | 84 | 84 | 84 |
| ETAS | 84 | 84 | 84 | 84 |
| Oracle | 12 | 12 | 12 | 12 |

## Test-Start Forecast Comparison

| Dataset | RTPP v2 selected configuration | RTPP v2 RMSE | ETAS selected configuration | ETAS RMSE | Oracle RMSE | Best result |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `CB_HAB1a` | `kernel` | 61.62 +/- 6.69 | `mamba` | **19.14 +/- 0.72** | 83.50 +/- 34.62 | ETAS |
| `CB_HAB4` | `kernel_norm_0` | **122.73 +/- 3.62** | `kernel_norm_0` | 144.37 +/- 0.85 | 166.06 +/- 0.81 | RTPP v2 |
| `PNR_1z` | `proportional_norm_0` | **50.57 +/- 2.99** | `mamba_norm_0` | 57.61 +/- 9.73 | 104.48 +/- 7.93 | RTPP v2 |
| `St1_2018` | `proportional_norm_0` | 33.59 +/- 5.18 | `mamba_norm_0` | **31.27 +/- 4.35** | 103.17 +/- 9.86 | ETAS |

The full set of test-start metrics for the selected configurations is:

| Dataset | Model | MAE | RMSE | CRPS | Coverage |
| --- | --- | ---: | ---: | ---: | ---: |
| `CB_HAB1a` | RTPP v2 `kernel` | 51.58 | 61.62 | 31.21 | 0.639 |
| `CB_HAB1a` | ETAS `mamba` | **14.82** | **19.14** | **10.32** | 0.694 |
| `CB_HAB1a` | Oracle | 53.45 | 83.50 | 36.19 | **0.861** |
| `CB_HAB4` | RTPP v2 `kernel_norm_0` | **99.05** | **122.73** | **86.50** | **0.889** |
| `CB_HAB4` | ETAS `kernel_norm_0` | 124.52 | 144.37 | 113.10 | 0.667 |
| `CB_HAB4` | Oracle | 146.26 | 166.06 | 144.93 | 0.000 |
| `PNR_1z` | RTPP v2 `proportional_norm_0` | **34.83** | **50.57** | **34.34** | **1.000** |
| `PNR_1z` | ETAS `mamba_norm_0` | 43.56 | 57.61 | 36.60 | 0.833 |
| `PNR_1z` | Oracle | 81.33 | 104.48 | 74.82 | 0.750 |
| `St1_2018` | RTPP v2 `proportional_norm_0` | 22.23 | 33.59 | 16.72 | 0.944 |
| `St1_2018` | ETAS `mamba_norm_0` | **17.95** | **31.27** | **14.97** | **1.000** |
| `St1_2018` | Oracle | 56.74 | 103.17 | 46.08 | 0.522 |

## Findings

### RTPP v2 and ETAS remain the operational models

- RTPP v2 is the better choice for `CB_HAB4` and `PNR_1z`.
- ETAS is clearly best for `CB_HAB1a` and marginally best for `St1_2018`.
- Relative to the selected RTPP v2 or ETAS configuration, Oracle is not the
  lowest-RMSE model on any dataset.
- Oracle is particularly weak on `PNR_1z` and `St1_2018`, where its test-start
  RMSE is roughly twice to more than three times the selected model's RMSE.

### Oracle is not an effective forecast upper bound in its current form

Oracle uses event-level feature construction and an injection-aware,
autoregressive sampling configuration:

```yaml
input_injection: true
oracle_sampling_mode: autoregressive
oracle_future_feature_names: [vm, sv, dTS, Mc]
```

This means it is not under exactly the same information assumptions as the
ordinary RTPP v2 and ETAS configurations. Nevertheless, its privileged feature
configuration does not produce better sliding-window forecasts here. It should
be treated as a diagnostic feature/injection baseline rather than as a model
selection upper bound until the feature and sampling path is validated.

### Oracle calibration is the weakest of the three

Oracle's average test-start coverage across the four datasets is 0.533. It is
well below the nominal 0.95 interval target and lower than the selected RTPP
v2 and ETAS configurations. The most serious case is `CB_HAB4`, where Oracle
has zero test-start coverage over the three available test windows.

The coverage results are noisy because the number of test-start windows differs
by dataset (`CB_HAB4` has only three windows per seed). However, the low
coverage across Oracle's other datasets, together with its high RMSE and CRPS,
indicates a broader sampling/calibration issue rather than only a small-sample
artifact.

### Likelihood and forecast metrics do not rank models identically

Oracle's test NLL is competitive on `PNR_1z` (`-6.3647`) but that run has a
test-start RMSE of 104.48, versus 50.57 for selected RTPP v2. Conversely,
RTPP v2 and ETAS frequently select `no_bg` as the best NLL configuration on
the two CB_HAB datasets while background-enabled variants provide much lower
sliding-window error.

Test NLL should therefore be used as a model-fit diagnostic, not as the sole
criterion for count-forecast model selection.

### Background modeling is still the strongest cross-experiment result

For RTPP v2 and ETAS, the best background configuration improves full-window
RMSE over `no_bg` in every dataset/model pairing. The reduction ranges from
32.6% to 52.1% for RTPP v2 and from 34.9% to 69.2% for ETAS. This supports
using a dataset-specific background model rather than `no_bg` for the final
forecasting workflow.

`norm_0` is not a global default: its impact changes with both dataset and
background architecture. It should remain a grid-search dimension instead of
being selected uniformly.

## Recommended Model Selection

| Dataset | Recommended model | Reason |
| --- | --- | --- |
| `CB_HAB1a` | ETAS + `mamba` | Dominates RMSE, MAE, and CRPS. |
| `CB_HAB4` | RTPP v2 + `kernel_norm_0` | Best test-start RMSE, MAE, CRPS, and coverage. |
| `PNR_1z` | RTPP v2 + `proportional_norm_0` | Best test-start error and calibrated coverage. |
| `St1_2018` | ETAS + `mamba_norm_0` | Lowest RMSE, MAE, and CRPS; RTPP v2 is a close fallback. |

Oracle should not be used as the final operational forecast model based on the
current results.

## Follow-Up Checks for Oracle

1. Verify that `oracle_future_feature_names` have the same time units,
   normalization, and availability at sliding-window inference as at training.
2. Check the alignment of `input_injection` time series with each forecast
   window, especially for `CB_HAB4`.
3. Audit `oracle_sampling_time_unit_minutes` and the generated forecast lengths
   relative to the one-day sliding duration.
4. Inspect the `CB_HAB4` windows individually to determine whether the zero
   coverage comes from underprediction, overprediction, or degenerate sampled
   sequences.
5. Evaluate broader predictive intervals or apply post-hoc calibration before
   making interval-based decisions from Oracle forecasts.
