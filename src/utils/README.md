# utils Layout

`src/utils` contains cross-module helpers that are reused by data loading,
modeling, training, and evaluation code.

## Module Map

- `analysis.py`: prediction collection and t-SNE helper functions.
- `binary_focal_loss.py`: focal-loss wrapper used in classification training.
- `bootstrap_ci.py`: structured bootstrap API (`BootstrapConfig`) with iid/block bootstrap, classification/regression metrics, and paired multi-model CI utilities.
- `bootstrap_presets.py`: reusable paired-bootstrap parameter presets with helper methods to build `BootstrapConfig` for curves and metrics.
- `catalog_pathing.py`: centralized catalog/checkpoint/config path resolution.
- `catalog_tests.py`: statistical tests for catalog evaluation.
- `catalog_utils.py`: sequence split and catalog slicing helpers.
- `debug_utils.py`: tensor anomaly inspection helpers.
- `file_utils.py`: path, cache, and config file helpers.
- `forecast_eval.py`: full sliding-window forecast evaluation and plotting pipeline.
- `plot_style.py`: shared plotting style/time-axis/save helpers.
- `forecast_sliding.py`: shared sliding-window forecast helper exports (delegates to `forecast_eval.py`).
- `forecast_eval_helpers.py`: backward-compatible alias to `forecast_sliding.py`.
- `interp.py`: interpolation and integration on uniform time grids.
- `interpretability.py`: model interpretability and attribution utilities.
- `likelihood_curve_helpers.py`: cumulative likelihood/NLL helper functions for analysis notebooks.
- `logging_utils.py`: logging setup and logger utilities.
- `mask_utils.py`: attention and padding mask builders.
- `metrics.py`: metric computation and metric visualization.
- `registrable.py`: lightweight class registry mechanism.
- `runtime_utils.py`: runtime helpers (`resolve_project_root`, `unwrap_compiled_model`).
- `tpp_experiments.py`: experiment-level dataset/model sampling helpers.
- `utils.py`: generic utilities (`set_seed`, time conversion helpers); re-exports selected runtime helpers.
- `viz_sequences.py`: clear-named exports for sequence/catalog visualization helpers.
- `viz_analysis.py`: clear-named exports for analysis helpers (`predict_all`, `tsne_scatter`).
- `viz_interpretability.py`: clear-named exports for interpretability plotting helpers.
- `visualization.py`: sequence/catalog plotting utilities.

## Conventions

- Keep modules focused by concern; avoid mixing training logic into utils.
- Add reusable helpers here only if they are used by multiple modules.
- Avoid committing runtime artifacts (`__pycache__`, `*.pyc`).
