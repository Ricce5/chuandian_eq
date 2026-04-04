# utils Layout

`src/utils` contains cross-module helpers that are reused by data loading,
modeling, training, and evaluation code.

## Module Map

- `analysis.py`: prediction collection and t-SNE helper functions.
- `binary_focal_loss.py`: focal-loss wrapper used in classification training.
- `catalog_pathing.py`: centralized catalog/checkpoint/config path resolution.
- `catalog_tests.py`: statistical tests for catalog evaluation.
- `catalog_utils.py`: sequence split and catalog slicing helpers.
- `debug_utils.py`: tensor anomaly inspection helpers.
- `file_utils.py`: path, cache, and config file helpers.
- `forecast_eval.py`: full sliding-window forecast evaluation and plotting pipeline.
- `forecast_eval_helpers.py`: notebook-friendly forecast evaluation helper functions.
- `interp.py`: interpolation and integration on uniform time grids.
- `interpretability.py`: model interpretability and attribution utilities.
- `likelihood_curve_helpers.py`: cumulative likelihood/NLL helper functions for analysis notebooks.
- `logging_utils.py`: logging setup and logger utilities.
- `mask_utils.py`: attention and padding mask builders.
- `metrics.py`: metric computation and metric visualization.
- `notebook_helpers.py`: shared notebook setup, plotting style, and model unwrapping helpers.
- `registrable.py`: lightweight class registry mechanism.
- `tpp_experiments.py`: experiment-level dataset/model sampling helpers.
- `utils.py`: generic utilities (`set_seed`, time conversion helpers).
- `visualization.py`: sequence/catalog plotting utilities.

## Conventions

- Keep modules focused by concern; avoid mixing training logic into utils.
- Add reusable helpers here only if they are used by multiple modules.
- Avoid committing runtime artifacts (`__pycache__`, `*.pyc`).
