# utils Layout

`src/utils` contains cross-module helpers that are reused by data loading,
modeling, training, and evaluation code.

## Module Map

- `analysis.py`: prediction collection and t-SNE helper functions.
- `binary_focal_loss.py`: focal-loss wrapper used in classification training.
- `catalog_tests.py`: statistical tests for catalog evaluation.
- `catalog_utils.py`: sequence split and catalog slicing helpers.
- `debug_utils.py`: tensor anomaly inspection helpers.
- `file_utils.py`: path, cache, and config file helpers.
- `interp.py`: interpolation and integration on uniform time grids.
- `interpretability.py`: model interpretability and attribution utilities.
- `mask_utils.py`: attention and padding mask builders.
- `metrics.py`: metric computation and metric visualization.
- `registrable.py`: lightweight class registry mechanism.
- `utils.py`: generic utilities (`set_seed`, time conversion helpers).
- `visualization.py`: sequence/catalog plotting utilities.

## Conventions

- Keep modules focused by concern; avoid mixing training logic into utils.
- Add reusable helpers here only if they are used by multiple modules.
- Avoid committing runtime artifacts (`__pycache__`, `*.pyc`).
