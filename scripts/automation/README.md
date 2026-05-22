## Automation Modules

This folder contains shared automation logic used by experiment runners under `scripts/` (primarily `scripts/run/`).

- `experiment_layout.py`: experiment folder/bootstrap helpers (workspace creation, snapshots, JSON writes, GPU parallel caps, resume summary loading).
- `grid_common.py`: shared grid-runner execution utilities (exp-config overrides, parsing, train/test task execution, parallel scheduling).
- `optuna_common.py`: shared Optuna profile runner utilities.
- `grid_script_utils.py`: shared runner-script helpers (run-name normalization, grid-point expansion, load-strategy parsing, encoder-load strategy apply).
- `summary_script_utils.py`: shared summarizer helpers (seed suffix parse, summary dedupe, cfg-path resolve, tf/mf group-mode inference, per-run row collection template and common group-key builders).

Compatibility wrappers remain for imports:

- `scripts/grid_runner_common.py` re-exports `automation.grid_common`.
- `scripts/optuna_profiles_common.py` re-exports `automation.optuna_common`.

Root-level legacy entrypoint wrappers (e.g. `scripts/run/run_clf_mf_tf_grid.py`) also delegate to the new grouped folders so existing commands continue to work.
