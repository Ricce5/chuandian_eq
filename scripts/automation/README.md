## Automation Modules

This folder contains shared automation logic used by experiment runners under `scripts/`.

- `experiment_layout.py`: experiment folder/bootstrap helpers (workspace creation, snapshots, JSON writes, GPU parallel caps, resume summary loading).
- `grid_common.py`: shared grid-runner execution utilities (exp-config overrides, parsing, train/test task execution, parallel scheduling).
- `optuna_common.py`: shared Optuna profile runner utilities.

Compatibility wrappers remain:

- `scripts/grid_runner_common.py` re-exports `automation.grid_common`.
- `scripts/optuna_profiles_common.py` re-exports `automation.optuna_common`.

So existing imports and commands continue to work while new code can import from `automation`.

