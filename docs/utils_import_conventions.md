# Utils Import Conventions

Use these import paths for new code and notebook updates.

## Preferred modules

- Runtime helpers:
  - `from src.utils.runtime_utils import resolve_project_root, unwrap_compiled_model`
- RF notebook config helper:
  - `from config.config_loader import load_rf_config`
- Plot styling / figure export:
  - `from src.utils.plot_style import apply_publication_style, save_pub_figure, format_time_axis, style_axes`
- Sliding-window forecast helpers:
  - `from src.utils.forecast_sliding import ...`
  - or `from src.utils.forecast_eval import ...` when using full evaluation pipeline
- Sequence/catalog visualization:
  - `from src.utils.viz_sequences import ...`
- General analysis helpers:
  - `import src.utils.viz_analysis as analysis`
- Interpretability plotting:
  - `import src.utils.viz_interpretability as interp`

## Backward-compatibility notes

- `src.utils.forecast_eval_helpers` is kept as an alias layer.
- Legacy modules (`src.utils.visualization`, `src.utils.analysis`, `src.utils.interpretability`) still work, but new code should prefer `viz_*` entrypoints for clearer organization.

