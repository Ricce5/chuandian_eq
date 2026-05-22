# Scripts Layout

`scripts/` is organized by responsibility so entrypoints are easier to discover and maintain.

## Directory Structure

- `scripts/run/`: experiment runners and orchestration scripts.
- `scripts/summarize/`: post-run metric summarization/report generation.
- `scripts/analyze/`: deeper analytical scripts and comparative studies.
- `scripts/maintenance/`: maintenance/backfill/reorganization helpers.
- `scripts/data/`: data preparation/build scripts.
- `scripts/diagnostics/`: debugging and state-comparison tools.
- `scripts/tuning/`: targeted tuning utilities.
- `scripts/automation/`: shared Python modules used by runner/summarizer scripts.

## Naming Conventions

- Use verb-first, snake_case names:
  - `run_*` for run/orchestration.
  - `summarize_*` for result aggregation.
  - `analyze_*` for deeper analysis.
  - `backfill_*`, `reorganize_*`, `compare_*`, `build_*`, `tune_*` for specific utility intent.
- Put each script in the matching functional folder above.

## Backward Compatibility

Legacy root-level script paths (for example `scripts/run_clf_mf_tf_grid.py`) are kept as thin wrappers that delegate to the new locations. Existing commands continue to work.

Preferred new invocation style:

```bash
python scripts/run/run_clf_mf_tf_grid.py ...
python scripts/summarize/summarize_clf_mf_tf_grid.py ...
python scripts/analyze/analyze_clf_trainhp_sweep.py ...
```
