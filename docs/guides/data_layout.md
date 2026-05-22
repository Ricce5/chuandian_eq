# Data Layout (Recommended)

## Goal
Keep source data in `raw/`, and store hashed catalog artifacts in `catalogs/`.
Do not use `source/`.

## Recommended Structure

For a concrete dataset (example: `FORGE2022`, `PNR_1z`, `PNR_2`):

```text
data/<dataset>/
  raw/                       # original source files
  processed/                 # *_eq_processed.csv, *_inj_*min_processed.csv, *_summary.json
  catalogs/                  # <hash>/catalog_cfg.json, metadata.pt, full_sequence.pt, norm_stats.pt
```

For grouped/family datasets (example: `PNR`, `SSFS`, `CooperBasin`):

```text
data/<family>/
  catalogs/                  # family-level split/cache artifacts
```

## Runtime Rule

- Source files: strictly from `raw/`.
- Catalog artifacts: strictly under `catalogs/`.
- No fallback to old mixed layout.

## Migration Commands (Example)

Use copy first, verify, then delete old files.

```bash
# all datasets: move hashed dirs from raw -> catalogs
for d in data/*; do
  [ -d "$d/raw" ] || continue
  mkdir -p "$d/catalogs"
  find "$d/raw" -maxdepth 1 -mindepth 1 -type d -regex '.*/[0-9a-f]{8}' -exec mv {} "$d/catalogs/" \;
done
```

## One-Command Reorganization

For newly added data, run:

```bash
python scripts/maintenance/reorganize_data_layout.py --data-root data
```

Preview only:

```bash
python scripts/maintenance/reorganize_data_layout.py --data-root data --dry-run
```

Only selected datasets:

```bash
python scripts/maintenance/reorganize_data_layout.py --data-root data --datasets FORGE2022 PNR_1z PNR_2
```
