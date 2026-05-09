# em_eqf

A unified deep-learning framework for earthquake-catalog forecasting, covering:
- Classification and regression tasks (window-based samples)
- Temporal point process (TPP) tasks (event-sequence based)
- Training, testing, Optuna search, visualization, and evaluation workflows

## 1. Installation

### 1.1 Core dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 1.2 Optional acceleration dependencies
Install these only if your selected model/config requires them:
- `flash_attn`
- `mamba_ssm`
- `causal_conv1d`

Use the official installation guides and make sure versions match your CUDA/PyTorch environment.

## 2. Repository layout

```text
.
├── main.py                 # Main entry point (train/test/optuna)
├── forecasting.py          # Sampling and forecast visualization script
├── config/                 # YAML configurations
├── data/                   # Data directory (raw/processed)
├── notebooks/              # Jupyter notebooks for preprocessing, analysis, and plotting
├── checkpoints/            # Training artifacts
├── src/
│   ├── catalogs/           # Multi-catalog builders and registration
│   ├── data/               # Data loading, splitting, batching
│   ├── distributions/      # Distributions and mixture distributions
│   ├── features/           # Seismic feature engineering
│   ├── models/             # Model definitions and components
│   │   ├── core/           # BaseModel / TaskModel / TaskHead composition primitives
│   │   ├── adapters/       # Input adapters grouped by task/domain
│   │   ├── builders/       # ModelBuilder registry and model factory entries
│   │   ├── bg/             # Background models
│   │   ├── tpp/            # Temporal point process models
│   │   └── transformer/    # Transformer family implementations
│   ├── train/              # Training pipeline and train steps
│   └── utils/              # Utilities (logging, metrics, visualization, etc.)
└── tests/                  # Tests
```

## 3. Quick start

### 3.1 Train
```bash
python main.py --model mixer_tpp --mode train --config config/mixer_tpp.yaml
```

Set a custom output directory:
```bash
python main.py --model mixer_tpp --mode train --checkpoint_dir checkpoints/my_exp
```

### 3.2 Test
By default, test uses `best_model_{trial_index}.pth`:
```bash
python main.py \
  --model mixer_tpp \
  --mode test \
  --checkpoint_dir checkpoints/my_exp \
  --ckpt_select best \
  --trial_index 1
```

Test with an epoch checkpoint:
```bash
python main.py \
  --model mixer_tpp \
  --mode test \
  --checkpoint_dir checkpoints/my_exp \
  --ckpt_select epoch \
  --ckpt_epoch 10 \
  --trial_index 1
```

For classification, you can manually set a threshold:
```bash
python main.py --model clf_mixer_attnpl_t --mode test --threshold 0.5
```

### 3.3 Optuna search
```bash
python main.py --model mixer_tpp --mode optuna --config config/mixer_tpp.yaml
```

## 4. CLI arguments (main.py)

- `--mode`: `train` / `test` / `optuna`
- `--model`: Model name (must match `model` in config)
- `--config`: Config file path, default `config/<model>.yaml`
- `--checkpoint_dir`: Training output directory or test input directory
- `--trial_index`: Checkpoint index suffix (default `1`)
- `--ckpt_select`: `best` / `last` / `epoch`
- `--ckpt_epoch`: Required when `--ckpt_select epoch`
- `--threshold`: Classification test threshold (overrides checkpoint threshold)
- `--no_val_threshold`: Do not use threshold stored in checkpoint `val_metrics`

Notes:
- In `train`/`optuna`, if `--checkpoint_dir` is omitted, a new directory is created as `checkpoints/<model>_<timestamp>`
- In `test`, if `--checkpoint_dir` is omitted, the latest checkpoint directory for the model is selected automatically

## 5. Configuration system

- Config files are under `config/`
- YAML is loaded through `OmegaConf`
- Typical fields: `model`, `dataset`, `task_type`, optimizer/scheduler params, model architecture params
- Checkpoint-related fields:
  - `resume_path`: checkpoint to restore from
  - `load_specific_parts`: only load matching parameter names from the checkpoint
  - `freeze_parts`: freeze parameters whose names match these keywords
  - `freeze_loaded_only`: when `true`, only freeze parameters actually loaded from checkpoint; when `false`, freeze all matching parameters, including randomly initialized ones
  - `exclude_freeze_parts`: keywords excluded from freezing
- `freeze_parts` also works without `resume_path`: the matched randomly initialized parameters will be frozen directly

Recommended starter configs:
- `config/mixer_tpp.yaml`
- `config/clf_mixer_attnpl_t.yaml`
- `config/clf_rnn.yaml`
- `config/reg_mixer_attnpl_t.yaml`
- `config/reg_rnn.yaml`
- `config/etas.yaml`
- `config/etas_zhuang.yaml`

## 6. Data

### 6.1 Classification/Regression data
Common path: `data/<dataset>/raw/*.csv`

Main columns used by preprocessing:
- `t`
- `Magnitude`
- `Latitude`
- `Longitude`
- `Depth`
- `dt` (recomputed from `t` during preprocessing)

### 6.2 TPP data
TPP pipeline resolves catalogs through the registration system using `<dataset>-Standard`:
- Example: `dataset: ChuanDian` -> `ChuanDian-Standard`

List registered catalogs:
```bash
python - <<'PY'
import src.catalogs
from src.data.catalog import Catalog
print(sorted(Catalog.list_available()))
PY
```

## 7. Available models

Current registered models (from `ModelBuilder.list_available()`):

`btpp`, `classifier`, `classifier_se`, `classifier_stm`, `classifier_stm_s`, `classifier_tm_s`, `clf_attnpl`, `clf_attnpl_t`, `clf_mixer_attnpl_t`, `clf_rnn`, `clf_tm_attnpl`, `clf_tm_attnpl_t`, `clf_tm_cv_attnpl_t`, `etas`, `etas_zhuang`, `lstm`, `mhp`, `mixer_tpp`, `mtpp`, `nhpp`, `reg_attnpl`, `reg_mixer_attnpl_t`, `reg_rnn`, `rtpp`, `thp`, `thp_deltat`

## 8. Artifacts and logging

Typical files under `checkpoints/<exp>/`:
- `config.yaml` (runtime config snapshot)
- `run.log` (unified logger output)
- `best_model_<idx>.pth`
- `last_model_<idx>.pth`
- `epoch_<k>_model_<idx>.pth` (if periodic checkpoint saving is enabled)
- `tensorboard/`

Testing writes:
- `metrics_test_<...>.json`

TensorBoard:
```bash
tensorboard --logdir checkpoints/<exp>/tensorboard --port 6006
```

## 9. Forecast script

`forecasting.py` generates samples from a trained checkpoint and saves forecast visualizations.

Example:
```bash
python forecasting.py --checkpoint_dir checkpoints/mixer_tpp_YYYYMMDD-HHMMSS --ckpt_select best
```

## 10. Notebooks

Notebook resources are under `notebooks/` (project-relative path: `./notebooks`):

- `preprocessing*.ipynb`: preprocessing pipelines for different catalogs/datasets
- `classifier_baseline.ipynb`, `classifier_analysis.ipynb`: classification baseline and result analysis
- `regressor_analysis.ipynb`, `regression_plot.ipynb`: regression result analysis and plotting
- `tpp_analysis.ipynb`, `tpp_evaluating.ipynb`: TPP behavior analysis and evaluation
- `forecasting*.ipynb`: interactive forecasting workflows
- `b_t_estimation.ipynb`: time-varying b-value estimation
- Output figures and cached plotting data are organized in `notebooks/figs/` and `notebooks/figs_data/`

## 11. Testing

Run all tests:
```bash
pytest -q
```

Run selected groups:
```bash
pytest -q tests/features
pytest -q tests/data
pytest -q tests/model
```

## 12. Reproducibility hint

To improve CUDA reproducibility:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## 13. License

This project is licensed under the MIT License. See `LICENSE`.
