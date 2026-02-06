
## Installation

Before proceeding, ensure your GPU supports `flash_attn` and `mamba_ssm`.

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Optional/accelerating libraries (install if relevant to your configuration):
- `flash_attn` — accelerated attention implementations (see https://github.com/Dao-AILab/flash-attention/releases)
- `mamba_ssm` / `causal_conv1d` — install per their repositories if used in your config

3. Install the package in editable mode (optional, for easier imports):
```bash
pip install -e .
```

## Project Overview

This repository implements a unified deep-learning forecasting framework for earthquake catalogs, including multiple temporal point process (TPP) sequence models and regression/classification models, along with training and evaluation pipelines.

Brief directory layout:
- `main.py`: Entry script — supports `train`, `test`, `optuna` modes.
- `config/`: Model and data YAML configs (e.g. `mixer_tpp.yaml`, `reg_mixer_attnpl_t.yaml`).
- `data/`: Raw and processed data (prepare data according to `src/data`).
- `src/`: Code for data processing, models, training, and evaluation.
- `checkpoints/`: Saved models and logs from training.

## Configuration

- All training/testing parameters are defined in YAML files under `config/`. Default convention: `config/<model>.yaml`.
- If `--config` is not provided, the program uses `config/{model}.yaml`.

Configurable items include learning rate, batch size, epochs, dataset name, random seed, and augmentation options.

## Quick Start

1) Train (creates `checkpoints/<model>_<timestamp>` by default):
```bash
python main.py --model mixer_tpp --mode train --config config/mixer_tpp.yaml
```
Specify a custom checkpoint directory:
```bash
python main.py --model mixer_tpp --mode train --config config/mixer_tpp.yaml --checkpoint_dir checkpoints/my_experiment
```

2) Test (load saved checkpoint):
```bash
python main.py --model mixer_tpp --mode test --config config/mixer_tpp.yaml --checkpoint_dir checkpoints/my_experiment --ckpt_select best --trial_index 1
```
Load by epoch:
```bash
python main.py --model mixer_tpp --mode test --config config/mixer_tpp.yaml --checkpoint_dir checkpoints/my_experiment --ckpt_select epoch --ckpt_epoch 10 --trial_index 1
```

3) Hyperparameter search (Optuna):
```bash
python main.py --model mixer_tpp --mode optuna --config config/mixer_tpp.yaml
```

## Common Arguments

- `--model`: Model name (must match `model` field in config).
- `--mode`: `train` / `test` / `optuna`.
- `--config`: Path to config file (default `config/<model>.yaml`).
- `--checkpoint_dir`: Directory to save/load training artifacts.
- `--ckpt_select`: `best`, `last`, or `epoch` (select checkpoint when testing).
- `--ckpt_epoch`: Epoch number when `--ckpt_select epoch` is used.

## Data & Augmentation

- Data is located in `data/`. Data preparation logic is in `src/data`.
- Example augmentation options in YAML:
```yaml
mag_noise_std: 0.05       # magnitude noise std for training; 0 disables
mag_noise_type: gaussian  # options: gaussian / uniform
```
Noise is typically applied only to non-padding timesteps; validation/test sets are unaffected.

## Logging & Visualization

- Training logs for TensorBoard are saved under `checkpoints/<exp>/tensorboard`.
```bash
tensorboard --logdir checkpoints/<exp>/tensorboard --port 6006
```
- Test metrics are saved as `metrics_test_*.json` in the experiment checkpoint directory.

## Reproducibility

For better numerical reproducibility (especially with CUDA), set:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

## Example Configs

Example configs are provided in `config/`: `mixer_tpp.yaml`, `reg_mixer_attnpl_t.yaml`, `clf_mixer_attnpl_t.yaml`, `etas.yaml`, `rtpp.yaml`, etc.

## Development & Contribution

- Run tests (if available):
```bash
pytest -q
```
- Follow code style and dependency specifications in `requirements.txt` and `setup.py`.

## Contact & License

For more information or to submit issues/PRs, open an Issue or submit a PR in the repository.

---

If you want this README translated back to Chinese or expanded with more examples (e.g., data format details, field explanations for configs), I can add that.

