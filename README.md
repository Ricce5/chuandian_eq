



## Installation

Before proceeding, ensure your GPU supports `flash_attn` and `mamba_ssm`.

1. Install the required dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
2. Install `flash_attn`:
    [Flash Attention Releases](https://github.com/Dao-AILab/flash-attention/releases)
3. Install `causal_conv1d`:
    [Causal Conv1D Releases](https://github.com/Dao-AILab/causal-conv1d/releases)
4. Install `mamba_ssm`:
    [Mamba Releases](https://github.com/state-spaces/mamba/releases)
5. Install the current package in editable mode:
    ```bash
    pip install -e .
    ```

## Training

Configuration files for training are available in the `configs/` directory.

### Models
- **Proposed Models**: `reg_mixer_attnpl_t`, `clf_mixer_attnpl_t`, `mixer_tpp`
- **Benchmark Models**: `etas`, `rtpp`

### Reproducibility
To ensure reproducibility, set the following environment variable:
```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### Training Command
To train the model, execute:
```bash
python main.py --model <model_name> --mode train --config <path_to_config_file>
python main.py --model <model_name> --mode train --config <path_to_config_file>
```

Replace `<model_name>` with one of the following:
- `reg_mixer_attnpl_t`
- `clf_mixer_attnpl_t`
- `mixer_tpp`
- `etas`
- `rtpp`

Replace `<path_to_config_file>` with the path to your configuration file.

## Testing

To test the model, use the following command:
```bash
python main.py --model <model_name> --mode test --config <path_to_config_file>
```

Replace `<model_name>` and `<path_to_config_file>` with the appropriate values for your setup.

```bash
python main.py --model <model_name> --mode test --config <path_to_config_file>
```

- If `--checkpoint` is not specified, the most recently trained model will be used by default.

## Data Augmentation: Magnitude Noise

To inject random noise into the input earthquake magnitudes during training (improves robustness), set the following in your config (e.g., `config/reg_mixer_attnpl_t.yaml`):

```yaml
# apply noise to input channel "Magnitude" only in training
mag_noise_std: 0.05       # noise scale; 0 disables
mag_noise_type: gaussian  # or: uniform
```

Notes:
- Noise is applied only on non-padded timesteps, inferred from arrival times.
- Validation and test remain untouched.

## Logs

To monitor training logs, use TensorBoard:

```bash
tensorboard --logdir <tensorboard_folder_in_checkpoint_directory> --port <port_number>
```

Replace `<tensorboard_folder_in_checkpoint_directory>` with the path to the TensorBoard logs directory and `<port_number>` with the desired port number.

## Notes
mamba-ssm has bugs, try to fix with:
- In `selective_state_update`, the value `0` has been replaced with `(0, 0)` to handle cases where the input is `None`.
- For `mamba2`, the operation `xBC.contiguous().transpose(1, 2)` now includes `contiguous` for improved memory layout handling.