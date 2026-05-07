import torch
from omegaconf import OmegaConf

from src.models.builders import ModelBuilder


def _make_args(model_name: str, *, task_type: str):
    stats = {
        "tau_min": 0.01,
        "tau_max": 10.0,
        "tau_mean": 1.0,
        "mag_completeness": 2.5,
        "b_value": 1.0,
    }
    args = {
        "model": model_name,
        "task_type": task_type,
        "stats": stats,
        "features_input_keys": ["mag", "log_inter_times"],
        "rnn_type": "lstm",
        "d_model": 16,
        "num_rnn_layers": 2,
        "rnn_dropout": 0.1,
        "bidirectional": False,
        "mlp_out": 1,
        "mlp_dropout": 0.1,
        "mlp_hdw": [16],
        "use_mag_cdf": False,
    }
    return OmegaConf.create(args)


def _make_dummy_batch(batch_size: int = 4, seq_len: int = 7):
    x = torch.zeros(batch_size, seq_len, 7, dtype=torch.float32)
    x[:, :, 0] = torch.arange(seq_len).float().unsqueeze(0).repeat(batch_size, 1) + 1.0
    x[:, :, 1] = torch.linspace(0.1, 1.0, steps=seq_len).unsqueeze(0).repeat(batch_size, 1)
    x[:, :, 2] = 3.0
    x[:, :, 3] = 30.0
    x[:, :, 4] = 120.0
    x[:, :, 5] = 10.0
    x[:, :, 6] = 0.5

    x[:, -2:, :] = 0.0
    x[:, 0, 0] = 1.0
    return x


def test_clf_rnn_forward_shape():
    args = _make_args("clf_rnn", task_type="classification")
    model = ModelBuilder.by_name(args.model)()(args, torch.device("cpu"))
    x = _make_dummy_batch()
    y = model(x)
    assert y.shape == (x.size(0),)


def test_reg_rnn_forward_shape():
    args = _make_args("reg_rnn", task_type="regression")
    model = ModelBuilder.by_name(args.model)()(args, torch.device("cpu"))
    x = _make_dummy_batch()
    y = model(x)
    assert y.shape == (x.size(0),)
