import pytest
import torch
from unittest.mock import patch

from src.data.dot_dict import DotDict
from src.models.bg import BGModel
import src.models.bg.gp_latent_bg as gp_latent_module
from src.models.bg.gp_latent_bg import GPyTorchGPLatentBGModel


pytest.importorskip("gpytorch")


class _DummySSM(torch.nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int):
        super().__init__()
        del d_model, d_state, d_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_device() -> torch.device:
    return torch.device("cpu")


def make_batch(
    device: torch.device,
    batch_size: int = 2,
    seq_len: int = 48,
    d_feature: int = 3,
) -> DotDict:
    time_series = torch.randn(batch_size, seq_len, d_feature, device=device)
    time_series_mask = torch.ones(batch_size, seq_len, device=device)
    time_series[1, -10:] = 0.0
    time_series_mask[1, -10:] = 0.0
    time_series_times = torch.linspace(0.0, 12.0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    arrival_times = torch.linspace(0.5, 11.5, seq_len // 3, device=device).unsqueeze(0).repeat(batch_size, 1)
    return DotDict(
        {
            "time_series": time_series,
            "time_series_mask": time_series_mask,
            "time_series_times": time_series_times,
            "arrival_times": arrival_times,
            "t_nll_start": torch.zeros(batch_size, device=device),
            "t_end": torch.full((batch_size,), 12.0, device=device),
            "nll_event_mask": torch.ones(batch_size, arrival_times.shape[1], device=device),
        }
    )


def make_model(device: torch.device, **kwargs) -> GPyTorchGPLatentBGModel:
    with patch.object(gp_latent_module, "Mamba", _DummySSM), patch.object(gp_latent_module, "Mamba2", _DummySSM):
        return GPyTorchGPLatentBGModel(device=device, **kwargs)


def test_gpytorch_gp_latent_bg_model_is_registered():
    assert BGModel.by_name("gp_latent_bg_gpytorch") is GPyTorchGPLatentBGModel


def test_gpytorch_gp_latent_bg_model_uses_gpytorch_kernel_parameters():
    model = make_model(
        get_device(),
        d_feature=3,
        d_model=8,
        d_latent=4,
        num_inducing=5,
    )
    assert hasattr(model, "gp_kernel_module")
    assert not hasattr(model, "log_gp_lengthscale")
    assert not hasattr(model, "log_gp_kernel_scale")
    assert torch.isfinite(model.gp_kernel_module.outputscale).all()
    assert torch.isfinite(model.gp_kernel_module.base_kernel.lengthscale).all()


def test_gpytorch_gp_latent_bg_model_is_random_in_train_mode():
    torch.manual_seed(0)
    device = get_device()
    batch = make_batch(device)
    model = make_model(
        device,
        d_feature=batch.time_series.shape[-1],
        d_model=8,
        d_latent=4,
        num_inducing=5,
    )
    model.train()

    intensity_a = model.intensity(batch)
    intensity_b = model.intensity(batch)

    assert intensity_a.shape == intensity_b.shape == (batch.time_series.shape[0], batch.arrival_times.shape[1])
    assert torch.all(intensity_a >= 0.0)
    assert torch.all(intensity_b >= 0.0)
    assert not torch.allclose(intensity_a, intensity_b)


def test_gpytorch_gp_latent_bg_model_is_deterministic_in_eval_by_default():
    torch.manual_seed(0)
    device = get_device()
    batch = make_batch(device)
    model = make_model(
        device,
        d_feature=batch.time_series.shape[-1],
        d_model=8,
        d_latent=4,
        num_inducing=5,
        stochastic_eval=False,
    )
    model.eval()

    intensity_a = model.intensity(batch)
    intensity_b = model.intensity(batch)

    assert torch.allclose(intensity_a, intensity_b)


def test_gpytorch_gp_latent_bg_model_nll_is_finite_and_tracks_kl():
    torch.manual_seed(0)
    device = get_device()
    batch = make_batch(device)
    model = make_model(
        device,
        d_feature=batch.time_series.shape[-1],
        d_model=8,
        d_latent=4,
        num_inducing=5,
        beta_kl=5e-3,
    )
    model.train()

    nll = model.nll(batch)
    kl = model.kl_term(batch)

    assert nll.shape == (batch.time_series.shape[0],)
    assert kl.shape == (batch.time_series.shape[0],)
    assert torch.isfinite(nll).all()
    assert torch.isfinite(kl).all()
    assert torch.all(kl >= 0.0)
    assert model.last_kl is not None
    assert torch.isfinite(model.last_kl).all()
