import torch
from unittest.mock import patch

from config.config_loader import load_args_from_yaml
from src.data.dot_dict import DotDict
from src.models.bg import BGModel
import src.models.bg.gp_latent_bg as gp_latent_module
from src.models.bg.gp_latent_bg import GPLatentBGModel


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


def make_model(device: torch.device, **kwargs) -> GPLatentBGModel:
    with patch.object(gp_latent_module, "Mamba", _DummySSM), patch.object(gp_latent_module, "Mamba2", _DummySSM):
        return GPLatentBGModel(device=device, **kwargs)


def test_gp_latent_bg_model_is_registered():
    assert BGModel.by_name("gp_latent_bg") is GPLatentBGModel


def test_gp_latent_bg_model_is_random_in_train_mode():
    torch.manual_seed(0)
    device = get_device()
    batch = make_batch(device)

    model = make_model(
        device,
        d_feature=batch.time_series.shape[-1],
        d_model=8,
        d_latent=4,
        num_inducing=5,
        beta_kl=1e-3,
    )
    model.train()

    intensity_a = model.intensity(batch)
    intensity_b = model.intensity(batch)

    assert intensity_a.shape == intensity_b.shape == (batch.time_series.shape[0], batch.arrival_times.shape[1])
    assert torch.all(intensity_a >= 0.0)
    assert torch.all(intensity_b >= 0.0)
    assert not torch.allclose(intensity_a, intensity_b)


def test_gp_latent_bg_model_is_deterministic_in_eval_by_default():
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


def test_gp_latent_bg_model_keeps_zero_input_zero_output():
    torch.manual_seed(0)
    device = get_device()
    batch_size, seq_len, d_feature = 2, 24, 3

    batch = DotDict(
        {
            "time_series": torch.zeros(batch_size, seq_len, d_feature, device=device),
            "time_series_mask": torch.ones(batch_size, seq_len, device=device),
            "time_series_times": torch.linspace(0.0, 6.0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1),
        }
    )

    model = make_model(
        device,
        d_feature=d_feature,
        d_model=8,
        d_latent=4,
        num_inducing=5,
    )
    model.eval()

    intensity_traj = model.intensity_trajectory(batch)

    assert torch.allclose(intensity_traj, torch.zeros_like(intensity_traj))


def test_gp_latent_bg_model_nll_variants_are_finite_and_track_kl():
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

    log_h_intensity = torch.zeros_like(batch.arrival_times, device=device)

    nll = model.nll(batch)
    nll_change = model.nll_change(batch, log_h_intensity)
    integral = model.intensity_integral(batch)

    assert nll.shape == (batch.time_series.shape[0],)
    assert nll_change.shape == (batch.time_series.shape[0],)
    assert integral.shape == (batch.time_series.shape[0],)
    assert torch.isfinite(nll).all()
    assert torch.isfinite(nll_change).all()
    assert torch.isfinite(integral).all()
    assert model.last_kl is not None
    assert model.last_kl.shape == (batch.time_series.shape[0],)
    assert torch.isfinite(model.last_kl).all()
    assert torch.all(model.last_kl >= 0.0)


def test_gp_latent_bg_model_uses_multiple_mc_samples_in_train_nll():
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
        mc_samples_train=3,
        mc_samples_eval=1,
    )
    model.train()

    with patch.object(model, "_sample_inducing", wraps=model._sample_inducing) as sample_spy:
        nll = model.nll(batch)

    assert nll.shape == (batch.time_series.shape[0],)
    assert torch.isfinite(nll).all()
    assert sample_spy.call_count == 3


def test_etas_config_uses_gp_latent_bg_and_builds_bg_model():
    torch.manual_seed(0)
    device = get_device()
    args = load_args_from_yaml("./config/etas.yaml")

    assert args.bg_model == "gp_latent_bg"

    with patch.object(gp_latent_module, "Mamba", _DummySSM), patch.object(gp_latent_module, "Mamba2", _DummySSM):
        model = BGModel.by_name(args.bg_model)(**args.bg_model_cfg, device=device)

    batch = make_batch(device, d_feature=args.bg_model_cfg.d_feature)
    model.eval()
    intensity = model.intensity(batch)

    assert isinstance(model, GPLatentBGModel)
    assert model.num_inducing == args.bg_model_cfg.num_inducing
    assert intensity.shape == (batch.time_series.shape[0], batch.arrival_times.shape[1])
    assert torch.isfinite(intensity).all()
