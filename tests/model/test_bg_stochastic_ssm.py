import torch

from src.data.dot_dict import DotDict
from src.models.bg.stochastic_ssm import StochasticSSMBGModel


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(device: torch.device, batch_size: int = 2, seq_len: int = 48, d_feature: int = 3) -> DotDict:
    time_series = torch.randn(batch_size, seq_len, d_feature, device=device)
    time_series_times = torch.linspace(0.0, 12.0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    arrival_times = torch.linspace(0.5, 11.5, seq_len // 3, device=device).unsqueeze(0).repeat(batch_size, 1)
    return DotDict(
        {
            "time_series": time_series,
            "time_series_times": time_series_times,
            "arrival_times": arrival_times,
            "t_nll_start": torch.zeros(batch_size, device=device),
            "t_end": torch.full((batch_size,), 12.0, device=device),
            "nll_event_mask": torch.ones(batch_size, arrival_times.shape[1], device=device),
        }
    )


def test_stochastic_ssm_bg_model_is_random_in_train_mode():
    torch.manual_seed(0)
    device = get_device()
    batch = make_batch(device)

    model = StochasticSSMBGModel(
        d_feature=batch.time_series.shape[-1],
        d_model=8,
        d_state=8,
        noise_scale_init=0.2,
        device=device,
    )
    model.train()

    intensity_a = model.intensity(batch)
    intensity_b = model.intensity(batch)

    assert intensity_a.shape == intensity_b.shape == (batch.time_series.shape[0], batch.arrival_times.shape[1])
    assert torch.all(intensity_a >= 0.0)
    assert torch.all(intensity_b >= 0.0)
    assert not torch.allclose(intensity_a, intensity_b)


def test_stochastic_ssm_bg_model_is_deterministic_in_eval_by_default():
    torch.manual_seed(0)
    device = get_device()
    batch = make_batch(device)

    model = StochasticSSMBGModel(
        d_feature=batch.time_series.shape[-1],
        d_model=8,
        d_state=8,
        noise_scale_init=0.2,
        stochastic_eval=False,
        device=device,
    )
    model.eval()

    intensity_a = model.intensity(batch)
    intensity_b = model.intensity(batch)

    assert torch.allclose(intensity_a, intensity_b)


def test_stochastic_ssm_bg_model_nll_variants_are_finite():
    torch.manual_seed(0)
    device = get_device()
    batch = make_batch(device)

    model = StochasticSSMBGModel(
        d_feature=batch.time_series.shape[-1],
        d_model=8,
        d_state=8,
        noise_scale_init=0.15,
        device=device,
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
