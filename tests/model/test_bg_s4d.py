import torch

from src.data.dot_dict import DotDict
from src.models.bg import BGModel
from src.models.bg.s4d import S4DBGModel


def make_batch(
    device: torch.device,
    batch_size: int = 2,
    seq_len: int = 24,
    d_feature: int = 3,
) -> DotDict:
    time_series = torch.randn(batch_size, seq_len, d_feature, device=device)
    time_series_mask = torch.ones(batch_size, seq_len, device=device)
    time_series[1, -4:] = 0.0
    time_series_mask[1, -4:] = 0.0
    time_series_times = torch.linspace(0.0, 6.0, seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
    arrival_times = torch.linspace(0.25, 5.75, seq_len // 3, device=device).unsqueeze(0).repeat(batch_size, 1)
    return DotDict(
        {
            "time_series": time_series,
            "time_series_mask": time_series_mask,
            "time_series_times": time_series_times,
            "arrival_times": arrival_times,
            "t_nll_start": torch.zeros(batch_size, device=device),
            "t_end": torch.full((batch_size,), 6.0, device=device),
            "nll_event_mask": torch.ones(batch_size, arrival_times.shape[1], device=device),
        }
    )


def test_s4d_bg_model_is_registered():
    assert BGModel.by_name("s4d") is S4DBGModel


def test_s4d_bg_scaled_intensity_and_context_shapes():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = make_batch(device)
    model = S4DBGModel(
        d_feature=batch.time_series.shape[-1],
        scale_init=100.0,
        d_model=8,
        d_state=16,
        dropout=0.0,
        activation="identity",
        use_output_linear=False,
        smooth_kernel_size=3,
        device=device,
    )

    scaled = model.scaled_intensity(batch.time_series)
    intensity = model.intensity(batch)
    traj_times, context = model.context_trajectory(batch)

    assert scaled.shape == (batch.time_series.shape[0], batch.time_series.shape[1], 1)
    assert intensity.shape == (batch.time_series.shape[0], batch.arrival_times.shape[1])
    assert torch.all(intensity >= 0.0)
    assert traj_times.shape == batch.time_series_times.shape
    assert context.shape == (batch.time_series.shape[0], batch.time_series.shape[1], 8)
    assert torch.allclose(context[1, -4:], torch.zeros_like(context[1, -4:]))


def test_s4d_bg_model_slow_branch_runs():
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = make_batch(device, seq_len=18, d_feature=2)
    model = S4DBGModel(
        d_feature=2,
        scale_init=100.0,
        d_model=4,
        d_state=8,
        dropout=0.0,
        activation="identity",
        use_output_linear=False,
        use_slow_branch=True,
        slow_kernel_type="mix",
        slow_kernel_size=8,
        smooth_kernel_size=None,
        device=device,
    )

    intensity = model.intensity(batch)
    integral = model.intensity_integral(batch)

    assert intensity.shape == (batch.time_series.shape[0], batch.arrival_times.shape[1])
    assert integral.shape == (batch.time_series.shape[0],)
    assert torch.isfinite(intensity).all()
    assert torch.isfinite(integral).all()

