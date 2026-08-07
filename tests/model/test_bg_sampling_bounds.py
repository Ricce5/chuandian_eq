import pytest
import torch

from src.models.bg.proportional import ProportionalBGModel


def _cached_model():
    model = ProportionalBGModel(
        d_feature=1,
        scale_init=1.0,
        device=torch.device("cpu"),
    )
    times = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    values = torch.ones((1, 3, 1), dtype=torch.float32)
    model.cache_batch(values, times)
    return model, times


def test_sample_nhpp_inverse_clamps_float_endpoint_residue():
    model, times = _cached_model()
    cache_end = times[0, -1]
    just_above_end = torch.nextafter(
        cache_end,
        torch.tensor(float("inf"), dtype=cache_end.dtype),
    )
    t0 = torch.tensor([1.0], dtype=torch.float32)
    dt = just_above_end.reshape(1) - t0

    sampled = model.sample_nhpp_inverse(B=1, t0=t0, dt=dt)

    assert sampled.shape == (1,)
    assert torch.isfinite(sampled).all()


def test_sample_nhpp_inverse_still_rejects_material_overrun():
    model, times = _cached_model()
    t0 = torch.tensor([1.0], dtype=torch.float32)
    dt = torch.tensor([1.01], dtype=torch.float32)

    with pytest.raises(ValueError, match="out of cached range"):
        model.sample_nhpp_inverse(B=1, t0=t0, dt=dt)
