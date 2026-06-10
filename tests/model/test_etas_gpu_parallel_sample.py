import pytest
import torch

from src.models.tpp.etas import ETAS


class _FixedBGSampler:
    def __init__(self, samples):
        self.samples = samples
        self.calls = 0
        self.kwargs = None

    def sample_nhpp_inverse(self, *, B, t0, dt, sample_sequence, mu):
        self.calls += 1
        self.kwargs = {
            "B": B,
            "t0": t0,
            "dt": dt,
            "sample_sequence": sample_sequence,
            "mu": mu,
        }
        return self.samples


def _quiet_etas(*, bg_model=None):
    return ETAS(
        base_rate_init=1e-12,
        productivity_k_init=1e-12,
        productivity_alpha_init=1.0,
        device=torch.device("cpu"),
        bg_model=bg_model,
        fix_mu=True,
        fixed_mu_value=0.0,
        enforce_subcritical=False,
        enforce_p_gt_one=False,
    )


def test_sample_gpu_parallel_uses_background_model_samples():
    bg_model = _FixedBGSampler(
        [
            torch.tensor([0.2], dtype=torch.float64),
            torch.tensor([0.3, 0.7], dtype=torch.float64),
        ]
    )
    model = _quiet_etas(bg_model=bg_model)

    sequences = model.sample_gpu_parallel(
        batch_size=2,
        duration=1.0,
        t_start=0.0,
        return_sequences=True,
        random_state=123,
    )

    assert bg_model.calls == 1
    assert bg_model.kwargs["B"] == 2
    assert bg_model.kwargs["sample_sequence"] is True
    assert bg_model.kwargs["mu"] == 0.0
    torch.testing.assert_close(
        sequences[0].arrival_times,
        torch.tensor([0.2], dtype=sequences[0].arrival_times.dtype),
    )
    torch.testing.assert_close(
        sequences[1].arrival_times,
        torch.tensor([0.3, 0.7], dtype=sequences[1].arrival_times.dtype),
    )


def test_sample_gpu_parallel_enforces_max_length_per_sequence():
    bg_model = _FixedBGSampler(
        [
            torch.tensor([0.2], dtype=torch.float64),
            torch.tensor([], dtype=torch.float64),
        ]
    )
    model = _quiet_etas(bg_model=bg_model)

    with pytest.raises(RuntimeError, match="max_length=0"):
        model.sample_gpu_parallel(
            batch_size=2,
            duration=1.0,
            t_start=0.0,
            max_length=0,
            return_sequences=True,
            random_state=123,
        )
