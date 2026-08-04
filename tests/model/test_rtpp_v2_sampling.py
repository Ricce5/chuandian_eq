from types import SimpleNamespace

import pytest
import torch

import src
from src.models.bg.proportional import ProportionalBGModel
from src.models.tpp.common.recurrent_blocks import RNNTPPBackbone
from src.models.tpp.common.sequence_ops import (
    build_sample_batch,
    evaluate_compensator_from_model,
)
from src.models.tpp.recurrent.model_v2 import RecurrentTPPV2
from src.models.tpp.recurrent.sampling import RecurrentTPPSamplingMixin


def _make_rtpp_v2(*, bg_model=None, time_use_bg_context=False, b_use_bg_context=False):
    args = SimpleNamespace(
        num_components=1,
        time_max=100.0,
        richter_b_mle=1.0,
        mag_completeness=0.0,
        mag_max=5.0,
        predict_b=True,
        loss_reduction="none",
        time_use_bg_context=time_use_bg_context,
        b_use_bg_context=b_use_bg_context,
    )
    backbone = RNNTPPBackbone(
        context_size=4,
        tau_mean=1.0,
        mag_mean=2.0,
        rnn_type="GRU",
        input_magnitude=True,
    )
    return RecurrentTPPV2(
        args,
        backbone=backbone,
        hypernet_time=torch.nn.Linear(4, 3),
        hypernet_mag=torch.nn.Linear(4, 1),
        device=torch.device("cpu"),
        bg_model=bg_model,
    )


def test_hidden_after_events_excludes_terminal_survival_slot():
    torch.manual_seed(7)
    backbone = RNNTPPBackbone(
        context_size=4,
        tau_mean=1.0,
        mag_mean=2.0,
        rnn_type="GRU",
        input_magnitude=True,
    )
    seq = src.data.Sequence(
        torch.tensor([1.0, 2.0, 3.0]),
        t_start=10.0,
        mag=torch.tensor([2.0, 2.5]),
    )
    batch = src.data.Batch.from_list([seq])
    features = backbone.build_features(batch)
    _, expected_hidden = backbone.rnn(features[:, : batch.end_idx.item()])

    restored_hidden = backbone.get_hidden_after_events(batch)
    _, hidden_with_terminal_zero = backbone.get_context_and_hidden(batch)

    assert torch.allclose(restored_hidden, expected_hidden)
    assert not torch.allclose(hidden_with_terminal_zero, expected_hidden)


def test_sampling_time_context_uses_last_event_not_forecast_boundary():
    class _BG(torch.nn.Module):
        context_dim = 1

        def __init__(self):
            super().__init__()
            self.ts_batch_cache = object()
            self.last_query = None

        def _get_bg_context(self, batch, query_times, *, clamp, left_endpoint):
            del batch, clamp, left_endpoint
            self.last_query = query_times.detach().clone()
            return query_times.unsqueeze(-1).to(dtype=torch.float32)

    class _Fuse(torch.nn.Module):
        def forward(self, context, bg_context):
            del bg_context
            return context

    bg = _BG()
    model = _make_rtpp_v2(bg_model=bg, time_use_bg_context=True)
    model.time_context_fuse = _Fuse()
    current_state = torch.zeros(2, 1, 4)
    t_last_event = torch.tensor([5.0, 7.0])

    model._get_sampling_time_context(
        current_state=current_state,
        t_last_event=t_last_event,
        lower_bound=torch.tensor([3.0, 4.0]),
    )

    assert torch.equal(bg.last_query.squeeze(-1), t_last_event)


def test_sampling_b_context_uses_last_event_query_time():
    class _BG(torch.nn.Module):
        context_dim = 1

        def __init__(self):
            super().__init__()
            self.ts_batch_cache = object()
            self.last_query = None

        def _get_bg_context(self, batch, query_times, *, clamp, left_endpoint):
            del batch, clamp, left_endpoint
            self.last_query = query_times.detach().clone()
            return query_times.unsqueeze(-1).to(dtype=torch.float32)

    class _Fuse(torch.nn.Module):
        def forward(self, context, bg_context):
            del bg_context
            return context

    bg = _BG()
    model = _make_rtpp_v2(bg_model=bg, b_use_bg_context=True)
    model.b_context_fuse = _Fuse()
    t_last_event = torch.tensor([5.0, 7.0])
    out = model._get_sampling_b_context(
        current_state=torch.zeros(2, 1, 4),
        t_last_event=t_last_event,
    )

    assert out.shape == (2, 1, 4)
    assert torch.equal(bg.last_query.squeeze(-1), t_last_event)


def test_sample_next_inter_time_caps_background_query_to_remaining_window():
    class _Dist:
        def sample(self):
            return torch.full((2, 1), 10.0)

    class _BG:
        def __init__(self):
            self.received_dt = None

        def sample_nhpp_inverse(self, batch_size, *, t0, dt):
            assert batch_size == 2
            del t0
            self.received_dt = dt.clone()
            return dt

    class _Sampler(RecurrentTPPSamplingMixin):
        pass

    sampler = _Sampler()
    sampler.bg_model = _BG()
    out = sampler.sample_next_inter_time(
        _Dist(),
        t_last_event=torch.tensor([1.0, 2.0]),
        max_inter_time=torch.tensor([0.5, 1.5]),
    )

    assert torch.equal(sampler.bg_model.received_dt, torch.tensor([0.5, 1.5]))
    assert torch.equal(out.squeeze(-1), torch.tensor([0.5, 1.5]))


def test_competing_trigger_and_background_samples_total_hazard():
    class _ExpTriggerDist:
        def sample(self):
            return torch.distributions.Exponential(torch.tensor(2.0)).sample((20_000, 1))

    class _Sampler(RecurrentTPPSamplingMixin):
        pass

    torch.manual_seed(19)
    bg = ProportionalBGModel(d_feature=1, scale_init=1.0, device=torch.device("cpu"))
    bg.cache_batch(
        torch.ones(1, 1_001, 1),
        torch.linspace(0.0, 100.0, 1_001).unsqueeze(0),
    )
    sampler = _Sampler()
    sampler.bg_model = bg
    tau = sampler.sample_next_inter_time(
        _ExpTriggerDist(),
        t_last_event=torch.zeros(20_000),
    ).squeeze(-1)

    assert tau.mean().item() == pytest.approx(1.0 / 3.0, abs=0.02)
    assert (tau > 0.5).float().mean().item() == pytest.approx(
        torch.exp(torch.tensor(-1.5)).item(),
        abs=0.02,
    )


def test_nhpp_inverse_rejects_windows_outside_cached_background_range():
    bg = ProportionalBGModel(d_feature=1, scale_init=1.0, device=torch.device("cpu"))
    bg.cache_batch(
        torch.ones(1, 11, 1),
        torch.linspace(0.0, 10.0, 11).unsqueeze(0),
    )

    with pytest.raises(ValueError, match="out of cached range"):
        bg.sample_nhpp_inverse(1, t0=9.0, dt=2.0)


def test_build_sample_batch_uses_absolute_arrival_times():
    batch = build_sample_batch(
        inter_times=torch.tensor([[1.0, 10.0]]),
        t_start=10.0,
        t_end=15.0,
        device=torch.device("cpu"),
    )

    assert torch.equal(batch.arrival_times, torch.tensor([[11.0, 15.0]]))
    assert torch.equal(batch.to_list()[0].arrival_times, torch.tensor([11.0]))


def test_sample_accepts_an_empty_history_sequence():
    torch.manual_seed(3)
    model = _make_rtpp_v2()
    model.eval()
    empty_history = src.data.Sequence(
        torch.tensor([1.0]),
        t_start=0.0,
        mag=torch.empty(0),
    )

    samples = model.sample(
        batch_size=2,
        duration=0.5,
        past_seq=empty_history,
        return_sequences=True,
    )

    assert len(samples) == 2
    assert all(sample.t_start == pytest.approx(1.0) for sample in samples)


def test_total_compensator_includes_background_and_absolute_start_time():
    class _Dist:
        def log_survival(self, x):
            return -2.0 * x

    class _BG:
        ts_batch_cache = object()

        @staticmethod
        def intensity_integral_between(batch, t_start, t_end):
            del batch
            return 3.0 * (t_end - t_start)

    class _Model:
        device = torch.device("cpu")
        bg_model = _BG()

        @staticmethod
        def get_context(batch):
            return torch.zeros(batch.batch_size, batch.seq_len, 1)

        @staticmethod
        def get_inter_time_dist(context):
            del context
            return _Dist()

    seq = src.data.Sequence(torch.tensor([1.0, 2.0]), t_start=10.0)
    grid, compensator = evaluate_compensator_from_model(
        model=_Model(),
        sequence=seq,
        num_grid_points=3,
    )

    assert torch.allclose(compensator, 5.0 * (grid - 10.0), atol=1e-3)
