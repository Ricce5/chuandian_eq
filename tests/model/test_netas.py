from argparse import Namespace

import numpy as np
import pytest
import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.builders import ModelBuilder
from src.models.tpp.common.recurrent_blocks import RNNTPPBackbone
from src.models.tpp.netas import MambaNETASEncoder, NETAS
from src.train.model_routing import get_model_family, get_train_step_module


class _ZeroEncoder(torch.nn.Module):
    def __init__(self, context_size: int) -> None:
        super().__init__()
        self.context_size = int(context_size)

    def get_parent_context(self, batch: Batch) -> torch.Tensor:
        return batch.inter_times.new_zeros(batch.batch_size, batch.seq_len, self.context_size)

    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype):
        return torch.zeros(batch_size, 1, self.context_size, device=device, dtype=dtype)

    def step_event(
        self,
        *,
        inter_time: torch.Tensor,
        magnitude: torch.Tensor,
        prev_state,
    ):
        context = torch.zeros(
            inter_time.shape[0],
            1,
            self.context_size,
            device=inter_time.device,
            dtype=inter_time.dtype,
        )
        return context, prev_state


class _CountingEncoder(torch.nn.Module):
    def __init__(self, context_size: int) -> None:
        super().__init__()
        self.context_size = int(context_size)
        self.step_calls = 0

    def get_parent_context(self, batch: Batch) -> torch.Tensor:
        return batch.inter_times.new_zeros(batch.batch_size, batch.seq_len, self.context_size)

    def initial_state(self, *, batch_size: int, device: torch.device, dtype: torch.dtype):
        return {"batch_size": batch_size, "device": device, "dtype": dtype}

    def step_event(
        self,
        *,
        inter_time: torch.Tensor,
        magnitude: torch.Tensor,
        prev_state,
    ):
        self.step_calls += 1
        context = torch.zeros(
            inter_time.shape[0],
            1,
            self.context_size,
            device=inter_time.device,
            dtype=inter_time.dtype,
        )
        return context, prev_state


class _CountingRNNEncoder(RNNTPPBackbone):
    def __init__(self, context_size: int) -> None:
        super().__init__(
            context_size=context_size,
            tau_mean=1.0,
            mag_mean=2.0,
            rnn_type="GRU",
            input_magnitude=True,
        )
        self.step_calls = 0

    def step(self, rnn_input: torch.Tensor, prev_hidden: torch.Tensor):
        self.step_calls += 1
        return super().step(rnn_input, prev_hidden)


class _StubBGSampler:
    def __init__(self) -> None:
        self.calls = 0

    def sample_nhpp_inverse(self, *, B, t0, dt, sample_sequence, mu):
        self.calls += 1
        assert sample_sequence is True
        assert mu == 0.0
        return [torch.empty((0,), dtype=t0.dtype, device=t0.device) for _ in range(int(B))]


class _CachingBGSampler:
    def __init__(self) -> None:
        self.calls = 0
        self.cache_calls = 0
        self.ts_batch_cache = None

    def cache_batch(self, time_series, time_series_times, cache_lambda=True):
        self.cache_calls += 1
        self.ts_batch_cache = type(
            "Cache",
            (),
            {"time_series": time_series, "time_series_times": time_series_times},
        )()

    def sample_nhpp_inverse(self, *, B, t0, dt, sample_sequence, mu):
        assert self.ts_batch_cache is not None
        self.calls += 1
        return [torch.empty((0,), dtype=t0.dtype, device=t0.device) for _ in range(int(B))]


def _manual_model(
    *,
    query_chunk_size: int = 0,
    max_history_events: int = 0,
    history_time_window: float | None = None,
) -> NETAS:
    model = NETAS(
        event_encoder=_ZeroEncoder(context_size=4),
        context_size=4,
        basis_family="exponential",
        num_basis=1,
        basis_rates=torch.tensor([2.0], dtype=torch.float32),
        base_rate_init=0.5,
        productivity_alpha_init=0.0,
        productivity_bias_init=0.0,
        eta_max=0.8,
        richter_b=1.0,
        mag_completeness=3.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        loss_reduction="none",
        query_chunk_size=query_chunk_size,
        history_chunk_size=3,
        max_history_events=max_history_events,
        history_time_window=history_time_window,
    )
    with torch.no_grad():
        model.productivity_head.weight.zero_()
        model.mixture_head.weight.zero_()
    return model


def test_netas_bounded_productivity_requires_subcritical_cap():
    with pytest.raises(ValueError, match="productivity_mode='bounded'"):
        NETAS(
            event_encoder=_ZeroEncoder(context_size=4),
            context_size=4,
            basis_family="exponential",
            num_basis=1,
            basis_rates=torch.tensor([1.0], dtype=torch.float32),
            productivity_mode="bounded",
            eta_max=1.1,
            richter_b=1.0,
            mag_completeness=2.0,
            mag_max=8.0,
        )


def test_netas_softplus_productivity_can_exceed_one():
    seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([5.0], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    model = NETAS(
        event_encoder=_ZeroEncoder(context_size=4),
        context_size=4,
        basis_family="exponential",
        num_basis=1,
        basis_rates=torch.tensor([1.0], dtype=torch.float32),
        productivity_alpha_init=1.0,
        productivity_bias_init=0.0,
        productivity_mode="softplus",
        eta_max=10.0,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        loss_reduction="none",
    )

    eta = model.parent_parameters(batch)["eta"]

    assert eta[0, 0] > 1.0


def test_netas_branching_penalty_contributes_to_total_loss():
    seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    model = NETAS(
        event_encoder=_ZeroEncoder(context_size=4),
        context_size=4,
        basis_family="exponential",
        num_basis=1,
        basis_rates=torch.tensor([1.0], dtype=torch.float32),
        base_rate_init=0.5,
        productivity_alpha_init=0.0,
        productivity_bias_init=0.0,
        eta_max=0.8,
        branching_penalty_weight=2.0,
        branching_penalty_target=0.1,
        richter_b=1.0,
        mag_completeness=3.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        loss_reduction="none",
    )

    out = model.nll_loss(batch, reduction="none", return_dict=True)

    assert "branching_penalty" in out
    assert out["branching_penalty"][0] > 0.0
    torch.testing.assert_close(
        out["total"],
        out["time"] + 2.0 * out["branching_penalty"],
        rtol=1e-5,
        atol=1e-7,
    )


def test_netas_exponential_nll_matches_manual_formula():
    seq = Sequence(
        inter_times=torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0, 3.0], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    model = _manual_model()

    got = model.nll_loss(batch, reduction="none")

    mu = torch.tensor(0.5)
    eta = torch.tensor(0.4)
    beta = torch.tensor(2.0)
    log_part = -torch.log(mu) - torch.log(mu + eta * beta * torch.exp(-beta * torch.tensor(0.5)))
    integral = mu * torch.tensor(2.0)
    integral = integral + eta * (1.0 - torch.exp(-beta * torch.tensor(1.0)))
    integral = integral + eta * (1.0 - torch.exp(-beta * torch.tensor(0.5)))
    expected = log_part + integral

    torch.testing.assert_close(got, expected.unsqueeze(0), rtol=1e-5, atol=1e-7)


def test_netas_chunked_nll_matches_full_nll():
    torch.manual_seed(7)
    seq1 = Sequence(
        inter_times=torch.rand(10) + 0.1,
        t_start=0.0,
        t_nll_start=0.2,
        mag=torch.rand(9) + 2.0,
    )
    seq2 = Sequence(
        inter_times=torch.rand(12) + 0.1,
        t_start=0.0,
        t_nll_start=0.3,
        mag=torch.rand(11) + 2.0,
    )
    batch = Batch.from_list([seq1, seq2])

    full_model = _manual_model(query_chunk_size=0)
    chunked_model = _manual_model(query_chunk_size=4)
    chunked_model.load_state_dict(full_model.state_dict())

    full_nll = full_model.nll_loss(batch, reduction="none")
    chunked_nll = chunked_model.nll_loss(batch, reduction="none")
    torch.testing.assert_close(full_nll, chunked_nll, rtol=1e-5, atol=1e-7)


def test_netas_max_history_events_limits_trigger_intensity():
    seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0, 3.0, 3.0], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    model = _manual_model(max_history_events=1)

    got = model.trigger_intensity(batch, t_query=torch.tensor([[3.0]], dtype=torch.float32))

    eta = torch.tensor(0.4)
    beta = torch.tensor(2.0)
    expected = eta * beta * torch.exp(-beta * torch.tensor(1.0))
    torch.testing.assert_close(got, expected.reshape(1, 1), rtol=1e-5, atol=1e-7)


def test_netas_history_time_window_limits_trigger_intensity():
    seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0, 3.0], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    model = _manual_model(history_time_window=1.5)

    got = model.trigger_intensity(batch, t_query=torch.tensor([[3.0]], dtype=torch.float32))

    eta = torch.tensor(0.4)
    beta = torch.tensor(2.0)
    expected = eta * beta * torch.exp(-beta * torch.tensor(1.0))
    torch.testing.assert_close(got, expected.reshape(1, 1), rtol=1e-5, atol=1e-7)


def test_netas_max_history_events_limits_trigger_integral():
    seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0, 3.0], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    model = _manual_model(max_history_events=1)

    got = model.trigger_integral_between(
        batch,
        t_start=torch.tensor([0.0], dtype=torch.float32),
        t_end=torch.tensor([3.0], dtype=torch.float32),
    )

    eta = torch.tensor(0.4)
    beta = torch.tensor(2.0)
    expected = 2.0 * eta * (1.0 - torch.exp(-beta * torch.tensor(1.0)))
    torch.testing.assert_close(got, expected.reshape(1, 1), rtol=1e-5, atol=1e-7)


def test_netas_learnable_lomax_basis_receives_gradients():
    seq = Sequence(
        inter_times=torch.tensor([0.4, 0.6, 0.8, 0.9], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.3, 2.7, 3.1], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    model = NETAS(
        event_encoder=_ZeroEncoder(context_size=4),
        context_size=4,
        basis_family="lomax",
        num_basis=2,
        basis_scales=torch.tensor([0.5, 2.0], dtype=torch.float32),
        basis_shapes=torch.tensor([0.6, 1.2], dtype=torch.float32),
        basis_learnable="per_basis",
        basis_max_log_deviation=0.5,
        basis_learn_shapes=True,
        base_rate_init=0.2,
        productivity_alpha_init=0.0,
        productivity_bias_init=0.0,
        eta_max=0.8,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        loss_reduction="mean",
        head_init_std=0.0,
    )

    torch.testing.assert_close(model.basis.effective_scales, model.basis.scales)
    torch.testing.assert_close(model.basis.effective_shapes, model.basis.shapes)

    loss = model.nll_loss(batch, reduction="mean")
    loss.backward()

    assert model.basis.scale_log_delta.grad is not None
    assert model.basis.shape_log_delta.grad is not None
    assert model.basis.scale_log_delta.grad.abs().sum() > 0.0
    assert model.basis.shape_log_delta.grad.abs().sum() > 0.0


def test_netas_head_init_allows_encoder_gradients_at_first_step():
    torch.manual_seed(11)
    seq = Sequence(
        inter_times=torch.tensor([1.0, 0.5, 0.7, 0.8], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5, 3.1, 2.9], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])
    encoder = RNNTPPBackbone(
        context_size=8,
        tau_mean=1.0,
        mag_mean=2.0,
        input_magnitude=True,
    )
    model = NETAS(
        event_encoder=encoder,
        context_size=8,
        basis_family="exponential",
        num_basis=3,
        base_rate_init=0.1,
        productivity_alpha_init=0.5,
        productivity_bias_init=-0.5,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        loss_reduction="mean",
        head_init_std=1e-2,
    )

    loss = model.nll_loss(batch, reduction="mean")
    loss.backward()
    encoder_grad = sum(
        0.0 if param.grad is None else float(param.grad.abs().sum().item())
        for param in model.event_encoder.parameters()
    )

    assert encoder_grad > 0.0


def test_netas_builder_constructs_rnn_model():
    args = Namespace(
        model="netas",
        d_model=8,
        tau_mean=1.0,
        mag_mean=3.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        num_components=3,
        bg_model=None,
        base_rate_init=0.2,
        rnn_type="GRU",
        num_rnn_layers=1,
        rnn_dropout=0.0,
        input_magnitude=True,
        time_max=20.0,
        loss_reduction="none",
        netas_basis_family="exponential",
        netas_num_basis=3,
        netas_eta_max=0.9,
        netas_productivity_alpha_init=0.8,
        netas_productivity_mode="softplus",
        netas_basis_learnable="per_basis",
        netas_basis_max_log_deviation=0.75,
        netas_head_init_std=0.02,
        netas_branching_penalty_weight=0.3,
        netas_branching_penalty_target=0.85,
        netas_max_history_events=128,
        netas_history_time_window=12.5,
    )
    model = ModelBuilder.by_name("netas")()(args, torch.device("cpu"))

    assert isinstance(model, NETAS)
    assert model.num_basis == 3
    assert model.context_size == 8
    assert model.productivity_mode == "softplus"
    assert model.branching_penalty_weight == pytest.approx(0.3)
    assert model.branching_penalty_target == pytest.approx(0.85)
    assert model.basis.learnable_mode == "per_basis"
    assert model.basis.max_log_deviation == pytest.approx(0.75)
    assert model.basis.rate_log_delta.shape == torch.Size([3])
    assert model.max_history_events == 128
    assert model.history_time_window == pytest.approx(12.5)
    assert get_model_family("netas") == "tpp"
    assert get_train_step_module("netas") == "tpp_train_step"


def test_netas_builder_constructs_mamba_model():
    args = Namespace(
        model="netas",
        d_model=8,
        tau_mean=1.0,
        mag_mean=3.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        num_components=3,
        bg_model=None,
        base_rate_init=0.2,
        rnn_dropout=0.0,
        input_magnitude=True,
        time_max=20.0,
        loss_reduction="none",
        netas_encoder_type="mamba",
        netas_basis_family="exponential",
        netas_num_basis=3,
        netas_eta_max=0.9,
        netas_productivity_alpha_init=0.8,
        netas_mamba_d_state=4,
        netas_mamba_d_conv=2,
        netas_mamba_expand=2,
        netas_mamba_use_conv=True,
        netas_max_inference_len=128,
    )
    model = ModelBuilder.by_name("netas")()(args, torch.device("cpu"))
    seq = Sequence(
        inter_times=torch.tensor([1.0, 0.5, 0.5], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0, 3.0], dtype=torch.float32),
    )
    batch = Batch.from_list([seq])

    assert isinstance(model, NETAS)
    assert isinstance(model.event_encoder, MambaNETASEncoder)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
        batch = batch.to(torch.device("cuda"))
        out = model.nll_loss(batch, reduction="none")
        assert out.shape == torch.Size([1])


def test_netas_sample_can_return_empty_sequences():
    model = NETAS(
        event_encoder=_ZeroEncoder(context_size=4),
        context_size=4,
        basis_family="exponential",
        num_basis=2,
        basis_rates=torch.tensor([1.0, 5.0], dtype=torch.float32),
        base_rate_init=1e-4,
        productivity_alpha_init=0.0,
        productivity_bias_init=0.0,
        eta_max=0.8,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        fix_mu=True,
        fixed_mu_value=0.0,
        loss_reduction="none",
    )

    batch = model.sample(batch_size=2, duration=3.0, random_state=0)
    sequences = batch.to_list()

    assert len(sequences) == 2
    for seq in sequences:
        assert seq.num_events == 0
        torch.testing.assert_close(seq.inter_times, torch.tensor([3.0]))


def test_netas_sample_reuses_past_history_encoding_across_batch():
    encoder = _CountingEncoder(context_size=4)
    model = NETAS(
        event_encoder=encoder,
        context_size=4,
        basis_family="exponential",
        num_basis=1,
        basis_rates=torch.tensor([1.0], dtype=torch.float32),
        base_rate_init=1e-4,
        productivity_alpha_init=0.0,
        productivity_bias_init=-100.0,
        eta_max=0.8,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        fix_mu=True,
        fixed_mu_value=0.0,
        loss_reduction="none",
    )
    past_seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5, 2.6], dtype=torch.float32),
    )

    samples = model.sample(
        batch_size=5,
        duration=2.0,
        past_seq=past_seq,
        random_state=0,
        return_sequences=True,
    )

    assert len(samples) == 5
    assert encoder.step_calls == past_seq.num_events


def test_netas_sample_batches_background_sampling_call():
    bg_model = _StubBGSampler()
    model = NETAS(
        event_encoder=_ZeroEncoder(context_size=4),
        context_size=4,
        basis_family="exponential",
        num_basis=1,
        basis_rates=torch.tensor([1.0], dtype=torch.float32),
        base_rate_init=1e-4,
        productivity_alpha_init=0.0,
        productivity_bias_init=-100.0,
        eta_max=0.8,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        bg_model=bg_model,
        fix_mu=True,
        fixed_mu_value=0.0,
        loss_reduction="none",
    )
    past_seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5], dtype=torch.float32),
    )

    model.sample(
        batch_size=4,
        duration=2.0,
        past_seq=past_seq,
        random_state=0,
        return_sequences=True,
    )

    assert bg_model.calls == 1


def test_netas_rnn_sample_steps_batch_in_parallel():
    encoder = _CountingRNNEncoder(context_size=4)
    model = NETAS(
        event_encoder=encoder,
        context_size=4,
        basis_family="exponential",
        num_basis=1,
        basis_rates=torch.tensor([1.0], dtype=torch.float32),
        base_rate_init=1e-4,
        productivity_alpha_init=0.0,
        productivity_bias_init=-100.0,
        eta_max=0.8,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        fix_mu=True,
        fixed_mu_value=50.0,
        loss_reduction="none",
    )

    samples = model.sample(
        batch_size=4,
        duration=0.2,
        random_state=0,
        return_sequences=True,
    )

    event_counts = [seq.num_events for seq in samples]
    assert sum(event_counts) > max(event_counts)
    assert encoder.step_calls == max(event_counts)


def test_netas_sample_can_auto_cache_background_from_full_sequence():
    bg_model = _CachingBGSampler()
    model = NETAS(
        event_encoder=_ZeroEncoder(context_size=4),
        context_size=4,
        basis_family="exponential",
        num_basis=1,
        basis_rates=torch.tensor([1.0], dtype=torch.float32),
        base_rate_init=1e-4,
        productivity_alpha_init=0.0,
        productivity_bias_init=-100.0,
        eta_max=0.8,
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=8.0,
        device=torch.device("cpu"),
        bg_model=bg_model,
        fix_mu=True,
        fixed_mu_value=0.0,
        loss_reduction="none",
    )
    full_seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0, 1.0, 2.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5, 2.6, 2.7], dtype=torch.float32),
        time_series=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32),
        time_series_times=torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0], dtype=torch.float32),
    )
    past_seq = full_seq.get_subsequence(0.0, 2.0, reset_t_nll_to_end=True)

    samples = model.sample(
        batch_size=3,
        duration=2.0,
        past_seq=past_seq,
        bg_cache_seq=full_seq,
        random_state=0,
        return_sequences=True,
    )

    assert len(samples) == 3
    assert bg_model.cache_calls == 1
    assert bg_model.calls == 1


def test_netas_sample_accepts_array_backed_initial_events():
    model = _manual_model()
    seq = model._sample_single_sequence(
        t_start=0.0,
        t_end=3.0,
        past_seq=None,
        rng=np.random.default_rng(0),
        max_length=100,
        history_cache=model._prepare_history_sampling_cache(
            t_start=0.0,
            t_end=3.0,
            past_seq=None,
        ),
        preset_initial_events=(
            np.asarray([0.5, 1.5], dtype=np.float64),
            np.asarray([3.0, 3.1], dtype=np.float32),
        ),
    )

    torch.testing.assert_close(
        seq.arrival_times,
        torch.tensor([0.5, 1.5], dtype=seq.arrival_times.dtype),
    )
