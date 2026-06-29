import math

import torch

from src.data.sequence import Sequence
from src.utils.likelihood_curve_helpers import cumulative_log_likelihood_curve


class _ConstantInterTimeDist:
    def __init__(self, trigger_rate: float):
        self.trigger_rate = trigger_rate

    def log_prob(self, inter_times: torch.Tensor) -> torch.Tensor:
        rate = torch.as_tensor(self.trigger_rate, device=inter_times.device, dtype=inter_times.dtype)
        return torch.log(rate) - rate * inter_times

    def log_survival(self, inter_times: torch.Tensor) -> torch.Tensor:
        return -self.trigger_rate * inter_times

    def log_hazard(self, inter_times: torch.Tensor) -> torch.Tensor:
        rate = torch.as_tensor(self.trigger_rate, device=inter_times.device, dtype=inter_times.dtype)
        return torch.log(rate).expand_as(inter_times)


class _ConstantBackgroundModel:
    def __init__(self, background_rate: float):
        self.background_rate = background_rate

    def intensity(self, batch, t_query: torch.Tensor | None = None) -> torch.Tensor:
        target = batch.arrival_times if t_query is None else t_query
        return torch.full_like(target, self.background_rate)

    def intensity_integral_between(self, ts_batch, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        return self.background_rate * (t_end - t_start)


class _ConstantRecurrentBGModel(torch.nn.Module):
    def __init__(self, trigger_rate: float, background_rate: float):
        super().__init__()
        self._device_marker = torch.nn.Parameter(torch.tensor(0.0))
        self.inter_time_dist = _ConstantInterTimeDist(trigger_rate)
        self.bg_model = _ConstantBackgroundModel(background_rate)

    def get_context(self, batch):
        return torch.zeros(batch.batch_size, batch.seq_len, 1, device=batch.arrival_times.device)

    def get_inter_time_dist(self, context):
        return self.inter_time_dist


def test_recurrent_bg_time_curve_includes_background_change():
    trigger_rate = 2.0
    background_rate = 1.0
    sequence = Sequence(torch.tensor([1.0, 2.0, 3.0]), t_start=0.0, t_nll_start=0.0)
    model = _ConstantRecurrentBGModel(trigger_rate=trigger_rate, background_rate=background_rate)

    curve, meta = cumulative_log_likelihood_curve(model, sequence, component="time", device=torch.device("cpu"))

    expected_trigger_ll = 2 * math.log(trigger_rate) - trigger_rate * (sequence.t_end - sequence.t_start)
    expected_time_ll = 2 * math.log(trigger_rate + background_rate) - (
        trigger_rate + background_rate
    ) * (sequence.t_end - sequence.t_start)

    assert meta["curve_method"] == "recurrent_total_fast"
    assert math.isclose(curve["cum_log_likelihood_trigger"][-1], expected_trigger_ll, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(curve["cum_log_likelihood_time"][-1], expected_time_ll, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(curve["cum_log_likelihood"][-1], expected_time_ll, rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(curve["cum_log_likelihood_total"][-1], expected_time_ll, rel_tol=0.0, abs_tol=1e-6)
    assert not math.isclose(curve["cum_log_likelihood"][-1], expected_trigger_ll, rel_tol=0.0, abs_tol=1e-6)
