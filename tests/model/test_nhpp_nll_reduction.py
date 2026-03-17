from argparse import Namespace
import math

import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.bg.base import BGModel
from src.models.tpp.nhpp import NHPP


class _ConstantBGModel(BGModel):
    def __init__(self, rate: float, device: torch.device):
        super().__init__(device=device, scale_init=rate)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(time_series[..., :1])


class _NegativeBGModel(BGModel):
    def __init__(self, raw_value: float, device: torch.device):
        super().__init__(device=device, scale_init=1.0)
        self.raw_value = torch.nn.Parameter(torch.tensor(float(raw_value), device=device))

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(time_series[..., :1]) * self.raw_value


def _build_batch() -> Batch:
    seq = Sequence(
        inter_times=torch.tensor([0.5, 1.0, 0.5, 1.0]),
        t_start=0.0,
        t_nll_start=1.0,
        mag=torch.tensor([2.0, 2.1, 2.2]),
        time_series=torch.zeros(4, 1),
        time_series_times=torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    return Batch.from_list([seq])


def test_bg_model_nll_respects_nll_window():
    batch = _build_batch()
    rate = 2.0
    model = _ConstantBGModel(rate=rate, device=torch.device("cpu"))

    raw_nll = model.nll(batch)

    span = (batch.t_end - batch.t_nll_start).item()
    num_events = int(batch.nll_event_mask.sum().item())
    expected = rate * span - num_events * math.log(rate)

    torch.testing.assert_close(raw_nll, torch.tensor([expected], dtype=raw_nll.dtype))


def test_nhpp_nll_reduction_modes_match_expected_values():
    batch = _build_batch()
    rate = 2.0
    args = Namespace(loss_reduction="per_time")
    bg_model = _ConstantBGModel(rate=rate, device=torch.device("cpu"))
    model = NHPP(args, device=torch.device("cpu"), bg_model=bg_model)

    legacy = model.nll_loss(batch)
    explicit = model.nll_loss(batch, reduction="per_time")
    raw = model.nll_loss(batch, reduction="none")
    out = model.nll_loss(batch, reduction="per_time", return_dict=True)

    span = batch.t_end - batch.t_nll_start

    torch.testing.assert_close(legacy, explicit)
    torch.testing.assert_close(explicit, raw / span)
    torch.testing.assert_close(out["bg"], explicit)
    torch.testing.assert_close(out["total"], explicit)


def test_bg_model_nll_keeps_event_gradient_when_raw_intensity_is_negative():
    batch = _build_batch()
    model = _NegativeBGModel(raw_value=-1.0, device=torch.device("cpu"))

    loss = model.nll(batch).sum()
    loss.backward()

    assert model.raw_value.grad is not None
    assert torch.isfinite(model.raw_value.grad)
    assert model.raw_value.grad.item() < 0.0
