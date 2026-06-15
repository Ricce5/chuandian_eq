import types

import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.tpp.recurrent.model_v1 import RecurrentTPP
from src.models.tpp.recurrent.model_v2 import RecurrentTPPV2


class _FakeInterTimeDist:
    def __init__(self, log_like: torch.Tensor):
        self.log_like = log_like

    def log_hazard(self, inter_times: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(inter_times)


class _FakeBG:
    def __init__(self, bg_nll: torch.Tensor):
        self.bg_nll = bg_nll

    def nll_change(
        self,
        batch,
        log_h_intensity: torch.Tensor,
        eps: float = 1e-10,
        include_kl: bool = False,
    ) -> torch.Tensor:
        return self.bg_nll.to(device=log_h_intensity.device, dtype=log_h_intensity.dtype)


def _batch() -> Batch:
    seq = Sequence(
        inter_times=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32),
        mag=torch.tensor([2.0, 2.5], dtype=torch.float32),
    )
    return Batch.from_list([seq])


def _install_common_stubs(model, *, trigger_nll: torch.Tensor, bg_nll: torch.Tensor):
    log_like = -trigger_nll
    dist = _FakeInterTimeDist(log_like)

    model.reduction = "none"
    model.bg_model = _FakeBG(bg_nll)
    model.get_inter_time_dist = types.MethodType(lambda self, context: dist, model)
    model.time_log_likelihood = types.MethodType(
        lambda self, **kwargs: log_like.to(
            device=kwargs["batch"].inter_times.device,
            dtype=kwargs["batch"].inter_times.dtype,
        ),
        model,
    )


def test_rtpp_time_nll_includes_background_change():
    batch = _batch()
    trigger_nll = torch.tensor([3.0])
    bg_nll = torch.tensor([2.0])

    model = object.__new__(RecurrentTPP)
    _install_common_stubs(model, trigger_nll=trigger_nll, bg_nll=bg_nll)
    model.get_context = types.MethodType(
        lambda self, batch: torch.zeros(batch.batch_size, batch.seq_len, 1),
        model,
    )
    model.weights = {
        "bg_weight": 1.0,
        "bg_kl_weight": 1.0,
        "bg_norm_weight": 0.0,
    }

    out = RecurrentTPP.nll_loss(model, batch, return_dict=True, reduction="none")

    assert torch.allclose(out["bg"], bg_nll)
    assert torch.allclose(out["time"], trigger_nll + bg_nll)
    assert torch.allclose(out["total"], trigger_nll + bg_nll)


def test_rtpp_v2_time_nll_includes_background_change():
    batch = _batch()
    trigger_nll = torch.tensor([3.0])
    bg_nll = torch.tensor([2.0])

    model = object.__new__(RecurrentTPPV2)
    _install_common_stubs(model, trigger_nll=trigger_nll, bg_nll=bg_nll)
    model.get_context = types.MethodType(
        lambda self, batch: torch.zeros(batch.batch_size, batch.seq_len, 1),
        model,
    )
    model._get_time_context = types.MethodType(lambda self, context, batch: context, model)
    model.predict_b = False
    model.weights = {
        "time_weight": 1.0,
        "mag_weight": 0.0,
        "b_weight": 0.0,
        "b_smooth_weight": 0.0,
        "bg_weight": 1.0,
        "bg_kl_weight": 1.0,
        "bg_norm_weight": 0.0,
    }

    out = RecurrentTPPV2.nll_loss(model, batch, return_dict=True, reduction="none")

    assert torch.allclose(out["bg"], bg_nll)
    assert torch.allclose(out["time"], trigger_nll + bg_nll)
    assert torch.allclose(out["total"], trigger_nll + bg_nll)
