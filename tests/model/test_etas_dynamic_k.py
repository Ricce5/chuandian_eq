from argparse import Namespace

import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.bg.base import BGModel
from src.models.builders import ModelBuilder
from src.models.tpp.etas import ETAS
from src.models.tpp.etas_zhuang import ETASZhuang


@BGModel.register("test_bg_ramp", overwrite=True)
class _TestBGRamp(BGModel):
    def __init__(self, d_feature=1, scale_init=1.0, device=None, no_weight_decay=False):
        super().__init__(device=device, scale_init=scale_init, no_weight_decay=no_weight_decay)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        return time_series[..., :1]


def _build_batch() -> Batch:
    seq = Sequence(
        inter_times=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0, 3.0], dtype=torch.float32),
        time_series=torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32),
        time_series_times=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32),
    )
    return Batch.from_list([seq])


def test_etas_k_at_adds_injection_dependent_component():
    batch = _build_batch()
    k_model = BGModel.by_name("test_bg_ramp")(d_feature=1, scale_init=1.0, device=torch.device("cpu"))
    model = ETAS(
        productivity_k_init=0.5,
        productivity_alpha_init=1.0,
        device=torch.device("cpu"),
        k_model=k_model,
    )

    k_hist = model.k_at(batch, t_query=batch.arrival_times)
    expected = torch.tensor([[1.5, 2.5, 2.5]], dtype=k_hist.dtype)
    torch.testing.assert_close(k_hist, expected)


def test_etas_zhuang_K_at_adds_injection_dependent_component():
    batch = _build_batch()
    k_model = BGModel.by_name("test_bg_ramp")(d_feature=1, scale_init=1.0, device=torch.device("cpu"))
    model = ETASZhuang(
        productivity_K_init=0.25,
        productivity_alpha_e_init=0.0,
        device=torch.device("cpu"),
        k_model=k_model,
    )

    K_hist = model.K_at(batch, t_query=batch.arrival_times)
    expected = torch.tensor([[1.25, 2.25, 2.25]], dtype=K_hist.dtype)
    torch.testing.assert_close(K_hist, expected)


def test_etas_builder_builds_optional_k_model():
    args = Namespace(
        model="etas",
        tau_mean=1.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        bg_model=None,
        k_model="test_bg_ramp",
        k_model_cfg={"d_feature": 1, "scale_init": 1.0},
        base_rate_init=0.41,
        omori_p_init=1.23,
        omori_c_init=0.07,
        productivity_k_init=0.013,
        productivity_alpha_init=0.83,
        fix_mu=False,
        fixed_mu_value=None,
        loss_reduction="none",
        etas_query_chunk_size=0,
        etas_history_chunk_size=0,
        etas_grad_checkpoint=False,
        etas_enforce_subcritical=False,
        etas_enforce_p_gt_one=False,
    )

    model = ModelBuilder.by_name("etas")()(args, torch.device("cpu"))
    assert model.k_model is not None
    assert model.k_model.__class__.__name__ == _TestBGRamp.__name__
