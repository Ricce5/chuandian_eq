from argparse import Namespace

import math
import torch

from src.models.builders import ModelBuilder
from src.models.bg.base import BGModel


@BGModel.register("test_bg", overwrite=True)
class _TestBG(BGModel):
    def __init__(self, d_feature=1, scale_init=1.0, device=None, no_weight_decay=False):
        super().__init__(device=device, scale_init=scale_init, no_weight_decay=no_weight_decay)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            time_series.shape[0],
            time_series.shape[1],
            1,
            device=time_series.device,
            dtype=time_series.dtype,
        )


def _args(**overrides):
    base = dict(
        model="etas",
        tau_mean=1.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        bg_model=None,
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
    base.update(overrides)
    return Namespace(**base)


def test_etas_builder_reads_init_params_from_args():
    args = _args()
    model = ModelBuilder.by_name("etas")()(args, torch.device("cpu"))

    assert math.isclose(model.p.item(), args.omori_p_init, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(model.c.item(), args.omori_c_init, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(model.mu.item(), args.base_rate_init, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(model.k.item(), args.productivity_k_init, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(model.alpha.item(), args.productivity_alpha_init, rel_tol=1e-6, abs_tol=1e-8)


def test_etas_builder_honors_base_rate_without_bg_model():
    args = _args(base_rate_init=0.73)
    model = ModelBuilder.by_name("etas")()(args, torch.device("cpu"))
    assert math.isclose(model.mu.item(), 0.73, rel_tol=1e-6, abs_tol=1e-8)


def test_etas_builder_defaults_mu_to_zero_with_bg_model():
    args = Namespace(
        model="etas",
        tau_mean=1.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        bg_model="test_bg",
        bg_model_cfg={"d_feature": 1, "scale_init": 1.0},
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
    assert math.isclose(model.mu.item(), 0.0, rel_tol=0.0, abs_tol=1e-12)
