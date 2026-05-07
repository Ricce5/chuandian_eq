from argparse import Namespace

import torch

from src.models.builders import ModelBuilder


def _args(use_double_precision: bool):
    return Namespace(
        model="etas_zhuang",
        tau_mean=1.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        bg_model=None,
        base_rate_init=0.26,
        productivity_K_init=0.11,
        productivity_alpha_e_init=2.302585092994046,
        fix_mu=False,
        fixed_mu_value=None,
        loss_reduction="none",
        etas_query_chunk_size=0,
        etas_history_chunk_size=16,
        etas_grad_checkpoint=False,
        etas_enforce_subcritical=True,
        etas_max_branching_ratio=0.98,
        etas_enforce_p_gt_one=True,
        etas_min_omori_p=1.001,
        etas_constraint_softness=1e-3,
        use_double_precision=use_double_precision,
    )


def test_etas_zhuang_builder_defaults_to_float32():
    model = ModelBuilder.by_name("etas_zhuang")()(_args(use_double_precision=False), torch.device("cpu"))
    assert model.log_c.dtype == torch.float32


def test_etas_zhuang_builder_supports_double_precision():
    model = ModelBuilder.by_name("etas_zhuang")()(_args(use_double_precision=True), torch.device("cpu"))
    assert model.log_c.dtype == torch.float64
