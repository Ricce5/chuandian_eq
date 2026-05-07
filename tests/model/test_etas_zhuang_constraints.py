import math

import torch

from src.models.tpp.etas_zhuang import ETASZhuang


def test_etas_zhuang_branching_ratio_is_capped_by_design():
    model = ETASZhuang(
        device=torch.device("cpu"),
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        enforce_subcritical=True,
        max_branching_ratio=0.92,
    )

    model.set_params(K=1e9, alpha_e=0.7 * math.log(10.0))
    assert model.branching_ratio < 0.92 + 1e-6


def test_etas_zhuang_p_is_enforced_above_one():
    model = ETASZhuang(
        device=torch.device("cpu"),
        enforce_p_gt_one=True,
        min_omori_p=1.01,
    )
    model.set_params(p=1.000001)
    assert model.p.item() > 1.0
    assert model.p.item() >= 1.01 - 1e-6


def test_etas_zhuang_unconstrained_mode_keeps_legacy_behavior():
    model = ETASZhuang(
        device=torch.device("cpu"),
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        enforce_subcritical=False,
        enforce_p_gt_one=False,
    )
    model.set_params(K=0.7, alpha_e=0.99 * math.log(10.0), p=1.02)
    assert math.isclose(model.K.item(), 0.7, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(model.p.item(), 1.02, rel_tol=1e-6, abs_tol=1e-8)
    assert model.branching_ratio > 1.0
