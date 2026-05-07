import math

import torch

from src.models.tpp.etas import ETAS


def test_etas_effective_branching_ratio_is_capped_by_design():
    model = ETAS(
        device=torch.device("cpu"),
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        enforce_subcritical=True,
        max_branching_ratio=0.92,
        effective_branching_t_max=1e4,
    )

    model.set_params(k=1e9, alpha=0.7)
    assert model.effective_branching_ratio(t_max=1e4) < 0.92 + 1e-6


def test_etas_p_is_enforced_above_one():
    model = ETAS(
        device=torch.device("cpu"),
        enforce_p_gt_one=True,
        min_omori_p=1.01,
    )
    model.set_params(p=0.2)
    assert model.p.item() > 1.0
    assert model.p.item() >= 1.01 - 1e-6


def test_etas_unconstrained_mode_keeps_legacy_behavior():
    model = ETAS(
        device=torch.device("cpu"),
        richter_b=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        enforce_subcritical=False,
        enforce_p_gt_one=False,
    )
    model.set_params(k=0.7, alpha=0.99, p=0.8)
    assert math.isclose(model.k.item(), 0.7, rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(model.p.item(), 0.8, rel_tol=1e-6, abs_tol=1e-8)
    assert model.branching_ratio > 1.0
