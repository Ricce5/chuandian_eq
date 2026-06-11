import pytest
import torch


pytest.importorskip("mamba_ssm")

from src.models.bg.kernel_mamba import KernelMambaBGModel


def test_kernel_mamba_defaults_use_linear_branch_heads():
    model = KernelMambaBGModel(
        d_feature=1,
        scale_init=1.0,
        model_type="mamba",
        d_model=4,
        d_state=4,
        kernel_types=("delta",),
        kernel_size=3,
        device=torch.device("cpu"),
    )

    assert model.kernel_activation == "identity"
    assert model.residual_activation == "identity"


def test_kernel_mamba_branch_heads_allow_signed_outputs():
    model = KernelMambaBGModel(
        d_feature=1,
        scale_init=1.0,
        model_type="mamba",
        d_model=4,
        d_state=4,
        kernel_types=("delta",),
        kernel_size=3,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        model.kernel_head.weight.fill_(-1.0)
        model.residual_head.weight.fill_(-1.0)

    time_series = torch.ones(1, 5, 1)
    kernel_output = model._kernel_output(time_series)
    residual_output = model._residual_output(torch.ones(1, 5, model.d_model))

    assert torch.any(kernel_output < 0.0)
    assert torch.any(residual_output < 0.0)
