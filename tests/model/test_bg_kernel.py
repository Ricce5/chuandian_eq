import math
import torch
import pytest

from src.models.bg.kernel import (
    _causal_depthwise_conv1d,
    ExpKernel,
    GammaKernel,
    LogNormalKernel,
    MixtureKernel,
    KernelBGModel,
)
from src.data.dot_dict import DotDict


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Tests for helper function _causal_depthwise_conv1d
# -----------------------------------------------------------------------------


def test_causal_depthwise_conv1d_shape_and_device():
    device = get_device()
    B, T, F = 2, 16, 4
    K = 5
    x = torch.randn(B, T, F, device=device)
    h = torch.randn(K, device=device)

    y = _causal_depthwise_conv1d(x, h)

    assert y.shape == (B, T, F)
    assert y.device == device


# -----------------------------------------------------------------------------
# Tests for kernels
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_cls, kwargs", [
    (ExpKernel, {"init_tau": 10.0}),
    (GammaKernel, {"init_k": 3.0, "init_beta": 0.3}),
    (LogNormalKernel, {"init_mu": 2.0, "init_sigma": 1.0}),
])
@pytest.mark.parametrize("kernel_size", [8, 17])
def test_single_kernels_basic_properties(kernel_cls, kwargs, kernel_size):
    device = get_device()

    kernel = kernel_cls(kernel_size=kernel_size, dt=1.0, normalize=True, **kwargs).to(device)
    h = kernel(device=device, dtype=torch.float32)

    assert h.shape == (kernel_size,)
    assert h.device == device
    # 非负
    assert torch.all(h >= 0.0)
    # 归一化（允许一定数值误差）
    assert torch.isclose(h.sum(), torch.tensor(1.0, device=device), atol=1e-4)


def test_mixture_kernel_properties():
    device = get_device()
    kernel_size = 16

    k1 = GammaKernel(kernel_size, dt=1.0, normalize=True, init_k=3.0, init_beta=0.3)
    k2 = ExpKernel(kernel_size, dt=1.0, normalize=True, init_tau=5.0)
    mix = MixtureKernel([k1, k2]).to(device)

    h = mix(device=device, dtype=torch.float32)

    assert h.shape == (kernel_size,)
    assert h.device == device
    assert torch.all(h >= 0.0)
    assert torch.isclose(h.sum(), torch.tensor(1.0, device=device), atol=1e-4)


# -----------------------------------------------------------------------------
# Tests for KernelBGModel
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("kernel_type", ["exp", "gamma", "lognormal", "mix"])
def test_kernel_bg_model_scaled_intensity_shape_and_grad(kernel_type):
    device = get_device()
    B, T, Fdim = 3, 20, 5

    model = KernelBGModel(
        d_feature=Fdim,
        kernel_type=kernel_type,
        kernel_size=16,
        dt=1.0,
        normalize_kernel=True,
        use_mlp=False,
        device=device,
    )

    x = torch.randn(B, T, Fdim, device=device, requires_grad=True)

    out = model.scaled_intensity(x)
    assert out.shape == (B, T, 1)
    assert out.device == device
    # softplus 输出应当是非负
    assert torch.all(out >= 0.0)

    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.any(x.grad != 0)


@pytest.mark.parametrize("kernel_type", ["exp", "gamma"])
def test_kernel_bg_model_intensity_and_integral_forward(kernel_type):
    device = get_device()
    B, T, Fdim = 2, 16, 4

    model = KernelBGModel(
        d_feature=Fdim,
        kernel_type=kernel_type,
        kernel_size=8,
        dt=1.0,
        normalize_kernel=True,
        use_mlp=True,
        hidden=8,
        device=device,
    )

    time_series = torch.randn(B, T, Fdim, device=device)
    time_series_times = torch.linspace(0.0, 10.0, T, device=device).unsqueeze(0).repeat(B, 1)
    arrival_times = torch.linspace(0.0, 10.0, T // 2, device=device).unsqueeze(0).repeat(B, 1)

    batch = DotDict({
        "time_series": time_series,
        "time_series_times": time_series_times,
        "arrival_times": arrival_times,
        "t_nll_start": torch.zeros(B, device=device),
        "t_end": torch.full((B,), 10.0, device=device),
        "nll_event_mask": torch.ones(B, arrival_times.shape[1], device=device),
    })

    intensity = model.intensity(batch)
    integral = model.intensity_integral(batch)

    assert intensity.shape == (B, arrival_times.shape[1])
    assert integral.shape == (B,)
    assert torch.all(intensity >= 0.0)
    assert torch.all(integral >= 0.0)


def test_kernel_bg_model_nll_and_nll_change():
    device = get_device()
    B, T, Fdim = 2, 12, 3

    model = KernelBGModel(
        d_feature=Fdim,
        kernel_type="gamma",
        kernel_size=6,
        dt=1.0,
        normalize_kernel=True,
        device=device,
    )

    time_series = torch.randn(B, T, Fdim, device=device)
    time_series_times = torch.linspace(0.0, 6.0, T, device=device).unsqueeze(0).repeat(B, 1)
    arrival_times = torch.linspace(0.0, 6.0, T // 2, device=device).unsqueeze(0).repeat(B, 1)

    event_mask = torch.ones(B, arrival_times.shape[1], device=device)

    batch = DotDict({
        "time_series": time_series,
        "time_series_times": time_series_times,
        "arrival_times": arrival_times,
        "t_nll_start": torch.zeros(B, device=device),
        "t_end": torch.full((B,), 6.0, device=device),
        "nll_event_mask": event_mask,
    })

    with torch.no_grad():
        lam_bg = torch.full_like(arrival_times, 0.5, device=device)
        log_h_intensity = torch.log(lam_bg)

    nll_val = model.nll(batch)
    nll_change_val = model.nll_change(batch, log_h_intensity)

    assert nll_val.shape == (B,)
    assert nll_change_val.shape == (B,)
