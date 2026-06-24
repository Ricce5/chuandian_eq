import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BGModel


def _causal_depthwise_conv1d(x_btf: torch.Tensor, h_k: torch.Tensor) -> torch.Tensor:
    """
    Causal depthwise conv with a kernel shared across channels.

    Args:
        x_btf: (B, T, F)
        h_k:   (K,) causal kernel where h_k[0] is the current-time response
               and h_k[i] multiplies x[t - i]

    Returns:
        y_btf: (B, T, F)
    """
    B, T, Fdim = x_btf.shape
    K = h_k.numel()

    x = x_btf.transpose(1, 2)                     # (B, F, T)
    x = F.pad(x, (K - 1, 0))                      # causal left pad
    w = h_k.flip(0).view(1, 1, K).repeat(Fdim, 1, 1)  # (F, 1, K)
    y = F.conv1d(x, w, groups=Fdim)               # (B, F, T)
    return y.transpose(1, 2)                      # (B, T, F)


def _fft_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _causal_depthwise_fft_conv1d(x_btf: torch.Tensor, h_k: torch.Tensor) -> torch.Tensor:
    """
    FFT implementation of ``_causal_depthwise_conv1d``.

    This preserves the ``_causal_depthwise_conv1d`` helper semantics:
    ``h_k[0]`` is the zero-lag/current-time coefficient.

    Args:
        x_btf: (B, T, F)
        h_k:   (K,) causal kernel where h_k[0] multiplies x[t]

    Returns:
        y_btf: (B, T, F)
    """
    if x_btf.dim() != 3:
        raise ValueError(f"Expected x_btf to be 3D, got shape {tuple(x_btf.shape)}")
    if h_k.dim() != 1:
        raise ValueError(f"Expected h_k to be 1D, got shape {tuple(h_k.shape)}")

    B, T, Fdim = x_btf.shape
    K = h_k.numel()
    if K <= 0:
        raise ValueError("h_k must contain at least one coefficient")
    if T <= 0:
        return x_btf.new_empty(B, T, Fdim)

    fft_size = 1 << (T + K - 2).bit_length()
    conv_dtype = _fft_dtype(torch.promote_types(x_btf.dtype, h_k.dtype))

    x = x_btf.transpose(1, 2).to(dtype=conv_dtype)       # (B, F, T)
    h = h_k.to(dtype=conv_dtype)                         # zero-lag first
    x_fft = torch.fft.rfft(x, n=fft_size)
    h_fft = torch.fft.rfft(h, n=fft_size)
    y = torch.fft.irfft(x_fft * h_fft.view(1, 1, -1), n=fft_size)[..., :T]
    return y.transpose(1, 2).to(dtype=x_btf.dtype)       # (B, T, F)


class _BaseKernel(nn.Module):
    """Return a nonnegative causal kernel h[0..K-1], usually normalized."""
    def __init__(self, kernel_size: int, dt: float = 1.0, normalize: bool = True):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dt = float(dt)
        self.normalize = bool(normalize)

    def _normalize(self, h: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return h
        return h / (h.sum() + 1e-8)

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        raise NotImplementedError


class ExpKernel(_BaseKernel):
    """
    h(tau) ∝ exp(-tau / tau_decay)
    """
    def __init__(self, kernel_size: int, dt: float = 1.0, normalize: bool = True, init_tau: float = 10.0):
        super().__init__(kernel_size, dt, normalize)
        self.log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        device = device or self.log_tau.device
        dtype = dtype or self.log_tau.dtype
        tau_decay = torch.exp(self.log_tau).clamp_min(1e-6)
        tau = torch.arange(self.kernel_size, device=device, dtype=dtype) * self.dt
        h = torch.exp(-tau / tau_decay)
        h = h.clamp_min(0.0)
        return self._normalize(h)


class GammaKernel(_BaseKernel):
    """
    h(tau) ∝ tau^(k-1) * exp(-beta * tau)
    k>0, beta>0. When k>1 there is a delayed peak (useful for injection delayed response).
    """
    def __init__(
        self,
        kernel_size: int,
        dt: float = 1.0,
        normalize: bool = True,
        init_k: float = 3.0,
        init_beta: float = 0.3,
    ):
        super().__init__(kernel_size, dt, normalize)
        self.log_k = nn.Parameter(torch.tensor(float(init_k)).log())
        self.log_beta = nn.Parameter(torch.tensor(float(init_beta)).log())

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        device = device or self.log_k.device
        dtype = dtype or self.log_k.dtype
        k = torch.exp(self.log_k).clamp_min(1e-4)
        beta = torch.exp(self.log_beta).clamp_min(1e-6)

        tau = torch.arange(self.kernel_size, device=device, dtype=dtype) * self.dt
        # avoid numerical issues with 0^(k-1)
        tau_safe = tau.clamp_min(1e-8)
        h = torch.pow(tau_safe, k - 1.0) * torch.exp(-beta * tau)
        h = h.clamp_min(0.0)
        return self._normalize(h)


class LogNormalKernel(_BaseKernel):
    """
    h(tau) ∝ 1/(tau*sigma*sqrt(2pi)) * exp(-(ln tau - mu)^2/(2 sigma^2))
    Long-tailed delays suitable for diffusion and long memory.
    """
    def __init__(
        self,
        kernel_size: int,
        dt: float = 1.0,
        normalize: bool = True,
        init_mu: float = 2.0,
        init_sigma: float = 1.0,
    ):
        super().__init__(kernel_size, dt, normalize)
        self.mu = nn.Parameter(torch.tensor(float(init_mu)))
        self.log_sigma = nn.Parameter(torch.tensor(float(init_sigma)).log())

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        device = device or self.mu.device
        dtype = dtype or self.mu.dtype
        sigma = torch.exp(self.log_sigma).clamp_min(1e-4)

        tau = torch.arange(self.kernel_size, device=device, dtype=dtype) * self.dt
        tau = tau.clamp_min(self.dt)  # avoid ln(0)
        log_tau = torch.log(tau)
        coeff = 1.0 / (tau * sigma * math.sqrt(2.0 * math.pi))
        expo = torch.exp(-0.5 * torch.square((log_tau - self.mu) / sigma))
        h = (coeff * expo).clamp_min(0.0)
        return self._normalize(h)


class PowerLawKernel(_BaseKernel):
    """
    h(tau) ∝ (1 + tau / tau_scale)^(-alpha)

    A stable heavy-tailed kernel for long-memory responses. Compared with
    exponential decay, it decays more slowly and can better preserve
    persistent effects over long horizons.
    """
    def __init__(
        self,
        kernel_size: int,
        dt: float = 1.0,
        normalize: bool = True,
        init_alpha: float = 1.5,
        init_tau: float = 10.0,
    ):
        super().__init__(kernel_size, dt, normalize)
        self.log_alpha = nn.Parameter(torch.tensor(float(init_alpha)).log())
        self.log_tau = nn.Parameter(torch.tensor(float(init_tau)).log())

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        device = device or self.log_alpha.device
        dtype = dtype or self.log_alpha.dtype
        alpha = torch.exp(self.log_alpha).clamp_min(1e-4)
        tau_scale = torch.exp(self.log_tau).clamp_min(1e-6)

        tau = torch.arange(self.kernel_size, device=device, dtype=dtype) * self.dt
        h = torch.pow(1.0 + tau / tau_scale, -alpha)
        h = h.clamp_min(0.0)
        return self._normalize(h)


class DeltaKernel(_BaseKernel):
    """Discrete delta kernel: h[0] = 1, h[t>0] = 0.

    This acts as an identity shortcut under causal convolution and can be used
    to represent the proportional branch as a kernel expert.
    """

    def __init__(self, kernel_size: int, dt: float = 1.0, normalize: bool = True):
        super().__init__(kernel_size, dt, normalize)

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        h = torch.zeros(self.kernel_size, device=device, dtype=dtype)
        h[0] = 1.0
        return self._normalize(h)


class MixtureKernel(_BaseKernel):
    """
    Mixture kernel: h = sum_r w_r * h_r.

    When ``normalize_weights=True``, mixture weights satisfy
    ``w_r >= 0`` and ``sum_r w_r = 1``.
    Otherwise, weights remain nonnegative but are not normalized.
    Used for multi-scale responses (short delay + long delay).
    """
    def __init__(
        self,
        kernels: list[_BaseKernel],
        normalize: bool = True,
        normalize_weights: bool = True,
    ):
        assert len(kernels) >= 2, "MixtureKernel needs >=2 component kernels."
        ks = kernels[0].kernel_size
        dt = kernels[0].dt
        for k in kernels:
            assert k.kernel_size == ks and k.dt == dt, "All kernels must share kernel_size and dt."
        super().__init__(kernel_size=ks, dt=dt, normalize=normalize)
        self.kernels = nn.ModuleList(kernels)
        self.logits = nn.Parameter(torch.zeros(len(kernels)))  # mixture weights
        self.normalize_weights = bool(normalize_weights)

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        # Each sub-kernel is typically normalized; normalize after mixing.
        hs = [k(device=device, dtype=dtype) for k in self.kernels]  # list of (K,)
        H = torch.stack(hs, dim=0)                                  # (R, K)
        if self.normalize_weights:
            w = torch.softmax(self.logits, dim=0).to(H.dtype)       # (R,)
        else:
            w = F.softplus(self.logits).to(H.dtype)                 # (R,)
        h = (w.unsqueeze(1) * H).sum(dim=0)                         # (K,)
        return self._normalize(h)


@BGModel.register("kernel")
class KernelBGModel(BGModel):
    """
    Background intensity model: injection time series x(t) is convolved with a causal kernel to produce response z(t),
    which is then mapped to a nonnegative intensity λ_bg(t).

    intensity_traj(t) = scale * softplus( Linear( z(t) ) + bias )

    Supported kernel_type:
      - "exp"
      - "gamma"
      - "lognormal"
      - "powerlaw"
            - "delta" / "proportional" / "identity"
      - "mix" (default: mixture of gamma + exp, configurable)
    """
    def __init__(
        self,
        d_feature: int,
        scale_init: float = 200.0,
        kernel_type: str = "gamma",
        kernel_size: int = 128,
        dt: float = 1.0,
        normalize_kernel: bool = True,
        use_mlp: bool = False,
        hidden: int = 32,
        device: torch.device | None = None,
        exp_init_tau: float = 10.0,
        gamma_init_k: float = 3.0,
        gamma_init_beta: float = 0.3,
        logn_init_mu: float = 2.0,
        logn_init_sigma: float = 1.0,
        powerlaw_init_alpha: float = 1.5,
        powerlaw_init_tau: float = 10.0,
        mix_normalize_weights: bool = True,
        use_fft: bool = False,
    ):
        super().__init__(device=device, scale_init=scale_init)
        self.d_feature = int(d_feature)
        self.kernel_size = int(kernel_size)
        self.dt = float(dt)
        self.use_fft = bool(use_fft)

        kt = kernel_type.lower()
        if kt == "exp":
            self.kernel = ExpKernel(kernel_size, dt, normalize_kernel, init_tau=exp_init_tau)
        elif kt == "gamma":
            self.kernel = GammaKernel(kernel_size, dt, normalize_kernel, init_k=gamma_init_k, init_beta=gamma_init_beta)
        elif kt in ("lognormal", "logn"):
            self.kernel = LogNormalKernel(kernel_size, dt, normalize_kernel, init_mu=logn_init_mu, init_sigma=logn_init_sigma)
        elif kt in ("powerlaw", "power_law", "power"):
            self.kernel = PowerLawKernel(
                kernel_size,
                dt,
                normalize_kernel,
                init_alpha=powerlaw_init_alpha,
                init_tau=powerlaw_init_tau,
            )
        elif kt in ("delta", "proportional", "identity"):
            self.kernel = DeltaKernel(kernel_size, dt, normalize_kernel)
        elif kt in ("mix", "mixture"):
            self.kernel = MixtureKernel(
                [
                    GammaKernel(kernel_size, dt, normalize_kernel, init_k=gamma_init_k, init_beta=gamma_init_beta),
                    ExpKernel(kernel_size, dt, normalize_kernel, init_tau=exp_init_tau),
                ],
                normalize=normalize_kernel,
                normalize_weights=mix_normalize_weights,
            )
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")


        self.use_mlp = bool(use_mlp)
        if self.use_mlp:
            self.head = nn.Sequential(
                nn.Linear(self.d_feature, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, 1, bias=False),
            )
        else:
            self.head = nn.Linear(self.d_feature, 1, bias=False)
            torch.nn.init.constant_(self.head.weight, 1.0)
        if device is not None:
            self.to(device)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_series: (B, T, F) injection feature sequence
        Returns:
            scaled_intensity: (B, T, 1) nonnegative (softplus applied later)
        """
        h = self.kernel(device=time_series.device, dtype=time_series.dtype)  # (K,)

        conv_fn = _causal_depthwise_fft_conv1d if self.use_fft else _causal_depthwise_conv1d
        z = conv_fn(time_series, h)                                          # (B,T,F)

        out = self.head(z)                                                   # (B,T,1)
     
        return out                                                        # (B,T,1)


@BGModel.register("kernel_fft")
class KernelFFTBGModel(KernelBGModel):
    """FFT-backed variant of :class:`KernelBGModel` with the same public args."""

    def __init__(self, *args, **kwargs):
        kwargs["use_fft"] = True
        super().__init__(*args, **kwargs)
