import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dot_dict import DotDict
from .base import BGModel


def _causal_depthwise_conv1d(x_btf: torch.Tensor, h_k: torch.Tensor) -> torch.Tensor:
    """
    Causal depthwise conv with shared kernel across channels.

    Args:
        x_btf: (B, T, F)
        h_k:   (K,) kernel, assumed causal, nonnegative preferred

    Returns:
        y_btf: (B, T, F)
    """
    B, T, Fdim = x_btf.shape
    K = h_k.numel()

    x = x_btf.transpose(1, 2)                     # (B, F, T)
    x = F.pad(x, (K - 1, 0))                      # causal left pad
    w = h_k.view(1, 1, K).repeat(Fdim, 1, 1)      # (F, 1, K)
    y = F.conv1d(x, w, groups=Fdim)               # (B, F, T)
    return y.transpose(1, 2)                      # (B, T, F)


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
    k>0, beta>0.  当 k>1 时有延迟峰值（非常适合注水延迟响应）。
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
        # 防止 0^(k-1) 数值问题
        tau_safe = tau.clamp_min(1e-8)
        h = torch.pow(tau_safe, k - 1.0) * torch.exp(-beta * tau)
        h = h.clamp_min(0.0)
        return self._normalize(h)


class LogNormalKernel(_BaseKernel):
    """
    h(tau) ∝ 1/(tau*sigma*sqrt(2pi)) * exp(-(ln tau - mu)^2/(2 sigma^2))
    有长尾延迟，适合扩散+长记忆。
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
        tau = tau.clamp_min(self.dt)  # 避免 tau=0 的 ln
        log_tau = torch.log(tau)
        coeff = 1.0 / (tau * sigma * math.sqrt(2.0 * math.pi))
        expo = torch.exp(-0.5 * torch.square((log_tau - self.mu) / sigma))
        h = (coeff * expo).clamp_min(0.0)
        return self._normalize(h)


class MixtureKernel(_BaseKernel):
    """
    混合核：h = sum_r w_r * h_r, w_r>=0 且 sum w_r = 1
    用于多尺度响应（短延迟 + 长延迟）。
    """
    def __init__(self, kernels: list[_BaseKernel]):
        assert len(kernels) >= 2, "MixtureKernel needs >=2 component kernels."
        ks = kernels[0].kernel_size
        dt = kernels[0].dt
        for k in kernels:
            assert k.kernel_size == ks and k.dt == dt, "All kernels must share kernel_size and dt."
        super().__init__(kernel_size=ks, dt=dt, normalize=True)
        self.kernels = nn.ModuleList(kernels)
        self.logits = nn.Parameter(torch.zeros(len(kernels)))  # mixture weights

    def forward(self, device=None, dtype=None) -> torch.Tensor:
        # 每个子核自己会 normalize（通常），这里再混合后整体 normalize
        hs = [k(device=device, dtype=dtype) for k in self.kernels]  # list of (K,)
        H = torch.stack(hs, dim=0)                                  # (R, K)
        w = torch.softmax(self.logits, dim=0).to(H.dtype)           # (R,)
        h = (w.unsqueeze(1) * H).sum(dim=0)                         # (K,)
        return self._normalize(h)


@BGModel.register("kernel")
class KernelBGModel(BGModel):
    """
    背景强度模型：注水时间序列 x(t) 经过“因果卷积核”得到响应 z(t)，
    再映射到非负强度 λ_bg(t)。

    intensity_traj(t) = scale * softplus( Linear( z(t) ) + bias )

    支持 kernel_type:
      - "exp"
      - "gamma"
      - "lognormal"
      - "mix" (默认：gamma + exp 的混合，可改)
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
    ):
        super().__init__(device=device, scale_init=scale_init)
        self.d_feature = int(d_feature)
        self.kernel_size = int(kernel_size)
        self.dt = float(dt)

        kt = kernel_type.lower()
        if kt == "exp":
            self.kernel = ExpKernel(kernel_size, dt, normalize_kernel, init_tau=exp_init_tau)
        elif kt == "gamma":
            self.kernel = GammaKernel(kernel_size, dt, normalize_kernel, init_k=gamma_init_k, init_beta=gamma_init_beta)
        elif kt in ("lognormal", "logn"):
            self.kernel = LogNormalKernel(kernel_size, dt, normalize_kernel, init_mu=logn_init_mu, init_sigma=logn_init_sigma)
        elif kt in ("mix", "mixture"):
            self.kernel = MixtureKernel([
                GammaKernel(kernel_size, dt, normalize_kernel, init_k=gamma_init_k, init_beta=gamma_init_beta),
                ExpKernel(kernel_size, dt, normalize_kernel, init_tau=exp_init_tau),
            ])
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")


        self.use_mlp = bool(use_mlp)
        if self.use_mlp:
            self.head = nn.Sequential(
                nn.Linear(self.d_feature, hidden),
                nn.SiLU(),
                nn.Linear(hidden, 1),
                nn.Softplus(),
            )
        else:
            self.head = nn.Linear(self.d_feature, 1,bias=False)
            torch.nn.init.constant_(self.head.weight, 1.0)
        if device is not None:
            self.to(device)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_series: (B, T, F) 注水特征序列
        Returns:
            scaled_intensity: (B, T, 1) 非负（softplus）
        """
        h = self.kernel(device=time_series.device, dtype=time_series.dtype)  # (K,)

        z = _causal_depthwise_conv1d(time_series, h)                         # (B,T,F)

        out = self.head(z)                                                   # (B,T,1)
     
        return out                                                        # (B,T,1)
