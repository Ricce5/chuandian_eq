import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2

from .base import BGModel
from .kernel import (
    _causal_depthwise_conv1d,
    ExpKernel,
    GammaKernel,
    LogNormalKernel,
    MixtureKernel,
    PowerLawKernel,
)


def _build_single_kernel(
    kernel_type: str,
    kernel_size: int,
    dt: float,
    normalize_kernel: bool,
    exp_init_tau: float,
    gamma_init_k: float,
    gamma_init_beta: float,
    logn_init_mu: float,
    logn_init_sigma: float,
    powerlaw_init_alpha: float,
    powerlaw_init_tau: float,
) -> nn.Module:
    kernel_type = str(kernel_type).lower()
    if kernel_type == "exp":
        return ExpKernel(kernel_size, dt, normalize_kernel, init_tau=exp_init_tau)
    if kernel_type == "gamma":
        return GammaKernel(
            kernel_size,
            dt,
            normalize_kernel,
            init_k=gamma_init_k,
            init_beta=gamma_init_beta,
        )
    if kernel_type in ("lognormal", "logn"):
        return LogNormalKernel(
            kernel_size,
            dt,
            normalize_kernel,
            init_mu=logn_init_mu,
            init_sigma=logn_init_sigma,
        )
    if kernel_type in ("powerlaw", "power_law", "power"):
        return PowerLawKernel(
            kernel_size,
            dt,
            normalize_kernel,
            init_alpha=powerlaw_init_alpha,
            init_tau=powerlaw_init_tau,
        )
    raise ValueError(f"Unknown kernel_type: {kernel_type}")


def _resolve_kernel_types(
    kernel_types: list[str] | tuple[str, ...],
) -> list[str]:
    resolved = list(kernel_types)
    if len(resolved) == 0:
        raise ValueError("kernel_types must contain at least one kernel type.")
    return [str(name).lower() for name in resolved]


def _build_kernel_module(
    kernel_types: list[str] | tuple[str, ...],
    kernel_size: int,
    dt: float,
    normalize_kernel: bool,
    exp_init_tau: float,
    gamma_init_k: float,
    gamma_init_beta: float,
    logn_init_mu: float,
    logn_init_sigma: float,
    powerlaw_init_alpha: float,
    powerlaw_init_tau: float,
) -> nn.Module:
    resolved_kernel_types = _resolve_kernel_types(kernel_types)
    kernels = [
        _build_single_kernel(
            kernel_type=name,
            kernel_size=kernel_size,
            dt=dt,
            normalize_kernel=normalize_kernel,
            exp_init_tau=exp_init_tau,
            gamma_init_k=gamma_init_k,
            gamma_init_beta=gamma_init_beta,
            logn_init_mu=logn_init_mu,
            logn_init_sigma=logn_init_sigma,
            powerlaw_init_alpha=powerlaw_init_alpha,
            powerlaw_init_tau=powerlaw_init_tau,
        )
        for name in resolved_kernel_types
    ]
    if len(kernels) == 1:
        return kernels[0]
    return MixtureKernel(
        kernels,
        normalize=normalize_kernel,
        normalize_weights=True,
    )


@BGModel.register("mamba_moe")
class MambaMoEBGModel(BGModel):
    """Use a Mamba gate to combine proportional and kernel experts.

    The gate predicts one weight per expert at each timestep. Set
    ``normalize_expert_gates=True`` to use a softmax gate whose weights sum to
    one across experts.
    """

    def __init__(
        self,
        d_feature: int,
        scale_init: float,
        model_type: str,
        d_model: int,
        d_state: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_scale: float = 1.0,
        a_init_range: tuple[float, float] | None = None,
        gate_activation: str = "sigmoid",
        kernel_types: list[str] | tuple[str, ...] = ("gamma",),
        kernel_size: int = 128,
        dt: float = 1.0,
        normalize_kernel: bool = True,
        normalize_expert_gates: bool = False,
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
        no_weight_decay: bool = False,
    ):
        super().__init__(
            device=device,
            scale_init=scale_init,
            no_weight_decay=no_weight_decay,
        )
        self.ts_batch_cache = None
        self.device = device
        self.d_feature = int(d_feature)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.kernel_types = _resolve_kernel_types(kernel_types)
        self.gate_activation = str(gate_activation).lower()
        self.normalize_expert_gates = bool(normalize_expert_gates)
        if self.gate_activation not in {"sigmoid", "softplus", "none"}:
            raise ValueError(
                f"Unknown gate_activation: {gate_activation}. "
                "Expected one of: sigmoid, softplus, none."
            )

        mamba_cls = {"mamba": Mamba, "mamba2": Mamba2}.get(model_type)
        if mamba_cls is None:
            raise ValueError(f"Unknown model_type: {model_type}")

        mamba_kwargs: dict = {
            "d_model": d_model,
            "d_state": d_state,
            "d_conv": 4,
            "dt_min": dt_min,
            "dt_max": dt_max,
        }
        if model_type == "mamba":
            mamba_kwargs["dt_scale"] = dt_scale
        elif model_type == "mamba2" and a_init_range is not None:
            mamba_kwargs["A_init_range"] = a_init_range

        self.fc_in = nn.Linear(d_feature, d_model, bias=False)
        self.mamba = mamba_cls(**mamba_kwargs)
        self.gate_head = nn.Linear(d_model, 2, bias=True)

        self.proportional_head = nn.Linear(d_feature, 1, bias=False)
        torch.nn.init.constant_(self.proportional_head.weight, 1.0)

        self.kernel = _build_kernel_module(
            kernel_types=self.kernel_types,
            kernel_size=kernel_size,
            dt=dt,
            normalize_kernel=normalize_kernel,
            exp_init_tau=exp_init_tau,
            gamma_init_k=gamma_init_k,
            gamma_init_beta=gamma_init_beta,
            logn_init_mu=logn_init_mu,
            logn_init_sigma=logn_init_sigma,
            powerlaw_init_alpha=powerlaw_init_alpha,
            powerlaw_init_tau=powerlaw_init_tau,
        )
        self.use_mlp = bool(use_mlp)
        if self.use_mlp:
            self.kernel_head = nn.Sequential(
                nn.Linear(self.d_feature, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, 1, bias=False),
            )
        else:
            self.kernel_head = nn.Linear(self.d_feature, 1, bias=False)
            torch.nn.init.constant_(self.kernel_head.weight, 1.0)

        if no_weight_decay:
            for param in self.parameters():
                param._no_weight_decay = True

        if device is not None:
            self.to(device)

    @property
    def context_dim(self) -> int:
        return int(self.fc_in.out_features)

    def _prepare_ts_inputs(
        self,
        ts_batch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if not hasattr(ts_batch, "time_series") or not hasattr(ts_batch, "time_series_times"):
            raise ValueError(
                "ts_batch must contain 'time_series' and 'time_series_times' to compute background context."
            )

        time_series = ts_batch.time_series.to(self.device, dtype=torch.float32)
        time_series_times = ts_batch.time_series_times.to(self.device)
        ts_mask = getattr(ts_batch, "time_series_mask", None)
        if ts_mask is not None:
            ts_mask = ts_mask.to(self.device)
        return time_series, time_series_times, ts_mask

    def _gate_hidden(self, time_series: torch.Tensor) -> torch.Tensor:
        gate_in = self.fc_in(time_series)
        return self.mamba(gate_in.contiguous())

    def _gate_weights(self, gate_hidden: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate_head(gate_hidden)
        if self.normalize_expert_gates:
            return torch.softmax(gate_logits, dim=-1)
        if self.gate_activation == "sigmoid":
            return torch.sigmoid(gate_logits)
        if self.gate_activation == "softplus":
            return F.softplus(gate_logits)
        return gate_logits

    def _kernel_expert(self, time_series: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel(device=time_series.device, dtype=time_series.dtype)
        conv_out = _causal_depthwise_conv1d(time_series, kernel)
        return self.kernel_head(conv_out)

    def context_trajectory(
        self,
        ts_batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_series, time_series_times, ts_mask = self._prepare_ts_inputs(ts_batch)
        gate_hidden = self._gate_hidden(time_series)
        if ts_mask is not None:
            gate_hidden = gate_hidden * ts_mask.to(
                device=gate_hidden.device,
                dtype=gate_hidden.dtype,
            ).unsqueeze(-1)
        return time_series_times, gate_hidden

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        proportional_out = self.proportional_head(time_series)
        kernel_out = self._kernel_expert(time_series)
        gate_hidden = self._gate_hidden(time_series)
        gate_weights = self._gate_weights(gate_hidden)

        proportional_weight = gate_weights[..., :1]
        kernel_weight = gate_weights[..., 1:]
        return proportional_weight * proportional_out + kernel_weight * kernel_out
