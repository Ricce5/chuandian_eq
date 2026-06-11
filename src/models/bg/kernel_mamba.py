import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2

from .base import BGModel
from .kernel import (
    _causal_depthwise_conv1d,
    DeltaKernel,
    ExpKernel,
    GammaKernel,
    LogNormalKernel,
    PowerLawKernel,
)


def _resolve_kernel_types(kernel_types: list[str] | tuple[str, ...]) -> list[str]:
    resolved = [str(name).lower() for name in kernel_types]
    if not resolved:
        raise ValueError("kernel_types must contain at least one kernel type.")
    return resolved


def _build_kernel(
    kernel_type: str,
    *,
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
    if kernel_type in {"delta", "proportional", "identity"}:
        return DeltaKernel(kernel_size, dt, normalize_kernel)
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
    if kernel_type in {"lognormal", "logn"}:
        return LogNormalKernel(
            kernel_size,
            dt,
            normalize_kernel,
            init_mu=logn_init_mu,
            init_sigma=logn_init_sigma,
        )
    if kernel_type in {"powerlaw", "power_law", "power"}:
        return PowerLawKernel(
            kernel_size,
            dt,
            normalize_kernel,
            init_alpha=powerlaw_init_alpha,
            init_tau=powerlaw_init_tau,
        )
    raise ValueError(f"Unknown kernel_type: {kernel_type}")


@BGModel.register("kernel_mamba")
class KernelMambaBGModel(BGModel):
    """Interpretable kernel branch with a Mamba residual.

    The kernel branch produces interpretable per-kernel responses, while the
    Mamba branch remains as a high-capacity residual path. Like ``KernelBGModel``
    and ``MambaBGModel``, branch heads are linear by default and the base
    ``BGModel`` clamps the final summed intensity to be non-negative.
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
        kernel_types: list[str] | tuple[str, ...] = ("gamma", "exp"),
        kernel_size: int = 64,
        dt: float = 1.0,
        normalize_kernel: bool = True,
        gate_type: str = "softmax",
        gate_temperature: float = 1.0,
        kernel_init_weight: float = 0.2,
        kernel_activation: str = "identity",
        residual_init_weight: float = 0.8,
        residual_activation: str = "identity",
        residual_penalty_weight: float = 1.0,
        gate_smooth_weight: float = 0.0,
        gate_entropy_weight: float = 0.0,
        smooth_kernel_size: int | None = None,
        exp_init_tau: float = 10.0,
        gamma_init_k: float = 3.0,
        gamma_init_beta: float = 0.3,
        logn_init_mu: float = 2.0,
        logn_init_sigma: float = 1.0,
        powerlaw_init_alpha: float = 1.5,
        powerlaw_init_tau: float = 10.0,
        no_weight_decay: bool = False,
        device: torch.device | None = None,
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
        self.kernel_types = tuple(_resolve_kernel_types(kernel_types))
        self.expert_names = self.kernel_types + ("mamba",)
        self.num_kernel_experts = len(self.kernel_types)
        self.gate_type = str(gate_type).lower()
        self.gate_temperature = float(gate_temperature)
        self.kernel_activation = str(kernel_activation).lower()
        self.residual_activation = str(residual_activation).lower()
        self.residual_penalty_weight = float(residual_penalty_weight)
        self.gate_smooth_weight = float(gate_smooth_weight)
        self.gate_entropy_weight = float(gate_entropy_weight)

        if self.gate_type not in {"softmax", "sigmoid", "none"}:
            raise ValueError("gate_type must be one of {'softmax', 'sigmoid', 'none'}.")
        if self.gate_temperature <= 0.0:
            raise ValueError("gate_temperature must be positive.")
        if self.kernel_activation not in {"relu", "softplus", "identity"}:
            raise ValueError("kernel_activation must be one of {'relu', 'softplus', 'identity'}.")
        if self.residual_activation not in {"softplus", "relu", "identity"}:
            raise ValueError("residual_activation must be one of {'softplus', 'relu', 'identity'}.")
        if self.residual_penalty_weight < 0.0 or self.gate_smooth_weight < 0.0 or self.gate_entropy_weight < 0.0:
            raise ValueError("Regularization weights must be non-negative.")

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
        self.residual_head = nn.Linear(d_model, 1, bias=False)
        self.gate_head = nn.Linear(d_model, self.num_kernel_experts, bias=True)

        self.kernel_modules = nn.ModuleList(
            [
                _build_kernel(
                    name,
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
                for name in self.kernel_types
            ]
        )
        self.kernel = self.kernel_modules[0] if self.num_kernel_experts == 1 else self.kernel_modules
        self.kernel_head = nn.Linear(d_feature, 1, bias=False)
        nn.init.constant_(self.kernel_head.weight, 1.0)

        kernel_init_weight = max(float(kernel_init_weight), 1e-8)
        residual_init_weight = max(float(residual_init_weight), 1e-8)
        self.log_kernel_weight = nn.Parameter(torch.tensor(kernel_init_weight).log())
        self.log_residual_weight = nn.Parameter(torch.tensor(residual_init_weight).log())

        if smooth_kernel_size is not None and smooth_kernel_size > 1:
            smoothing = torch.ones(smooth_kernel_size, dtype=torch.float32) / float(smooth_kernel_size)
            self.register_buffer("smoothing_kernel", smoothing, persistent=False)
        else:
            self.smoothing_kernel = None

        if no_weight_decay:
            for param in self.parameters():
                param._no_weight_decay = True

        if device is not None:
            self.to(device)

    @property
    def context_dim(self) -> int:
        return int(self.fc_in.out_features)

    @property
    def kernel_weight(self) -> torch.Tensor:
        return torch.exp(self.log_kernel_weight)

    @property
    def residual_weight(self) -> torch.Tensor:
        return torch.exp(self.log_residual_weight)

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

    def _mamba_hidden(self, time_series: torch.Tensor) -> torch.Tensor:
        mamba_in = self.fc_in(time_series)
        hidden = self.mamba(mamba_in.contiguous())
        if self.smoothing_kernel is not None:
            hidden = _causal_depthwise_conv1d(hidden, self.smoothing_kernel)
        return hidden

    def _kernel_raw_outputs(self, time_series: torch.Tensor) -> torch.Tensor:
        raw_outputs = []
        for kernel_module in self.kernel_modules:
            kernel = kernel_module(device=time_series.device, dtype=time_series.dtype)
            conv_out = _causal_depthwise_conv1d(time_series, kernel)
            raw_outputs.append(self.kernel_head(conv_out))
        return torch.cat(raw_outputs, dim=-1)

    def _kernel_gates(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.gate_head(hidden) / self.gate_temperature
        if self.gate_type == "softmax":
            return torch.softmax(logits, dim=-1)
        if self.gate_type == "sigmoid":
            return torch.sigmoid(logits)
        return torch.ones_like(logits)

    def _residual_output(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.residual_head(hidden)
        if self.residual_activation == "softplus":
            return F.softplus(residual)
        if self.residual_activation == "relu":
            return F.relu(residual)
        return residual

    def _kernel_output(self, time_series: torch.Tensor) -> torch.Tensor:
        kernel_raw = self._kernel_raw_outputs(time_series)
        if self.kernel_activation == "softplus":
            return F.softplus(kernel_raw)
        if self.kernel_activation == "relu":
            return F.relu(kernel_raw)
        return kernel_raw

    def _component_values(
        self,
        time_series: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self._mamba_hidden(time_series)
        gates = self._kernel_gates(hidden)
        kernel_outputs = self._kernel_output(time_series)
        kernel_components = self.kernel_weight * gates * kernel_outputs
        residual_component = self.residual_weight * self._residual_output(hidden)
        total = kernel_components.sum(dim=-1, keepdim=True) + residual_component
        return total, kernel_components, residual_component, gates

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        total, _, _, _ = self._component_values(time_series)
        return total

    def context_trajectory(
        self,
        ts_batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_series, time_series_times, ts_mask = self._prepare_ts_inputs(ts_batch)
        hidden = self._mamba_hidden(time_series)
        if ts_mask is not None:
            hidden = hidden * ts_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
        return time_series_times, hidden

    def gate_trajectory(self, ts_batch) -> tuple[torch.Tensor, torch.Tensor]:
        time_series, time_series_times, ts_mask = self._prepare_ts_inputs(ts_batch)
        gates = self._kernel_gates(self._mamba_hidden(time_series))
        if ts_mask is not None:
            gates = gates * ts_mask.to(device=gates.device, dtype=gates.dtype).unsqueeze(-1)
        return time_series_times, gates

    gate_weights_trajectory = gate_trajectory

    def component_trajectory(self, ts_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return signed scaled additive components: kernel experts plus Mamba residual.

        Shape is ``(B, T, E + 1)`` in ``self.expert_names`` order. Components are
        multiplied by the global BG scale. Their sum is the pre-clamp background
        trajectory; ``BGModel`` applies the final non-negative clamp.
        """
        time_series, time_series_times, ts_mask = self._prepare_ts_inputs(ts_batch)
        _, kernel_components, residual_component, _ = self._component_values(time_series)
        components = torch.cat([kernel_components, residual_component], dim=-1) * self._scale
        if ts_mask is not None:
            components = components * ts_mask.to(device=components.device, dtype=components.dtype).unsqueeze(-1)
        return time_series_times, components

    def expert_trajectory(self, ts_batch) -> tuple[torch.Tensor, torch.Tensor]:
        return self.component_trajectory(ts_batch)

    def residual_ratio(self, ts_batch, eps: float = 1e-8) -> torch.Tensor:
        _, components = self.component_trajectory(ts_batch)
        abs_components = components.abs()
        residual = abs_components[..., -1].sum(dim=1)
        total = abs_components.sum(dim=(1, 2)).clamp_min(eps)
        return residual / total

    def kl_term(self, batch, eps: float = 1e-8) -> torch.Tensor:
        """Return per-sample regularization through the legacy ``kl_term`` hook.

        This is not a probabilistic KL divergence. It is a weighted sum of:
        - residual share penalty: discourages the Mamba residual branch from
          explaining everything instead of the interpretable kernel experts;
        - gate smoothness penalty: discourages fast time-to-time gate changes;
        - gate entropy penalty: when minimized, encourages sharper expert
          selection for softmax gates.

        The component penalties use absolute signed contributions because
        ``kernel_activation`` and ``residual_activation`` can be ``identity``.
        Returns a tensor of shape ``(B,)``.
        """
        time_series = batch.time_series.to(self.device, dtype=torch.float32)
        _, kernel_components, residual_component, gates = self._component_values(time_series)

        kernel_abs_mass = kernel_components.abs().sum(dim=(1, 2))
        residual_abs_mass = residual_component.abs().sum(dim=(1, 2))
        total_abs_mass = (kernel_abs_mass + residual_abs_mass).clamp_min(eps)

        residual_share = residual_abs_mass / total_abs_mass
        regularizer = self.residual_penalty_weight * residual_share

        if self.gate_smooth_weight > 0.0 and gates.size(1) > 1:
            gate_step_change = gates[:, 1:, :] - gates[:, :-1, :]
            gate_smoothness = gate_step_change.square().mean(dim=(1, 2))
            regularizer = regularizer + self.gate_smooth_weight * gate_smoothness

        if self.gate_entropy_weight > 0.0:
            gate_probabilities = gates.clamp_min(eps)
            gate_entropy = -(gate_probabilities * gate_probabilities.log()).sum(dim=-1)
            mean_gate_entropy = gate_entropy.mean(dim=1)
            regularizer = regularizer + self.gate_entropy_weight * mean_gate_entropy

        return regularizer
