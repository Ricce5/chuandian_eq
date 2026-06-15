import torch

from src.models.mamba.s4d import S4D

from .base import BGModel
from .kernel import (
    _causal_depthwise_conv1d,
    ExpKernel,
    GammaKernel,
    LogNormalKernel,
    MixtureKernel,
    PowerLawKernel,
)


def _build_slow_kernel(
    slow_kernel_type: str,
    slow_kernel_size: int,
    slow_kernel_dt: float,
    slow_kernel_normalize: bool,
    slow_exp_init_tau: float,
    slow_gamma_init_k: float,
    slow_gamma_init_beta: float,
    slow_logn_init_mu: float,
    slow_logn_init_sigma: float,
    slow_powerlaw_init_alpha: float,
    slow_powerlaw_init_tau: float,
    slow_mix_normalize_weights: bool,
):
    kernel_type = str(slow_kernel_type).lower()
    if kernel_type == "exp":
        return ExpKernel(
            kernel_size=slow_kernel_size,
            dt=slow_kernel_dt,
            normalize=slow_kernel_normalize,
            init_tau=slow_exp_init_tau,
        )
    if kernel_type == "gamma":
        return GammaKernel(
            kernel_size=slow_kernel_size,
            dt=slow_kernel_dt,
            normalize=slow_kernel_normalize,
            init_k=slow_gamma_init_k,
            init_beta=slow_gamma_init_beta,
        )
    if kernel_type in {"lognormal", "logn"}:
        return LogNormalKernel(
            kernel_size=slow_kernel_size,
            dt=slow_kernel_dt,
            normalize=slow_kernel_normalize,
            init_mu=slow_logn_init_mu,
            init_sigma=slow_logn_init_sigma,
        )
    if kernel_type in {"powerlaw", "power_law", "power"}:
        return PowerLawKernel(
            kernel_size=slow_kernel_size,
            dt=slow_kernel_dt,
            normalize=slow_kernel_normalize,
            init_alpha=slow_powerlaw_init_alpha,
            init_tau=slow_powerlaw_init_tau,
        )
    if kernel_type in {"mix", "mixture"}:
        return MixtureKernel(
            [
                GammaKernel(
                    kernel_size=slow_kernel_size,
                    dt=slow_kernel_dt,
                    normalize=slow_kernel_normalize,
                    init_k=slow_gamma_init_k,
                    init_beta=slow_gamma_init_beta,
                ),
                ExpKernel(
                    kernel_size=slow_kernel_size,
                    dt=slow_kernel_dt,
                    normalize=slow_kernel_normalize,
                    init_tau=slow_exp_init_tau,
                ),
            ],
            normalize=slow_kernel_normalize,
            normalize_weights=slow_mix_normalize_weights,
        )
    raise ValueError(f"Unknown slow_kernel_type: {slow_kernel_type}")


@BGModel.register("s4d")
class S4DBGModel(BGModel):
    """S4D-based background intensity model.

    The model mirrors :class:`MambaBGModel`: input features are projected to a
    hidden sequence, processed by an S4D layer with FFT-kernel convolution, and
    then projected to a scalar background intensity trajectory.
    """

    def __init__(
        self,
        d_feature: int,
        scale_init: float,
        d_model: int,
        d_state: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_skip: bool = True,
        use_output_linear: bool = True,
        gated_output: bool = True,
        input_norm: bool = False,
        input_linear: bool = False,
        s4d_bias: bool = False,
        A_real_init: float = 0.5,
        C_init_scale: float = 1.0,
        learnable_A_imag: bool = True,
        learnable_dt: bool = True,
        kernel_chunk_size: int | None = None,
        use_slow_branch: bool = False,
        slow_kernel_type: str = "mix",
        slow_kernel_size: int = 64,
        slow_kernel_dt: float = 1.0,
        slow_kernel_normalize: bool = True,
        slow_exp_init_tau: float = 10.0,
        slow_gamma_init_k: float = 3.0,
        slow_gamma_init_beta: float = 0.3,
        slow_logn_init_mu: float = 2.0,
        slow_logn_init_sigma: float = 1.0,
        slow_powerlaw_init_alpha: float = 1.5,
        slow_powerlaw_init_tau: float = 10.0,
        slow_mix_normalize_weights: bool = True,
        fast_mix_init: float = 0.7,
        smooth_kernel_size: int | None = None,
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
        self.use_slow_branch = bool(use_slow_branch)

        self.fc_in = torch.nn.Linear(d_feature, d_model, bias=False)
        self.s4d = S4D(
            d_model=d_model,
            d_state=d_state,
            dropout=dropout,
            transposed=False,
            activation=activation,
            use_skip=use_skip,
            use_output_linear=use_output_linear,
            gated_output=gated_output,
            input_norm=input_norm,
            input_linear=input_linear,
            bias=s4d_bias,
            dt_min=dt_min,
            dt_max=dt_max,
            A_real_init=A_real_init,
            C_init_scale=C_init_scale,
            learnable_A_imag=learnable_A_imag,
            learnable_dt=learnable_dt,
            kernel_chunk_size=kernel_chunk_size,
            device=device,
        )
        self.fc_out = torch.nn.Linear(d_model, 1, bias=False)

        if self.use_slow_branch:
            self.slow_kernel = _build_slow_kernel(
                slow_kernel_type=slow_kernel_type,
                slow_kernel_size=slow_kernel_size,
                slow_kernel_dt=slow_kernel_dt,
                slow_kernel_normalize=slow_kernel_normalize,
                slow_exp_init_tau=slow_exp_init_tau,
                slow_gamma_init_k=slow_gamma_init_k,
                slow_gamma_init_beta=slow_gamma_init_beta,
                slow_logn_init_mu=slow_logn_init_mu,
                slow_logn_init_sigma=slow_logn_init_sigma,
                slow_powerlaw_init_alpha=slow_powerlaw_init_alpha,
                slow_powerlaw_init_tau=slow_powerlaw_init_tau,
                slow_mix_normalize_weights=slow_mix_normalize_weights,
            )
            fast_mix_init = float(fast_mix_init)
            fast_mix_init = min(max(fast_mix_init, 1e-4), 1.0 - 1e-4)
            init_logit = torch.logit(torch.tensor(fast_mix_init, dtype=torch.float32))
            self.fast_mix_logit = torch.nn.Parameter(init_logit)
        else:
            self.slow_kernel = None
            self.fast_mix_logit = None

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

    def context_trajectory(self, ts_batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return S4D fast-branch hidden states on the time-series grid."""
        time_series, time_series_times, ts_mask = self._prepare_ts_inputs(ts_batch)
        ssm_in = self.fc_in(time_series)
        fast_out = self.s4d(ssm_in.contiguous())
        if ts_mask is not None:
            fast_out = fast_out * ts_mask.to(
                device=fast_out.device,
                dtype=fast_out.dtype,
            ).unsqueeze(-1)
        return time_series_times, fast_out

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        """Project input features to hidden states and apply S4D."""
        ssm_in = self.fc_in(time_series)
        fast_out = self.s4d(ssm_in.contiguous())
        if self.use_slow_branch:
            slow_h = self.slow_kernel(device=ssm_in.device, dtype=ssm_in.dtype)
            slow_out = _causal_depthwise_conv1d(ssm_in, slow_h)
            fast_mix = torch.sigmoid(self.fast_mix_logit).to(dtype=fast_out.dtype)
            ssm_out = fast_mix * fast_out + (1.0 - fast_mix) * slow_out
        else:
            ssm_out = fast_out

        if self.smoothing_kernel is not None:
            ssm_out = _causal_depthwise_conv1d(ssm_out, self.smoothing_kernel)

        return self.fc_out(ssm_out)

