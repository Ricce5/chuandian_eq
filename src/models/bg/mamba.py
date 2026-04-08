import torch
from mamba_ssm import Mamba, Mamba2
from .base import BGModel
from .kernel import (
    _causal_depthwise_conv1d,
    ExpKernel,
    GammaKernel,
    LogNormalKernel,
    MixtureKernel,
)


@BGModel.register("mamba")
class MambaBGModel(BGModel):
    """SSM-based background intensity model using Mamba.

    Maps input features over time through a Mamba SSM, then projects
    the hidden states to a scalar non-negative intensity with a
    positive weight vector from :class:`BGModel`.
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
        fast_mix_init: float = 0.7,
        device: torch.device | None = None,
        smooth_kernel_size: int | None = None,
    ):
        super().__init__(device=device, scale_init=scale_init)
        self.d_feature = d_feature
        self.d_state = d_state
        self.use_slow_branch = bool(use_slow_branch)

        cls = {"mamba": Mamba, "mamba2": Mamba2}.get(model_type)
        if cls is None:
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

        self.mamba = cls(**mamba_kwargs)
        self.fc_in = torch.nn.Linear(d_feature, d_model, bias=False)
        self.fc_out = torch.nn.Linear(d_model, 1, bias=False)

        if self.use_slow_branch:
            kt = str(slow_kernel_type).lower()
            if kt == "exp":
                self.slow_kernel = ExpKernel(
                    kernel_size=slow_kernel_size,
                    dt=slow_kernel_dt,
                    normalize=slow_kernel_normalize,
                    init_tau=slow_exp_init_tau,
                )
            elif kt == "gamma":
                self.slow_kernel = GammaKernel(
                    kernel_size=slow_kernel_size,
                    dt=slow_kernel_dt,
                    normalize=slow_kernel_normalize,
                    init_k=slow_gamma_init_k,
                    init_beta=slow_gamma_init_beta,
                )
            elif kt in ("lognormal", "logn"):
                self.slow_kernel = LogNormalKernel(
                    kernel_size=slow_kernel_size,
                    dt=slow_kernel_dt,
                    normalize=slow_kernel_normalize,
                    init_mu=slow_logn_init_mu,
                    init_sigma=slow_logn_init_sigma,
                )
            elif kt in ("mix", "mixture"):
                self.slow_kernel = MixtureKernel(
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
                    ]
                )
            else:
                raise ValueError(f"Unknown slow_kernel_type: {slow_kernel_type}")

            fast_mix_init = float(fast_mix_init)
            fast_mix_init = min(max(fast_mix_init, 1e-4), 1.0 - 1e-4)
            init_logit = torch.logit(torch.tensor(fast_mix_init, dtype=torch.float32))
            self.fast_mix_logit = torch.nn.Parameter(init_logit)
        else:
            self.slow_kernel = None
            self.fast_mix_logit = None

        # Optional causal depthwise smoothing to suppress fast spikes while staying online/causal.
        if smooth_kernel_size is not None and smooth_kernel_size > 1:
            h = torch.ones(smooth_kernel_size, dtype=torch.float32) / float(smooth_kernel_size)
            self.register_buffer("smoothing_kernel", h, persistent=False)
        else:
            self.smoothing_kernel = None
        
        if device is not None:
            self.to(device)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        """Project input features to hidden states and apply Mamba SSM.

        Args:
            time_series: input features of shape (B, T, F)

        Returns:
            ``(B, T, 1)`` tensor of (non-negative) intensities.
        """
        ssm_in = self.fc_in(time_series)  # (B, T, d_model)
        fast_out = self.mamba(ssm_in.contiguous())  # (B, T, d_model)
        if self.use_slow_branch:
            # Slow branch explicitly models long-memory response, then fuses with fast Mamba path.
            slow_h = self.slow_kernel(device=ssm_in.device, dtype=ssm_in.dtype)  # (K,)
            slow_out = _causal_depthwise_conv1d(ssm_in, slow_h)  # (B, T, d_model)
            fast_mix = torch.sigmoid(self.fast_mix_logit).to(dtype=fast_out.dtype)
            ssm_out = fast_mix * fast_out + (1.0 - fast_mix) * slow_out
        else:
            ssm_out = fast_out

        if self.smoothing_kernel is not None:
            ssm_out = _causal_depthwise_conv1d(ssm_out, self.smoothing_kernel)
        
        intensity = self.fc_out(ssm_out)
        return intensity  # (B, T, 1)
