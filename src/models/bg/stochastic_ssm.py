import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dot_dict import DotDict
from src.distributions import clamp_preserve_gradients
from src.utils.interp import integrate_uniform_time_series, interp_uniform_time_series

from .base import BGModel
from .kernel import _causal_depthwise_conv1d


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 1e-8:
        return -20.0
    return math.log(math.expm1(value))


@BGModel.register("stochastic_ssm")
class StochasticSSMBGModel(BGModel):
    """Stochastic state-space background model with linear-time processing.

    The latent state follows a lightweight OU-like stochastic recurrence:

        x_t = W_in u_t
        h_t = a_t * h_{t-1} + (1 - a_t) * x_t + sigma_t * sqrt(dt_t) * eps_t
        lambda_t = scale * clamp_min(W_out h_t, 0)

    Compared with NJDTPP-style multi-step Euler updates, this keeps the
    per-sequence cost linear in T with only one noise sample per observed time
    step, which is much more practical for very long background sequences.
    """

    def __init__(
        self,
        d_feature: int,
        d_model: int,
        d_state: int,
        scale_init: float = 200.0,
        dt: float = 1.0,
        noise_scale_init: float = 0.05,
        stochastic_eval: bool = False,
        input_dependent_noise: bool = True,
        smooth_kernel_size: int | None = None,
        no_weight_decay: bool = False,
        device: torch.device | None = None,
    ):
        super().__init__(
            device=device,
            scale_init=scale_init,
            no_weight_decay=no_weight_decay,
        )
        self.d_feature = int(d_feature)
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.dt = float(dt)
        self.stochastic_eval = bool(stochastic_eval)
        self.input_dependent_noise = bool(input_dependent_noise)

        self.fc_in = nn.Linear(self.d_feature, self.d_model, bias=False)
        self.state_in = nn.Linear(self.d_model, self.d_state, bias=False)
        self.fc_out = nn.Linear(self.d_state, 1, bias=False)
        self.state_init = nn.Parameter(torch.zeros(self.d_state))
        self.raw_decay = nn.Parameter(torch.full((self.d_state,), _inverse_softplus(0.1)))

        raw_noise_init = _inverse_softplus(noise_scale_init)
        self.raw_noise_scale = nn.Parameter(torch.full((self.d_state,), raw_noise_init))
        if self.input_dependent_noise:
            self.noise_gate = nn.Linear(self.d_feature, self.d_state, bias=False)
        else:
            self.noise_gate = None

        if smooth_kernel_size is not None and smooth_kernel_size > 1:
            h = torch.ones(smooth_kernel_size, dtype=torch.float32) / float(smooth_kernel_size)
            self.register_buffer("smoothing_kernel", h, persistent=False)
        else:
            self.smoothing_kernel = None

        if no_weight_decay:
            self.fc_in.weight._no_weight_decay = True
            self.state_in.weight._no_weight_decay = True
            self.fc_out.weight._no_weight_decay = True
            self.state_init._no_weight_decay = True
            self.raw_decay._no_weight_decay = True
            self.raw_noise_scale._no_weight_decay = True
            if self.noise_gate is not None:
                self.noise_gate.weight._no_weight_decay = True

        if device is not None:
            self.to(device)

    def _constant_delta(
        self,
        batch_size: int,
        seq_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        d_model: int | None = None,
    ) -> torch.Tensor:
        d_model = self.d_state if d_model is None else int(d_model)
        delta = torch.full((batch_size, seq_len), self.dt, device=device, dtype=dtype)
        delta[:, 0] = 0.0
        return delta.unsqueeze(-1).expand(-1, -1, d_model)

    def _should_sample_noise(self) -> bool:
        return self.training or self.stochastic_eval

    def _latent_scan(
        self,
        model_in: torch.Tensor,
        delta: torch.Tensor,
        time_series: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = model_in.shape
        decay_rate = F.softplus(self.raw_decay).view(1, 1, -1).to(model_in.dtype)
        alpha = torch.exp(-decay_rate * delta)
        drive = (1.0 - alpha) * model_in

        if self._should_sample_noise():
            sigma = F.softplus(self.raw_noise_scale).view(1, 1, -1).to(model_in.dtype)
            if self.noise_gate is not None:
                sigma = sigma * torch.sigmoid(self.noise_gate(time_series))
            drive = drive + torch.randn_like(model_in) * sigma * torch.sqrt(delta + 1e-8)

        state = self.state_init.to(device=model_in.device, dtype=model_in.dtype).unsqueeze(0).expand(batch_size, -1)
        states = []
        for t in range(seq_len):
            state = alpha[:, t, :] * state + drive[:, t, :]
            states.append(state)
        return torch.stack(states, dim=1)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = time_series.shape
        model_in = self.state_in(self.fc_in(time_series))
        delta = self._constant_delta(
            batch_size,
            seq_len,
            dtype=model_in.dtype,
            device=model_in.device,
        )
        latent = self._latent_scan(model_in, delta, time_series)
        if self.smoothing_kernel is not None:
            latent = _causal_depthwise_conv1d(latent, self.smoothing_kernel.to(latent))
        return self.fc_out(latent)

    def _compute_shared_nll_terms(self, batch: DotDict, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
        time_series_times, intensity_traj = super()._compute_intensity_traj(batch)
        arrival_times = getattr(batch, "arrival_times", time_series_times).to(
            self.device,
            dtype=time_series_times.dtype,
        )
        intensity = interp_uniform_time_series(
            t=time_series_times,
            x=intensity_traj,
            t_query=arrival_times,
            clamp=True,
        ).squeeze(-1)
        intensity = clamp_preserve_gradients(intensity, eps, float("inf"))
        integral = integrate_uniform_time_series(
            t=time_series_times,
            x=intensity_traj,
            t_start=batch.t_nll_start,
            t_end=batch.t_end,
        ).squeeze(-1).squeeze(-1)
        return intensity, integral

    def nll(self, batch: DotDict, eps: float = 1e-8) -> torch.Tensor:
        intensity, integral = self._compute_shared_nll_terms(batch, eps=eps)
        log_intensity = torch.log(intensity) * batch.nll_event_mask
        return -(log_intensity.sum(dim=1) - integral)

    def nll_change(self, batch: DotDict, log_h_intensity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        intensity, integral = self._compute_shared_nll_terms(batch, eps=eps)
        h_intensity = torch.exp(log_h_intensity)
        denom = clamp_preserve_gradients(h_intensity, eps, float("inf"))
        ratio = intensity / denom
        mask = getattr(batch, "nll_event_mask", None)
        if mask is None:
            raise ValueError("batch must contain 'nll_event_mask' for nll_change computation.")
        log_change = torch.log1p(ratio) * mask
        return -(log_change.sum(dim=1) - integral)
