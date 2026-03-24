import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2

from src.data.dot_dict import DotDict
from src.distributions import clamp_preserve_gradients
from src.utils.interp import integrate_uniform_time_series, interp_uniform_time_series

from .base import BGModel
from .kernel import _causal_depthwise_conv1d


@BGModel.register("latent_bg")
class LatentBGModel(BGModel):
    """Background model with latent variables over background trajectories.

    The model first extracts a deterministic hidden trajectory ``h_t`` with an
    Mamba backbone. It then infers either:
    - a sequence-level posterior ``q(z | X)``, or
    - a time-varying posterior ``q(z_t | h_t)``

    The sampled latent modulates the hidden trajectory before decoding it into
    the background intensity trajectory.

    Using multiplicative latent modulation keeps the zero-input -> zero-output
    property expected by the rest of the background-model stack.
    """

    def __init__(
        self,
        d_feature: int,
        d_model: int,
        d_latent: int,
        scale_init: float = 200.0,
        backbone_type: str = "mamba",
        latent_mode: str = "time_varying",
        num_layers: int = 1,
        d_state: int = 64,
        beta_kl: float = 1e-3,
        stochastic_eval: bool = False,
        mc_samples_train: int = 1,
        mc_samples_eval: int = 1,
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
        self.d_latent = int(d_latent)
        self.latent_mode = str(latent_mode).lower()
        self.beta_kl = float(beta_kl)
        self.stochastic_eval = bool(stochastic_eval)
        self.mc_samples_train = int(mc_samples_train)
        self.mc_samples_eval = int(mc_samples_eval)

        if self.mc_samples_train < 1 or self.mc_samples_eval < 1:
            raise ValueError("mc_samples_train and mc_samples_eval must be >= 1.")

        if self.latent_mode not in {"sequence", "time_varying"}:
            raise ValueError(
                f"Unknown latent_mode: {latent_mode}. Must be 'sequence' or 'time_varying'."
            )

        model_cls = {"mamba": Mamba, "mamba2": Mamba2}.get(backbone_type.lower())
        if model_cls is None:
            raise ValueError(f"Unknown backbone_type: {backbone_type}. Must be 'mamba' or 'mamba2'.")

        self.fc_in = nn.Linear(self.d_feature, self.d_model, bias=False)
        self.backbone_layers = nn.ModuleList(
            [
                model_cls(d_model=self.d_model, d_state=int(d_state), d_conv=4)
                for _ in range(int(num_layers))
            ]
        )
        self.mu_head = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.logvar_head = nn.Linear(self.d_model, self.d_latent, bias=False)
        self.latent_to_model = nn.Linear(self.d_latent, self.d_model, bias=False)
        self.decoder_in = nn.Linear(self.d_model * 2, self.d_model, bias=False)
        self.decoder_out = nn.Linear(self.d_model, 1, bias=False)

        if smooth_kernel_size is not None and smooth_kernel_size > 1:
            kernel = torch.ones(smooth_kernel_size, dtype=torch.float32) / float(smooth_kernel_size)
            self.register_buffer("smoothing_kernel", kernel, persistent=False)
        else:
            self.smoothing_kernel = None

        self._last_kl: torch.Tensor | None = None

        if no_weight_decay:
            self.fc_in.weight._no_weight_decay = True
            self.mu_head.weight._no_weight_decay = True
            self.logvar_head.weight._no_weight_decay = True
            self.latent_to_model.weight._no_weight_decay = True
            self.decoder_in.weight._no_weight_decay = True
            self.decoder_out.weight._no_weight_decay = True
            for param in self.backbone_layers.parameters():
                param._no_weight_decay = True

        if device is not None:
            self.to(device)

    @property
    def last_kl(self) -> torch.Tensor | None:
        return self._last_kl

    def _should_sample_latent(self) -> bool:
        return self.training or self.stochastic_eval

    def _num_mc_samples(self) -> int:
        return self.mc_samples_train if self.training else self.mc_samples_eval

    def _masked_mean(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)
        weight = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        return (x * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _encode(
        self,
        time_series: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_in = self.fc_in(time_series)
        hidden = backbone_in.contiguous()
        for layer in self.backbone_layers:
            hidden = layer(hidden)

        if self.latent_mode == "sequence":
            posterior_input = self._masked_mean(hidden, mask)
        else:
            posterior_input = hidden

        mu = self.mu_head(posterior_input)
        logvar = self.logvar_head(posterior_input).clamp(min=-8.0, max=8.0)
        return hidden, mu, logvar

    def _sample_latent(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        sample_latent: bool | None = None,
    ) -> torch.Tensor:
        should_sample = self._should_sample_latent() if sample_latent is None else bool(sample_latent)
        if not should_sample:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=-1)
        if kl.dim() == 1:
            return kl
        if mask is None:
            return kl.sum(dim=1)
        return (kl * mask.to(device=kl.device, dtype=kl.dtype)).sum(dim=1)

    def _decode(
        self,
        hidden: torch.Tensor,
        latent: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        latent_mod = torch.tanh(self.latent_to_model(latent))
        if latent_mod.dim() == 2:
            latent_mod = latent_mod.unsqueeze(1)
        modulated_hidden = hidden * (1.0 + latent_mod)
        decoder_input = torch.cat([hidden, modulated_hidden], dim=-1)
        scaled_intensity = self.decoder_out(F.silu(self.decoder_in(decoder_input)))
        if self.smoothing_kernel is not None:
            scaled_intensity = _causal_depthwise_conv1d(
                scaled_intensity,
                self.smoothing_kernel.to(scaled_intensity),
            )
        if mask is not None:
            scaled_intensity = scaled_intensity * mask.to(
                device=scaled_intensity.device,
                dtype=scaled_intensity.dtype,
            ).unsqueeze(-1)
        return scaled_intensity

    def _build_scaled_intensity(
        self,
        time_series: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        sample_latent: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mu, logvar = self._encode(time_series, mask)
        latent = self._sample_latent(mu, logvar, sample_latent=sample_latent)
        scaled_intensity = self._decode(hidden, latent, mask)
        kl = self._kl_divergence(mu, logvar, mask)
        return scaled_intensity, kl

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        scaled_intensity, _ = self._build_scaled_intensity(
            time_series,
            mask=None,
            sample_latent=None,
        )
        return scaled_intensity

    def _compute_intensity_traj(self, ts_batch: DotDict):
        time_series = ts_batch.time_series.to(self.device, dtype=torch.float32)
        time_series_times = ts_batch.time_series_times.to(self.device)
        time_series_mask = getattr(ts_batch, "time_series_mask", None)
        if time_series_mask is not None:
            time_series_mask = time_series_mask.to(self.device)

        scaled_intensity, _ = self._build_scaled_intensity(
            time_series,
            mask=time_series_mask,
            sample_latent=None,
        )
        intensity_traj = scaled_intensity * self._scale
        intensity_traj = intensity_traj + (intensity_traj.clamp_min(0.0) - intensity_traj).detach()
        return time_series_times, intensity_traj

    def _compute_shared_nll_terms(
        self,
        batch: DotDict,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_series = batch.time_series.to(self.device, dtype=torch.float32)
        time_series_times = batch.time_series_times.to(self.device)
        time_series_mask = getattr(batch, "time_series_mask", None)
        if time_series_mask is not None:
            time_series_mask = time_series_mask.to(self.device)

        hidden, mu, logvar = self._encode(time_series, time_series_mask)
        kl = self._kl_divergence(mu, logvar, time_series_mask)

        sample_latent = self._should_sample_latent()
        num_samples = self._num_mc_samples() if sample_latent else 1

        arrival_times = getattr(batch, "arrival_times", time_series_times).to(
            self.device,
            dtype=time_series_times.dtype,
        )

        intensity_samples = []
        integral_samples = []
        for _ in range(num_samples):
            latent = self._sample_latent(mu, logvar, sample_latent=sample_latent)
            scaled_intensity = self._decode(hidden, latent, time_series_mask)
            intensity_traj = scaled_intensity * self._scale
            intensity_traj = intensity_traj + (intensity_traj.clamp_min(0.0) - intensity_traj).detach()

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

            intensity_samples.append(intensity)
            integral_samples.append(integral)

        intensity = torch.stack(intensity_samples, dim=0)
        integral = torch.stack(integral_samples, dim=0)
        self._last_kl = kl.detach()
        return intensity, integral, kl

    def nll(self, batch: DotDict, eps: float = 1e-8) -> torch.Tensor:
        intensity_samples, integral_samples, kl = self._compute_shared_nll_terms(batch, eps=eps)
        event_mask = batch.nll_event_mask.unsqueeze(0).to(
            device=intensity_samples.device,
            dtype=intensity_samples.dtype,
        )
        log_intensity = torch.log(intensity_samples) * event_mask
        expected_log_sum = log_intensity.sum(dim=2).mean(dim=0)
        expected_integral = integral_samples.mean(dim=0)
        return -(expected_log_sum - expected_integral) + self.beta_kl * kl

    def kl_term(self, batch: DotDict, eps: float = 1e-8) -> torch.Tensor:
        """Return KL regularization term only (beta_kl * KL) for each sequence."""
        time_series = batch.time_series.to(self.device, dtype=torch.float32)
        time_series_mask = getattr(batch, "time_series_mask", None)
        if time_series_mask is not None:
            time_series_mask = time_series_mask.to(self.device)

        _, mu, logvar = self._encode(time_series, time_series_mask)
        kl = self._kl_divergence(mu, logvar, time_series_mask)
        self._last_kl = kl.detach()
        return self.beta_kl * kl

    def nll_change(self, batch: DotDict, log_h_intensity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        intensity_samples, integral_samples, kl = self._compute_shared_nll_terms(batch, eps=eps)
        h_intensity = torch.exp(log_h_intensity)
        denom = clamp_preserve_gradients(h_intensity, eps, float("inf"))
        ratio = intensity_samples / denom.unsqueeze(0)
        mask = getattr(batch, "nll_event_mask", None)
        if mask is None:
            raise ValueError("batch must contain 'nll_event_mask' for nll_change computation.")
        mask = mask.unsqueeze(0).to(device=ratio.device, dtype=ratio.dtype)
        log_change = torch.log1p(ratio) * mask
        expected_log_change = log_change.sum(dim=2).mean(dim=0)
        expected_integral = integral_samples.mean(dim=0)
        return -(expected_log_change - expected_integral) + self.beta_kl * kl
