import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import kl_divergence
from mamba_ssm import Mamba, Mamba2

try:
    import gpytorch
    from linear_operator.operators import DiagLinearOperator
except ModuleNotFoundError:
    gpytorch = None
    DiagLinearOperator = None

from src.data.dot_dict import DotDict
from src.distributions import clamp_preserve_gradients
from src.utils.interp import integrate_uniform_time_series, interp_uniform_time_series

from .base import BGModel
from .kernel import _causal_depthwise_conv1d

logger = logging.getLogger(__name__)


@BGModel.register("gp_latent_bg")
class GPLatentBGModel(BGModel):
    """Background model with a sparse variational GP latent trajectory.

    The model keeps the current background-model interface, but replaces the
    finite-dimensional latent vector with a continuous latent function sampled
    on the observed time grid through GP kernel interpolation.
    """

    def __init__(
        self,
        d_feature: int,
        d_model: int,
        d_latent: int,
        num_inducing: int = 16,
        scale_init: float = 200.0,
        backbone_type: str = "mamba",
        num_layers: int = 1,
        d_state: int = 64,
        beta_kl: float = 1e-3,
        stochastic_eval: bool = False,
        mc_samples_train: int = 1,
        mc_samples_eval: int = 1,
        gp_lengthscale_init: float = 0.2,
        gp_kernel_scale_init: float = 1.0,
        gp_jitter: float = 1e-4,
        sample_gp_residual: bool = True,
        smooth_kernel_size: int | None = None,
        time_normalization: str = "per_sequence",
        global_time_min: float | None = None,
        global_time_max: float | None = None,
        clamp_normalized_time: bool = True,
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
        self.num_inducing = int(num_inducing)
        self.beta_kl = float(beta_kl)
        self.stochastic_eval = bool(stochastic_eval)
        self.mc_samples_train = int(mc_samples_train)
        self.mc_samples_eval = int(mc_samples_eval)
        self.gp_jitter = float(gp_jitter)
        self.sample_gp_residual = bool(sample_gp_residual)
        self.time_normalization = str(time_normalization).strip().lower()
        self.global_time_min = None if global_time_min is None else float(global_time_min)
        self.global_time_max = None if global_time_max is None else float(global_time_max)
        self.clamp_normalized_time = bool(clamp_normalized_time)
        self._auto_global_time_bounds = False
        self._auto_global_time_bounds_logged = False

        if self.mc_samples_train < 1 or self.mc_samples_eval < 1:
            raise ValueError("mc_samples_train and mc_samples_eval must be >= 1.")
        if self.num_inducing < 2:
            raise ValueError("num_inducing must be >= 2 for gp_latent_bg.")
        if float(gp_lengthscale_init) <= 0.0:
            raise ValueError("gp_lengthscale_init must be > 0.")
        if float(gp_kernel_scale_init) <= 0.0:
            raise ValueError("gp_kernel_scale_init must be > 0.")
        if self.gp_jitter <= 0.0:
            raise ValueError("gp_jitter must be > 0.")
        if self.time_normalization not in {"per_sequence", "global"}:
            raise ValueError(
                "time_normalization must be one of {'per_sequence', 'global'}."
            )
        if self.time_normalization == "global":
            has_global_min = self.global_time_min is not None
            has_global_max = self.global_time_max is not None
            if has_global_min != has_global_max:
                raise ValueError(
                    "global_time_min and global_time_max must be both set or both "
                    "unset when time_normalization='global'."
                )
            self._auto_global_time_bounds = not has_global_min
        if (
            self.global_time_min is not None
            and self.global_time_max is not None
            and self.global_time_max <= self.global_time_min
        ):
            raise ValueError("global_time_max must be > global_time_min.")

        model_cls = {"mamba": Mamba, "mamba2": Mamba2}.get(backbone_type.lower())
        if model_cls is None:
            raise ValueError(
                f"Unknown backbone_type: {backbone_type}. Must be 'mamba' or 'mamba2'."
            )

        self.fc_in = nn.Linear(self.d_feature, self.d_model, bias=False)
        self.backbone_layers = nn.ModuleList(
            [
                model_cls(d_model=self.d_model, d_state=int(d_state), d_conv=4)
                for _ in range(int(num_layers))
            ]
        )
        self.inducing_mu_head = nn.Linear(
            self.d_model,
            self.num_inducing * self.d_latent,
            bias=False,
        )
        self.inducing_logvar_head = nn.Linear(
            self.d_model,
            self.num_inducing * self.d_latent,
            bias=False,
        )
        self.latent_to_model = nn.Linear(self.d_latent, self.d_model, bias=False)
        self.decoder_in = nn.Linear(self.d_model * 2, self.d_model, bias=False)
        self.decoder_out = nn.Linear(self.d_model, 1, bias=False)

        self.log_gp_lengthscale = nn.Parameter(
            torch.log(torch.tensor(float(gp_lengthscale_init), dtype=torch.float32))
        )
        self.log_gp_kernel_scale = nn.Parameter(
            torch.log(torch.tensor(float(gp_kernel_scale_init), dtype=torch.float32))
        )
        self.register_buffer(
            "inducing_positions",
            torch.linspace(0.0, 1.0, self.num_inducing, dtype=torch.float32),
        )

        if smooth_kernel_size is not None and smooth_kernel_size > 1:
            kernel = torch.ones(smooth_kernel_size, dtype=torch.float32) / float(smooth_kernel_size)
            self.register_buffer("smoothing_kernel", kernel, persistent=False)
        else:
            self.smoothing_kernel = None

        self._last_kl: torch.Tensor | None = None

        if no_weight_decay:
            self.fc_in.weight._no_weight_decay = True
            self.inducing_mu_head.weight._no_weight_decay = True
            self.inducing_logvar_head.weight._no_weight_decay = True
            self.latent_to_model.weight._no_weight_decay = True
            self.decoder_in.weight._no_weight_decay = True
            self.decoder_out.weight._no_weight_decay = True
            self.log_gp_lengthscale._no_weight_decay = True
            self.log_gp_kernel_scale._no_weight_decay = True
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

    def _default_time_grid(self, time_series: torch.Tensor) -> torch.Tensor:
        batch_size, steps, _ = time_series.shape
        grid = torch.linspace(
            0.0,
            1.0,
            steps,
            device=time_series.device,
            dtype=time_series.dtype,
        )
        return grid.unsqueeze(0).expand(batch_size, -1)

    def _normalize_times(
        self,
        time_series_times: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        time_series_times = time_series_times.to(dtype=torch.float32)
        if mask is None:
            effective_times = time_series_times
            t_last = time_series_times[:, -1:]
        else:
            mask_bool = mask.to(device=time_series_times.device).bool()
            lengths = mask_bool.long().sum(dim=1).clamp_min(1)
            last_idx = lengths - 1
            t_last = time_series_times.gather(1, last_idx.unsqueeze(1))
            effective_times = torch.where(mask_bool, time_series_times, t_last.expand_as(time_series_times))
            # In masked positions, we set time to the last valid time to avoid extrapolation in GP interpolation. The normalization will still be based on the actual last time point.

        if self.time_normalization == "global":
            self._maybe_init_global_time_bounds(effective_times, t_last)
            t0 = torch.full_like(
                t_last,
                fill_value=float(self.global_time_min),
            )
            t1 = torch.full_like(
                t_last,
                fill_value=float(self.global_time_max),
            )
        else:
            t0 = time_series_times[:, :1]
            t1 = t_last

        duration = (t1 - t0).clamp_min(1e-6)
        normalized_times = (effective_times - t0) / duration
        if self.clamp_normalized_time:
            normalized_times = normalized_times.clamp(0.0, 1.0)
        return normalized_times

    def _maybe_init_global_time_bounds(
        self,
        effective_times: torch.Tensor,
        t_last: torch.Tensor,
    ) -> None:
        if not self._auto_global_time_bounds:
            return
        if self.global_time_min is not None and self.global_time_max is not None:
            return

        inferred_min = float(effective_times.amin().item())
        inferred_max = float(t_last.amax().item())
        if inferred_max <= inferred_min:
            inferred_max = inferred_min + 1e-6

        self.global_time_min = inferred_min
        self.global_time_max = inferred_max
        self._auto_global_time_bounds = False
        if not self._auto_global_time_bounds_logged:
            logger.info(
                "Auto-initialized global time bounds for %s: "
                "global_time_min=%.6f, global_time_max=%.6f",
                self.__class__.__name__,
                self.global_time_min,
                self.global_time_max,
            )
            self._auto_global_time_bounds_logged = True

    def _gp_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between x and y with learnable lengthscale and amplitude.
        x: (N, d), y: (M, d) -> (N, M)
        """
        lengthscale = torch.exp(self.log_gp_lengthscale).to(  
            device=x.device,
            dtype=x.dtype,
        ).clamp_min(1e-4)
        amplitude = torch.exp(self.log_gp_kernel_scale).to(
            device=x.device,
            dtype=x.dtype,
        ).clamp_min(1e-4)
        dist2 = (x.unsqueeze(-1) - y.unsqueeze(-2)).pow(2)
        return amplitude.pow(2) * torch.exp(-0.5 * dist2 / lengthscale.pow(2))

    def _gp_prior_stats(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prior statistics for the inducing points: positions, inverse kernel matrix, and log determinant.
        Returns:
            inducing: (num_inducing,)
            kuu_inv: (num_inducing, num_inducing)
            logdet_kuu: scalar
        """
        inducing = self.inducing_positions.to(device=device, dtype=dtype)
        kuu = self._gp_kernel(inducing, inducing)
        eye = torch.eye(self.num_inducing, device=device, dtype=dtype)
        kuu = kuu + self.gp_jitter * eye
        chol = torch.linalg.cholesky(kuu)
        kuu_inv = torch.cholesky_inverse(chol)
        logdet_kuu = 2.0 * torch.log(torch.diagonal(chol)).sum()
        return inducing, kuu_inv, logdet_kuu

    def _encode(
        self,
        time_series: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        backbone_in = self.fc_in(time_series)
        hidden = backbone_in.contiguous()
        for layer in self.backbone_layers:
            hidden = layer(hidden)

        summary = self._masked_mean(hidden, mask) # (batch_size, d_model)
        mu = self.inducing_mu_head(summary).view(-1, self.num_inducing, self.d_latent) # (batch_size, num_inducing, d_latent)
        logvar = self.inducing_logvar_head(summary).view(-1, self.num_inducing, self.d_latent)
        logvar = logvar.clamp(min=-8.0, max=8.0)
        return hidden, mu, logvar

    def _sample_inducing(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        sample_latent: bool | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        should_sample = self._should_sample_latent() if sample_latent is None else bool(sample_latent)
        if not should_sample:
            if num_samples == 1:
                return mu
            return mu.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        std = torch.exp(0.5 * logvar)
        if num_samples == 1:
            return mu + std * torch.randn_like(std)
        noise = torch.randn(
            (num_samples,) + std.shape,
            device=std.device,
            dtype=std.dtype,
        )
        return mu.unsqueeze(0) + std.unsqueeze(0) * noise

    def _project_inducing_to_grid(
        self,
        time_grid: torch.Tensor,
        inducing_values: torch.Tensor,
        *,
        sample_latent: bool | None = None,
    ) -> torch.Tensor:
        """Project inducing point values to the time grid using GP kernel interpolation.
        time_grid: (batch_size, seq_len)
        inducing_values: (batch_size, num_inducing, d_latent)
        Returns:
            latent_mean: (batch_size, seq_len, d_latent)
        """
        # Project f(t) into subspace spanned by k(t, u) for u in inducing positions. 
        inducing, kuu_inv, _ = self._gp_prior_stats(
            device=time_grid.device,
            dtype=time_grid.dtype,
        )
        k_tu = self._gp_kernel(time_grid, inducing) # (batch_size, seq_len, num_inducing)
        if inducing_values.dim() == 3:
            alpha = torch.einsum("mn,bnd->bmd", kuu_inv, inducing_values)
            latent_mean = torch.einsum("btm,bmd->btd", k_tu, alpha)
        elif inducing_values.dim() == 4:
            alpha = torch.einsum("mn,sbnd->sbmd", kuu_inv, inducing_values)
            latent_mean = torch.einsum("btm,sbmd->sbtd", k_tu, alpha)
        else:
            raise ValueError(
                "inducing_values must have shape (B, M, D) or (S, B, M, D)."
            )
        # \mu_{t\mid u} = K_{t,u} K_{u,u}^{-1} u

        should_sample = self._should_sample_latent() if sample_latent is None else bool(sample_latent)
        if not (should_sample and self.sample_gp_residual):
            return latent_mean

        proj = torch.einsum("mn,btn->btm", kuu_inv, k_tu)
        quad = (k_tu * proj).sum(dim=-1)
        amplitude_sq = torch.exp(2.0 * self.log_gp_kernel_scale).to(
            device=time_grid.device,
            dtype=time_grid.dtype,
        )
        # Jitter is only used for stabilizing K_uu inversion; using it as a
        # variance floor can inject artificial noise when kernel amplitude is small.
        cond_var = (amplitude_sq - quad).clamp_min(0.0)
        positive_mask = (cond_var > 0.0).to(dtype=cond_var.dtype)
        sqrt_eps = torch.finfo(cond_var.dtype).eps
        # Keep zero conditional variance exactly zero in forward while avoiding
        # undefined/infinite sqrt gradients at zero in backward.
        residual_std = torch.sqrt(cond_var + (1.0 - positive_mask) * sqrt_eps) * positive_mask
        return latent_mean + residual_std.unsqueeze(-1) * torch.randn_like(latent_mean)

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence between the variational distribution q(u) = N(mu, diag(exp(logvar))) and the GP prior p(u) = N(0, K_{u,u})."""
        _, kuu_inv, logdet_kuu = self._gp_prior_stats(device=mu.device, dtype=mu.dtype)
        mu_bt = mu.transpose(1, 2)
        logvar_bt = logvar.transpose(1, 2)
        var_bt = logvar_bt.exp()

        trace_term = torch.einsum("bdm,m->bd", var_bt, torch.diagonal(kuu_inv))
        quad_term = torch.einsum("bdm,mn,bdn->bd", mu_bt, kuu_inv, mu_bt)
        logdet_q = logvar_bt.sum(dim=-1)
        kl_per_latent = 0.5 * (
            trace_term + quad_term - self.num_inducing + logdet_kuu - logdet_q
        )
        return kl_per_latent.sum(dim=1) # (batch_size, d_latent) -> (batch_size,)

    def _repeat_batch_tensor(self, x: torch.Tensor | None, num_samples: int) -> torch.Tensor | None:
        if x is None or num_samples == 1:
            return x
        if x.dim() == 0:
            return x
        return x.repeat((num_samples,) + (1,) * (x.dim() - 1))

    def _finalize_intensity_traj(self, scaled_intensity: torch.Tensor) -> torch.Tensor:
        intensity_traj = scaled_intensity * self._scale
        return intensity_traj + (intensity_traj.clamp_min(0.0) - intensity_traj).detach()

    def _decode_mc_samples(
        self,
        hidden: torch.Tensor,
        latent_traj: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if latent_traj.dim() == 3:
            return self._decode(hidden, latent_traj, mask).unsqueeze(0)
        if latent_traj.dim() != 4:
            raise ValueError(
                "latent_traj must have shape (B, T, D) or (S, B, T, D)."
            )

        num_samples, batch_size, seq_len, d_latent = latent_traj.shape
        hidden_mc = hidden.repeat((num_samples, 1, 1))
        mask_mc = self._repeat_batch_tensor(mask, num_samples)
        scaled_flat = self._decode(
            hidden_mc,
            latent_traj.reshape(num_samples * batch_size, seq_len, d_latent),
            mask_mc,
        )
        return scaled_flat.reshape(num_samples, batch_size, seq_len, -1)

    def _mc_nll_observation_terms(
        self,
        *,
        time_series_times: torch.Tensor,
        intensity_traj: torch.Tensor,
        arrival_times: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if intensity_traj.dim() == 3:
            intensity_traj = intensity_traj.unsqueeze(0)
        if intensity_traj.dim() != 4:
            raise ValueError(
                "intensity_traj must have shape (B, T, F) or (S, B, T, F)."
            )

        num_samples, batch_size, seq_len, num_features = intensity_traj.shape
        flat_intensity_traj = intensity_traj.reshape(
            num_samples * batch_size,
            seq_len,
            num_features,
        )
        flat_time_series_times = self._repeat_batch_tensor(time_series_times, num_samples)
        flat_arrival_times = self._repeat_batch_tensor(arrival_times, num_samples)
        flat_t_start = self._repeat_batch_tensor(t_start, num_samples)
        flat_t_end = self._repeat_batch_tensor(t_end, num_samples)

        intensity = interp_uniform_time_series(
            t=flat_time_series_times,
            x=flat_intensity_traj,
            t_query=flat_arrival_times,
            clamp=True,
        ).squeeze(-1)
        intensity = clamp_preserve_gradients(intensity, eps, float("inf"))

        integral = integrate_uniform_time_series(
            t=flat_time_series_times,
            x=flat_intensity_traj,
            t_start=flat_t_start,
            t_end=flat_t_end,
        ).squeeze(-1).squeeze(-1)

        return (
            intensity.reshape(num_samples, batch_size, -1),
            integral.reshape(num_samples, batch_size),
        )

    def _decode(
        self,
        hidden: torch.Tensor,
        latent_traj: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        latent_mod = torch.tanh(self.latent_to_model(latent_traj))
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
        time_series_times: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        sample_latent: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, mu, logvar = self._encode(time_series, mask)
        inducing_values = self._sample_inducing(mu, logvar, sample_latent=sample_latent)
        time_grid = self._normalize_times(time_series_times, mask)
        latent_traj = self._project_inducing_to_grid(
            time_grid,
            inducing_values,
            sample_latent=sample_latent,
        )
        scaled_intensity = self._decode(hidden, latent_traj, mask)
        kl = self._kl_divergence(mu, logvar)
        return scaled_intensity, kl

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        time_series = time_series.to(self.device, dtype=torch.float32)
        pseudo_times = self._default_time_grid(time_series)
        scaled_intensity, _ = self._build_scaled_intensity(
            time_series,
            pseudo_times,
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
            time_series_times,
            mask=time_series_mask,
            sample_latent=None,
        )
        intensity_traj = self._finalize_intensity_traj(scaled_intensity)
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
        kl = self._kl_divergence(mu, logvar)
        time_grid = self._normalize_times(time_series_times, time_series_mask)

        sample_latent = self._should_sample_latent()
        num_samples = self._num_mc_samples() if sample_latent else 1

        arrival_times = getattr(batch, "arrival_times", time_series_times).to(
            self.device,
            dtype=time_series_times.dtype,
        )

        inducing_values = self._sample_inducing(
            mu,
            logvar,
            sample_latent=sample_latent,
            num_samples=num_samples,
        )
        latent_traj = self._project_inducing_to_grid(
            time_grid,
            inducing_values,
            sample_latent=sample_latent,
        )
        scaled_intensity = self._decode_mc_samples(hidden, latent_traj, time_series_mask)
        intensity_traj = self._finalize_intensity_traj(scaled_intensity)
        intensity, integral = self._mc_nll_observation_terms(
            time_series_times=time_series_times,
            intensity_traj=intensity_traj,
            arrival_times=arrival_times,
            t_start=batch.t_nll_start,
            t_end=batch.t_end,
            eps=eps,
        )
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
        del eps
        time_series = batch.time_series.to(self.device, dtype=torch.float32)
        time_series_mask = getattr(batch, "time_series_mask", None)
        if time_series_mask is not None:
            time_series_mask = time_series_mask.to(self.device)

        _, mu, logvar = self._encode(time_series, time_series_mask)
        kl = self._kl_divergence(mu, logvar)
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


@BGModel.register("gp_latent_bg_gpytorch")
class GPyTorchGPLatentBGModel(GPLatentBGModel):
    """GP latent background model backed by gpytorch kernels."""

    def __init__(
        self,
        d_feature: int,
        d_model: int,
        d_latent: int,
        num_inducing: int = 16,
        scale_init: float = 200.0,
        backbone_type: str = "mamba",
        num_layers: int = 1,
        d_state: int = 64,
        beta_kl: float = 1e-3,
        stochastic_eval: bool = False,
        mc_samples_train: int = 1,
        mc_samples_eval: int = 1,
        gp_lengthscale_init: float = 0.2,
        gp_kernel_scale_init: float = 1.0,
        gp_jitter: float = 1e-4,
        sample_gp_residual: bool = True,
        smooth_kernel_size: int | None = None,
        time_normalization: str = "per_sequence",
        global_time_min: float | None = None,
        global_time_max: float | None = None,
        clamp_normalized_time: bool = True,
        no_weight_decay: bool = False,
        device: torch.device | None = None,
    ):
        if gpytorch is None:
            raise ModuleNotFoundError(
                "gpytorch is required for GPyTorchGPLatentBGModel. "
                "Install it with `pip install gpytorch`."
            )
        super().__init__(
            d_feature=d_feature,
            d_model=d_model,
            d_latent=d_latent,
            num_inducing=num_inducing,
            scale_init=scale_init,
            backbone_type=backbone_type,
            num_layers=num_layers,
            d_state=d_state,
            beta_kl=beta_kl,
            stochastic_eval=stochastic_eval,
            mc_samples_train=mc_samples_train,
            mc_samples_eval=mc_samples_eval,
            gp_lengthscale_init=gp_lengthscale_init,
            gp_kernel_scale_init=gp_kernel_scale_init,
            gp_jitter=gp_jitter,
            sample_gp_residual=sample_gp_residual,
            smooth_kernel_size=smooth_kernel_size,
            time_normalization=time_normalization,
            global_time_min=global_time_min,
            global_time_max=global_time_max,
            clamp_normalized_time=clamp_normalized_time,
            no_weight_decay=no_weight_decay,
            device=device,
        )
        del self.log_gp_lengthscale
        del self.log_gp_kernel_scale

        self.gp_kernel_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        self.gp_kernel_module.base_kernel.lengthscale = float(gp_lengthscale_init)
        self.gp_kernel_module.outputscale = float(gp_kernel_scale_init) ** 2

        if no_weight_decay:
            for param in self.gp_kernel_module.parameters():
                param._no_weight_decay = True

        if device is not None:
            self.to(device)

    def _to_kernel_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return x.unsqueeze(-1)
        if x.dim() == 2:
            return x.unsqueeze(-1)
        if x.dim() == 3:
            return x
        raise ValueError(f"Expected x to have 1, 2 or 3 dims, got {x.dim()}.")

    def _gp_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_in = self._to_kernel_input(x)
        y_in = self._to_kernel_input(y)
        return self.gp_kernel_module(x_in, y_in).to_dense()

    def _gp_prior_operator(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ):
        inducing = self.inducing_positions.to(device=device, dtype=dtype)
        inducing_in = self._to_kernel_input(inducing)
        kuu_op = self.gp_kernel_module(inducing_in, inducing_in).add_jitter(self.gp_jitter)
        return inducing, kuu_op

    def _gp_prior_stats(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inducing, kuu_op = self._gp_prior_operator(device=device, dtype=dtype)
        eye = torch.eye(self.num_inducing, device=device, dtype=dtype)
        kuu_inv = kuu_op.solve(eye)
        _, logdet_kuu = kuu_op.inv_quad_logdet(logdet=True)
        return inducing, kuu_inv, logdet_kuu

    def _project_inducing_to_grid(
        self,
        time_grid: torch.Tensor,
        inducing_values: torch.Tensor,
        *,
        sample_latent: bool | None = None,
    ) -> torch.Tensor:
        inducing, kuu_op = self._gp_prior_operator(
            device=time_grid.device,
            dtype=time_grid.dtype,
        )
        time_input = self._to_kernel_input(time_grid)
        inducing_input = self._to_kernel_input(inducing)
        k_tu = self.gp_kernel_module(time_input, inducing_input).to_dense()
        flat_inducing_values = inducing_values.reshape(-1, self.num_inducing, self.d_latent)
        alpha = kuu_op.solve(flat_inducing_values).reshape(inducing_values.shape)
        if inducing_values.dim() == 3:
            latent_mean = torch.einsum("btm,bmd->btd", k_tu, alpha)
        elif inducing_values.dim() == 4:
            latent_mean = torch.einsum("btm,sbmd->sbtd", k_tu, alpha)
        else:
            raise ValueError(
                "inducing_values must have shape (B, M, D) or (S, B, M, D)."
            )

        should_sample = self._should_sample_latent() if sample_latent is None else bool(sample_latent)
        if not (should_sample and self.sample_gp_residual):
            return latent_mean

        proj = kuu_op.solve(k_tu.transpose(-1, -2))
        quad = (k_tu * proj.transpose(-1, -2)).sum(dim=-1)
        k_tt_diag = self.gp_kernel_module(time_input, time_input, diag=True)
        cond_var = (k_tt_diag - quad).clamp_min(0.0)
        positive_mask = (cond_var > 0.0).to(dtype=cond_var.dtype)
        sqrt_eps = torch.finfo(cond_var.dtype).eps
        residual_std = torch.sqrt(cond_var + (1.0 - positive_mask) * sqrt_eps) * positive_mask
        return latent_mean + residual_std.unsqueeze(-1) * torch.randn_like(latent_mean)

    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if DiagLinearOperator is None:
            raise ModuleNotFoundError(
                "linear_operator is required for GPyTorchGPLatentBGModel KL computation."
            )

        mu_bt = mu.transpose(1, 2).contiguous()
        var_bt = logvar.transpose(1, 2).contiguous().exp()
        flat_mu = mu_bt.view(-1, self.num_inducing)
        flat_var = var_bt.view(-1, self.num_inducing)
        flat_batch = flat_mu.shape[0]

        _, kuu_op = self._gp_prior_operator(device=mu.device, dtype=mu.dtype)
        prior = gpytorch.distributions.MultivariateNormal(
            mean=torch.zeros_like(flat_mu),
            covariance_matrix=kuu_op.expand(flat_batch, self.num_inducing, self.num_inducing),
        )
        posterior = gpytorch.distributions.MultivariateNormal(
            mean=flat_mu,
            covariance_matrix=DiagLinearOperator(flat_var),
        )
        kl_flat = kl_divergence(posterior, prior)
        return kl_flat.view(mu.shape[0], self.d_latent).sum(dim=1)


if gpytorch is not None:

    class _StandardSVGPTemporalLatent(gpytorch.models.ApproximateGP):
        """Standard multitask SVGP over normalized time in [0, 1]."""

        def __init__(
            self,
            *,
            d_latent: int,
            num_inducing: int,
            inducing_positions: torch.Tensor,
            gp_lengthscale_init: float,
            gp_kernel_scale_init: float,
            gp_jitter: float,
            learn_inducing_locations: bool,
        ):
            batch_shape = torch.Size([int(d_latent)])
            inducing_points = inducing_positions.view(1, num_inducing, 1).repeat(int(d_latent), 1, 1)
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=int(num_inducing),
                batch_shape=batch_shape,
            )
            base_variational_strategy = gpytorch.variational.VariationalStrategy(
                self,
                inducing_points=inducing_points,
                variational_distribution=variational_distribution,
                learn_inducing_locations=bool(learn_inducing_locations),
                jitter_val=float(gp_jitter),
            )
            variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
                base_variational_strategy,
                num_tasks=int(d_latent),
                task_dim=0,
            )
            super().__init__(variational_strategy=variational_strategy)
            self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(batch_shape=batch_shape),
                batch_shape=batch_shape,
            )
            self.covar_module.base_kernel.lengthscale = float(gp_lengthscale_init)
            self.covar_module.outputscale = float(gp_kernel_scale_init) ** 2

        def forward(self, x: torch.Tensor):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
else:
    _StandardSVGPTemporalLatent = None


@BGModel.register("gp_latent_bg_svgp")
class StandardSVGPGPLatentBGModel(GPLatentBGModel):
    """Background model with a pure standard multitask SVGP latent trajectory."""

    def __init__(
        self,
        d_feature: int,
        d_model: int,
        d_latent: int,
        num_inducing: int = 16,
        scale_init: float = 200.0,
        backbone_type: str = "mamba",
        num_layers: int = 1,
        d_state: int = 64,
        beta_kl: float = 1e-3,
        stochastic_eval: bool = False,
        mc_samples_train: int = 1,
        mc_samples_eval: int = 1,
        gp_lengthscale_init: float = 0.2,
        gp_kernel_scale_init: float = 1.0,
        gp_jitter: float = 1e-4,
        sample_gp_residual: bool = True,
        smooth_kernel_size: int | None = None,
        time_normalization: str = "global",
        global_time_min: float | None = None,
        global_time_max: float | None = None,
        clamp_normalized_time: bool = True,
        no_weight_decay: bool = False,
        learn_inducing_locations: bool = True,
        diagonal_sampling: bool = True,
        device: torch.device | None = None,
    ):
        if gpytorch is None:
            raise ModuleNotFoundError(
                "gpytorch is required for StandardSVGPGPLatentBGModel. "
                "Install it with `pip install gpytorch`."
            )
        super().__init__(
            d_feature=d_feature,
            d_model=d_model,
            d_latent=d_latent,
            num_inducing=num_inducing,
            scale_init=scale_init,
            backbone_type=backbone_type,
            num_layers=num_layers,
            d_state=d_state,
            beta_kl=beta_kl,
            stochastic_eval=stochastic_eval,
            mc_samples_train=mc_samples_train,
            mc_samples_eval=mc_samples_eval,
            gp_lengthscale_init=gp_lengthscale_init,
            gp_kernel_scale_init=gp_kernel_scale_init,
            gp_jitter=gp_jitter,
            sample_gp_residual=sample_gp_residual,
            smooth_kernel_size=smooth_kernel_size,
            time_normalization=time_normalization,
            global_time_min=global_time_min,
            global_time_max=global_time_max,
            clamp_normalized_time=clamp_normalized_time,
            no_weight_decay=no_weight_decay,
            device=device,
        )
        del self.inducing_mu_head
        del self.inducing_logvar_head
        del self.log_gp_lengthscale
        del self.log_gp_kernel_scale

        if _StandardSVGPTemporalLatent is None:
            raise ModuleNotFoundError(
                "gpytorch is required for StandardSVGPGPLatentBGModel."
            )
        self.svgp = _StandardSVGPTemporalLatent(
            d_latent=self.d_latent,
            num_inducing=self.num_inducing,
            inducing_positions=self.inducing_positions.detach().clone(),
            gp_lengthscale_init=gp_lengthscale_init,
            gp_kernel_scale_init=gp_kernel_scale_init,
            gp_jitter=self.gp_jitter,
            learn_inducing_locations=learn_inducing_locations,
        )
        self.diagonal_sampling = bool(diagonal_sampling)

        if no_weight_decay:
            for param in self.svgp.parameters():
                param._no_weight_decay = True

        if device is not None:
            self.to(device)

    def _encode_hidden(self, time_series: torch.Tensor) -> torch.Tensor:
        backbone_in = self.fc_in(time_series)
        hidden = backbone_in.contiguous()
        for layer in self.backbone_layers:
            hidden = layer(hidden)
        return hidden

    def _svgp_dist(self, time_grid: torch.Tensor):
        batch_size, seq_len = time_grid.shape
        x_flat = time_grid.reshape(-1, 1).to(device=time_grid.device, dtype=torch.float32)
        latent_dist = self.svgp(x_flat)
        return latent_dist, batch_size, seq_len

    def _svgp_kl(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base_strategy = self.svgp.variational_strategy.base_variational_strategy
        kl_scalar = base_strategy.kl_divergence().sum().to(device=device, dtype=dtype)
        return kl_scalar.expand(batch_size)

    def _sample_latent_traj(
        self,
        time_grid: torch.Tensor,
        *,
        sample_latent: bool | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1.")
        latent_dist, batch_size, seq_len = self._svgp_dist(time_grid)
        latent_mean = latent_dist.mean.reshape(batch_size, seq_len, self.d_latent)
        should_sample = self._should_sample_latent() if sample_latent is None else bool(sample_latent)
        if should_sample and self.sample_gp_residual:
            if self.diagonal_sampling:
                latent_var = latent_dist.variance.reshape(batch_size, seq_len, self.d_latent)
                latent_std = torch.sqrt(latent_var.clamp_min(self.gp_jitter))
                if num_samples == 1:
                    return latent_mean + latent_std * torch.randn_like(latent_mean)
                noise = torch.randn(
                    (num_samples,) + latent_mean.shape,
                    device=latent_mean.device,
                    dtype=latent_mean.dtype,
                )
                return latent_mean.unsqueeze(0) + latent_std.unsqueeze(0) * noise
            if num_samples == 1:
                latent_flat = latent_dist.rsample()
                return latent_flat.reshape(batch_size, seq_len, self.d_latent)
            latent_flat = latent_dist.rsample(sample_shape=torch.Size([num_samples]))
            return latent_flat.reshape(num_samples, batch_size, seq_len, self.d_latent)

        if num_samples == 1:
            return latent_mean
        return latent_mean.unsqueeze(0).repeat(num_samples, 1, 1, 1)

    def _build_scaled_intensity(
        self,
        time_series: torch.Tensor,
        time_series_times: torch.Tensor,
        mask: torch.Tensor | None = None,
        *,
        sample_latent: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_hidden(time_series)
        time_grid = self._normalize_times(time_series_times, mask)
        latent_traj = self._sample_latent_traj(time_grid, sample_latent=sample_latent)
        scaled_intensity = self._decode(hidden, latent_traj, mask)
        kl = self._svgp_kl(
            batch_size=time_series.shape[0],
            device=scaled_intensity.device,
            dtype=scaled_intensity.dtype,
        )
        return scaled_intensity, kl

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

        hidden = self._encode_hidden(time_series)
        time_grid = self._normalize_times(time_series_times, time_series_mask)

        sample_latent = self._should_sample_latent()
        num_samples = self._num_mc_samples() if sample_latent else 1

        arrival_times = getattr(batch, "arrival_times", time_series_times).to(
            self.device,
            dtype=time_series_times.dtype,
        )

        latent_traj = self._sample_latent_traj(
            time_grid,
            sample_latent=sample_latent,
            num_samples=num_samples,
        )
        scaled_intensity = self._decode_mc_samples(hidden, latent_traj, time_series_mask)
        intensity_traj = self._finalize_intensity_traj(scaled_intensity)
        intensity, integral = self._mc_nll_observation_terms(
            time_series_times=time_series_times,
            intensity_traj=intensity_traj,
            arrival_times=arrival_times,
            t_start=batch.t_nll_start,
            t_end=batch.t_end,
            eps=eps,
        )
        kl = self._svgp_kl(
            batch_size=time_series.shape[0],
            device=intensity.device,
            dtype=intensity.dtype,
        )
        self._last_kl = kl.detach()
        return intensity, integral, kl

    def kl_term(self, batch: DotDict, eps: float = 1e-8) -> torch.Tensor:
        del eps
        batch_size = batch.time_series.shape[0]
        kl = self._svgp_kl(
            batch_size=batch_size,
            device=self.device,
            dtype=torch.float32,
        )
        self._last_kl = kl.detach()
        return self.beta_kl * kl
