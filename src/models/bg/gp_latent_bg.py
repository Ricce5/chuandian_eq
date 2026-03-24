import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba, Mamba2

from src.data.dot_dict import DotDict
from src.distributions import clamp_preserve_gradients
from src.utils.interp import integrate_uniform_time_series, interp_uniform_time_series

from .base import BGModel
from .kernel import _causal_depthwise_conv1d


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

        if self.mc_samples_train < 1 or self.mc_samples_eval < 1:
            raise ValueError("mc_samples_train and mc_samples_eval must be >= 1.")
        if self.num_inducing < 2:
            raise ValueError("num_inducing must be >= 2 for gp_latent_bg.")
        if self.gp_jitter <= 0.0:
            raise ValueError("gp_jitter must be > 0.")

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
        t0 = time_series_times[:, :1]
        if mask is None:
            effective_times = time_series_times
            t1 = time_series_times[:, -1:]
        else:
            mask_bool = mask.to(device=time_series_times.device).bool()
            lengths = mask_bool.long().sum(dim=1).clamp_min(1)
            last_idx = lengths - 1
            t_last = time_series_times.gather(1, last_idx.unsqueeze(1))
            effective_times = torch.where(mask_bool, time_series_times, t_last.expand_as(time_series_times))
            t1 = t_last

        duration = (t1 - t0).clamp_min(1e-6)
        return ((effective_times - t0) / duration).clamp(0.0, 1.0)

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

        summary = self._masked_mean(hidden, mask)
        mu = self.inducing_mu_head(summary).view(-1, self.num_inducing, self.d_latent)
        logvar = self.inducing_logvar_head(summary).view(-1, self.num_inducing, self.d_latent)
        logvar = logvar.clamp(min=-8.0, max=8.0)
        return hidden, mu, logvar

    def _sample_inducing(
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
        inducing, kuu_inv, _ = self._gp_prior_stats(
            device=time_grid.device,
            dtype=time_grid.dtype,
        )
        k_tu = self._gp_kernel(time_grid, inducing) # (batch_size, seq_len, num_inducing)
        alpha = torch.einsum("mn,bnd->bmd", kuu_inv, inducing_values)
        latent_mean = torch.einsum("btm,bmd->btd", k_tu, alpha) 
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
        cond_var = (amplitude_sq - quad).clamp_min(self.gp_jitter) 
        # Var_{t|u} = K_{t,t} - K_{t,u} K_{u,u}^{-1} K_{u,t}, but K_{t,t} = amplitude^2 for RBF kernel with zero noise
        return latent_mean + cond_var.sqrt().unsqueeze(-1) * torch.randn_like(latent_mean)

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
        kl = self._kl_divergence(mu, logvar)
        time_grid = self._normalize_times(time_series_times, time_series_mask)

        sample_latent = self._should_sample_latent()
        num_samples = self._num_mc_samples() if sample_latent else 1

        arrival_times = getattr(batch, "arrival_times", time_series_times).to(
            self.device,
            dtype=time_series_times.dtype,
        )

        intensity_samples = []
        integral_samples = []
        for _ in range(num_samples):
            inducing_values = self._sample_inducing(mu, logvar, sample_latent=sample_latent)
            latent_traj = self._project_inducing_to_grid(
                time_grid,
                inducing_values,
                sample_latent=sample_latent,
            )
            scaled_intensity = self._decode(hidden, latent_traj, time_series_mask)
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
