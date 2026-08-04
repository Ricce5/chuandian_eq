# Modular RNN-based TPP model for flexible parametrization experiments.
from __future__ import annotations

import inspect
from typing import Any

import torch

import src
import src.distributions as dist

from ..common.inter_time_decoding import WeibullMixtureDecoder
from ..common.recurrent_blocks import FiLMContextFuse, RNNTPPBackbone
from .sampling import RecurrentTPPSamplingMixin
from ..tpp_model import TPPModel


class RecurrentTPPV2(RecurrentTPPSamplingMixin, TPPModel):
    """RNN TPP with decoupled backbone and inter-time parametrization."""

    def __init__(
        self,
        args,
        *,
        backbone: RNNTPPBackbone,
        hypernet_time: torch.nn.Module,
        hypernet_mag: torch.nn.Module,
        device=None,
        bg_model=None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.backbone = backbone
        self.hypernet_time = hypernet_time
        self.hypernet_mag = hypernet_mag
        self.bg_model = bg_model

        self.input_magnitude = bool(self.backbone.input_magnitude)
        self.num_extra_features = self.backbone.num_extra_features
        self.context_size = self.backbone.context_size
        self.num_components = int(args.num_components)
        self.scale_range = getattr(args, "scale_range", "positive")
        if self.scale_range not in {"positive", "decay"}:
            raise ValueError("scale_range must be one of ['positive', 'decay']")

        self.inter_time_decoder = WeibullMixtureDecoder(
            num_components=self.num_components,
            parametrization=getattr(args, "rtpp_time_parametrization", "legacy"),
            scale_range=self.scale_range,
            normalize_mixture_logits=getattr(args, "rtpp_normalize_mixture_logits", True),
        )

        self.register_buffer("time_max", torch.tensor(args.time_max, dtype=torch.float32))
        self.register_buffer("richter_b", torch.tensor(args.richter_b_mle, dtype=torch.float32))
        self.register_buffer(
            "mag_completeness", torch.tensor(args.mag_completeness, dtype=torch.float32)
        )
        mag_max = float(getattr(args, "mag_max", 10.0))
        if mag_max <= float(args.mag_completeness):
            raise ValueError("mag_max must be greater than mag_completeness.")
        self.register_buffer("M_m", torch.tensor(mag_max, dtype=torch.float32))
        self.predict_b = bool(getattr(args, "predict_b", False))
        self.use_b_updater = bool(getattr(args, "use_b_updater", False))
        self.b_updater_name = getattr(args, "b_updater_name", None)
        if self.b_updater_name is not None:
            self.b_updater_name = str(self.b_updater_name)
        self.b_updater_cfg = dict(getattr(args, "b_updater_cfg", {}) or {})
        self.b_use_bg_context = bool(getattr(args, "b_use_bg_context", False))
        self.time_use_bg_context = bool(getattr(args, "time_use_bg_context", False))
        self.weights = self._resolve_loss_weights(args)
        self.mag_loss_weight = self.weights["mag_weight"]
        self.bg_norm_weight = self.weights["bg_norm_weight"]
        self.b_init = float(getattr(args, "b_init", 1.0))
        b_range = getattr(args, "b_range", None)
        if b_range is None:
            self.b_min, self.b_max = 0.5, 2.0
        else:
            self.b_min, self.b_max = float(b_range[0]), float(b_range[1])

        self.b_context_fuse: torch.nn.Module | None = None
        self.time_context_fuse: torch.nn.Module | None = None
        if self.b_use_bg_context or self.time_use_bg_context:
            if self.bg_model is None or not hasattr(self.bg_model, "_get_bg_context"):
                raise ValueError(
                    "time/b context fusion requires bg_model with _get_bg_context()."
                )
            bg_context_dim = self._resolve_bg_context_dim(self.bg_model, self.context_size)
            if self.b_use_bg_context:
                self.b_context_fuse = FiLMContextFuse(
                    context_dim=self.context_size,
                    bg_context_dim=bg_context_dim,
                )
            if self.time_use_bg_context:
                self.time_context_fuse = FiLMContextFuse(
                    context_dim=self.context_size,
                    bg_context_dim=bg_context_dim,
                )

        self.reduction = getattr(args, "loss_reduction", "per_time")
        self.to(self.device)

    @property
    def tau_mean(self) -> torch.Tensor:
        return self.backbone.tau_mean

    @property
    def log_tau_mean(self) -> torch.Tensor:
        return self.backbone.log_tau_mean

    @property
    def mag_mean(self) -> torch.Tensor:
        return self.backbone.mag_mean

    @property
    def rnn(self) -> torch.nn.Module:
        return self.backbone.rnn

    @property
    def dropout(self) -> torch.nn.Module:
        return self.backbone.dropout

    def encode_time(self, inter_times: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_time(inter_times)

    def encode_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_magnitude(mag)

    def encode_extra_features(self, extra_feat: torch.Tensor) -> torch.Tensor:
        return self.backbone.encode_extra_features(extra_feat)

    def get_context(self, batch: src.data.Batch) -> torch.Tensor:
        return self.backbone.get_context(batch)

    def _get_context_and_hidden(
        self,
        batch: src.data.Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.backbone.get_context(batch)
        hidden = self.backbone.get_hidden_after_events(batch)
        return context, hidden

    def get_inter_time_dist(self, context: torch.Tensor) -> dist.MixtureSameFamily:
        return self.inter_time_decoder.from_context(context, self.hypernet_time)

    def forward(self, batch: src.data.Batch):
        return self.backbone(batch)

    @staticmethod
    def _resolve_bg_context_dim(bg_model: torch.nn.Module, fallback_dim: int) -> int:
        return int(getattr(bg_model, "context_dim", getattr(bg_model, "fast_out_dim", fallback_dim)))

    @staticmethod
    def _resolve_loss_weights(args) -> dict[str, float]:
        raw_cfg = dict(getattr(args, "loss_weights", {}) or {})
        if not raw_cfg:
            for legacy_key in (
                "time_weight",
                "mag_weight",
                "b_weight",
                "b_smooth_weight",
                "bg_weight",
                "bg_kl_weight",
                "bg_norm_weight",
            ):
                if hasattr(args, legacy_key):
                    raw_cfg[legacy_key] = getattr(args, legacy_key)
        weights = {
            "time_weight": float(raw_cfg.pop("time_weight", 1.0)),
            "mag_weight": float(raw_cfg.pop("mag_weight", 1.0)),
            "b_weight": float(raw_cfg.pop("b_weight", 0.0)),
            "b_smooth_weight": float(raw_cfg.pop("b_smooth_weight", 0.0)),
            "bg_weight": float(raw_cfg.pop("bg_weight", 1.0)),
            "bg_kl_weight": float(raw_cfg.pop("bg_kl_weight", 1.0)),
            "bg_norm_weight": float(raw_cfg.pop("bg_norm_weight", 0.0)),
        }
        if raw_cfg:
            unknown = ", ".join(sorted(raw_cfg.keys()))
            raise ValueError(f"Unknown loss_weights keys: {unknown}")

        for key, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{key} must be >= 0, got {value}.")
        return weights

    def _context_query_times(self, batch: src.data.Batch) -> torch.Tensor:
        return torch.cat([batch.t_start[:, None], batch.arrival_times[:, :-1]], dim=1)

    def _get_bg_context(
        self,
        batch: src.data.Batch,
        *,
        dtype: torch.dtype,
        expected_len: int,
    ) -> torch.Tensor:
        if self.bg_model is None:
            raise ValueError("bg_model is required when using bg context.")
        query_times = self._context_query_times(batch)
        bg_ctx = self.bg_model._get_bg_context(
            batch,
            query_times,
            clamp=True,
            left_endpoint=True,
        )
        if bg_ctx.shape[1] != expected_len:
            raise ValueError(
                f"Background context length mismatch: expected {expected_len}, got {bg_ctx.shape[1]}."
            )
        if hasattr(batch, "input_mask"):
            mask = batch.input_mask[:, :expected_len].to(
                device=bg_ctx.device,
                dtype=bg_ctx.dtype,
            )
            bg_ctx = bg_ctx * mask.unsqueeze(-1)
        return bg_ctx.to(dtype=dtype)

    def _get_bg_context_from_query_times(
        self,
        query_times: torch.Tensor,
        *,
        dtype: torch.dtype,
        expected_len: int,
    ) -> torch.Tensor:
        if self.bg_model is None:
            raise ValueError("bg_model is required when using bg context.")
        cached_batch = getattr(self.bg_model, "ts_batch_cache", None)
        if cached_batch is None:
            raise ValueError(
                "bg_model.ts_batch_cache is required for time context fusion during sampling."
            )

        bg_ctx = self.bg_model._get_bg_context(
            cached_batch,
            query_times,
            clamp=True,
            left_endpoint=True,
        )
        if bg_ctx.shape[1] != expected_len:
            raise ValueError(
                f"Background context length mismatch: expected {expected_len}, got {bg_ctx.shape[1]}."
            )
        return bg_ctx.to(dtype=dtype)

    def _get_b_context(
        self,
        context: torch.Tensor,
        *,
        batch: src.data.Batch | None = None,
    ) -> torch.Tensor:
        if not self.b_use_bg_context or self.b_context_fuse is None or batch is None:
            return context
        bg_ctx = self._get_bg_context(
            batch,
            dtype=context.dtype,
            expected_len=context.shape[1],
        )
        return self.b_context_fuse(context, bg_ctx)

    def _get_time_context(
        self,
        context: torch.Tensor,
        *,
        batch: src.data.Batch | None = None,
    ) -> torch.Tensor:
        if not self.time_use_bg_context or self.time_context_fuse is None or batch is None:
            return context
        bg_ctx = self._get_bg_context(
            batch,
            dtype=context.dtype,
            expected_len=context.shape[1],
        )
        return self.time_context_fuse(context, bg_ctx)

    def _get_sampling_time_context(
        self,
        *,
        current_state: torch.Tensor,
        t_last_event: torch.Tensor,
        lower_bound: torch.Tensor | float | None = None,
    ) -> torch.Tensor:
        if not self.time_use_bg_context or self.time_context_fuse is None:
            return current_state

        # The time decoder is trained with the background context at the start
        # of an inter-event interval.  For the first forecast event after a
        # censored historical interval, ``lower_bound`` is used only to
        # condition the waiting-time distribution; it must not move the
        # decoder context from the last event to the forecast boundary.
        del lower_bound
        query_times = t_last_event
        if query_times.ndim == 1:
            query_times = query_times.unsqueeze(-1)

        bg_ctx = self._get_bg_context_from_query_times(
            query_times,
            dtype=current_state.dtype,
            expected_len=current_state.shape[1],
        )
        return self.time_context_fuse(current_state, bg_ctx)

    def _get_sampling_b_context(
        self,
        *,
        current_state: torch.Tensor,
        t_last_event: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse the b-value decoder with background context during sampling.

        This mirrors ``_get_b_context`` during training: both query the
        background trajectory at the latest event, i.e. at the beginning of
        the mark/inter-event conditional distribution.
        """
        if not self.b_use_bg_context or self.b_context_fuse is None:
            return current_state
        query_times = t_last_event
        if query_times.ndim == 1:
            query_times = query_times.unsqueeze(-1)
        bg_ctx = self._get_bg_context_from_query_times(
            query_times,
            dtype=current_state.dtype,
            expected_len=current_state.shape[1],
        )
        return self.b_context_fuse(current_state, bg_ctx)

    def _get_b_pred(
        self,
        context: torch.Tensor,
        *,
        batch: src.data.Batch | None = None,
        predict_b: bool | None = None,
    ) -> torch.Tensor:
        predict_b = self.predict_b if predict_b is None else predict_b
        if not predict_b:
            return context.new_full(context.shape[:2], float(self.richter_b))

        b_context = self._get_b_context(context, batch=batch)
        b_raw = self.hypernet_mag(b_context).squeeze(-1) + self.b_init
        clamped = b_raw.clamp(self.b_min, self.b_max)
        return b_raw + (clamped - b_raw).detach()
    

    def compute_b_value(
        self,
        seq: src.data.Sequence,
        predict_b: bool = False,
    ) -> torch.Tensor:
        """Compute per-step b-value predictions for one historical sequence.

        Returns:
            Tensor of shape (1, L), aligned with model context steps.
        """
        if seq is None:
            raise ValueError("seq cannot be None when calling compute_b_value().")
        past_batch = src.data.Batch.from_list([seq]).to(self.device)
        context = self.get_context(past_batch)
        return self._get_b_pred(context, batch=past_batch, predict_b=predict_b)

    def get_magnitude_dist(
        self,
        context: torch.Tensor,
        *,
        batch: src.data.Batch | None = None,
        predict_b: bool | None = None,
        return_b: bool = False,
    ):
        b = self._get_b_pred(context, batch=batch, predict_b=predict_b)
        mag_min = self.mag_completeness * torch.ones_like(b)
        mag_max = self.M_m * torch.ones_like(b)
        gr = dist.GutenbergRichter(b=b, mag_min=mag_min, mag_max=mag_max)
        if return_b:
            return gr, b
        return gr

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 0.0,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        context = self.get_context(batch)
        time_context = self._get_time_context(context, batch=batch)

        inter_time_dist = self.get_inter_time_dist(time_context)
        d_t_obs = batch.inter_times
        log_like = self.time_log_likelihood(
            batch=batch,
            inter_time_dist=inter_time_dist,
            state=time_context,
            dist_from_state=self.get_inter_time_dist,
            pdf_inter_times=d_t_obs,
            survival_inter_times=batch.inter_times,
        )

        reduction = self.reduction if reduction is None else reduction
        weights = self.weights

        nll_trigger_time = -log_like
        nll_time = nll_trigger_time
        nll_total = weights["time_weight"] * nll_time
        out = {
            "time": nll_time,
            "total": nll_total,
        }
        if self.predict_b and weights["mag_weight"] > 0.0:
            mag_dist = self.get_magnitude_dist(context, batch=batch, predict_b=True)
            mag_mask = batch.nll_event_mask.bool()
            log_like_mag = mag_dist.log_likelihood(batch.mag, mag_mask)
            nll_mag = -log_like_mag
            nll_total = nll_total + weights["mag_weight"] * nll_mag
            out["mag"] = nll_mag
            out["total"] = nll_total

        if self.bg_model is not None:
            nll_bg, bg_kl, bg_norm = self._compute_bg_terms(
                batch,
                inter_time_dist,
                eps=eps,
            )

            nll_time = nll_trigger_time + nll_bg
            nll_total = weights["time_weight"] * nll_trigger_time + weights["bg_weight"] * nll_bg
            if bg_kl is not None:
                nll_total = nll_total + weights["bg_kl_weight"] * bg_kl
            if bg_norm is not None:
                nll_total = nll_total + weights["bg_norm_weight"] * bg_norm

            out["time"] = nll_time
            out["bg"] = nll_bg
            if bg_kl is not None:
                out["bg_kl"] = bg_kl
            if bg_norm is not None:
                out["bg_norm"] = bg_norm
            out["total"] = nll_total

        out = self.reduce_nll_dict(out, batch, reduction=reduction, eps=eps)
        if return_dict:
            return out
        return out["total"]
    
    def _bg_model_supports_include_kl(self) -> bool:
        nll_change_sig = inspect.signature(self.bg_model.nll_change)
        return "include_kl" in nll_change_sig.parameters

    def _compute_bg_terms(
        self,
        batch: src.data.Batch,
        inter_time_dist: dist.MixtureSameFamily,
        *,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        log_h_intensity = inter_time_dist.log_hazard(batch.inter_times.clamp_min(eps))

        supports_include_kl = self._bg_model_supports_include_kl()
        if supports_include_kl:
            nll_bg_raw = self.bg_model.nll_change(batch, log_h_intensity, include_kl=False)
        else:
            nll_bg_raw = self.bg_model.nll_change(batch, log_h_intensity)

        bg_kl = self.bg_model.kl_term(batch, eps=eps) if hasattr(self.bg_model, "kl_term") else None

        nll_bg = nll_bg_raw
        if bg_kl is not None and not supports_include_kl:
            nll_bg = nll_bg_raw - bg_kl

        bg_norm = None
        if self.weights["bg_norm_weight"] > 0.0 and hasattr(self.bg_model, "normalizing_term"):
            bg_norm = self.bg_model.normalizing_term(
                batch,
                log_h_intensity,
                eps=eps,
            )

        return nll_bg, bg_kl, bg_norm

    def _sampling_state_step(
        self,
        rnn_input: torch.Tensor,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backbone.step(rnn_input, current_hidden)

    def _build_sampling_b_updater(
        self,
        *,
        updater_name: str | None = None,
        updater_cfg: dict[str, Any] | None = None,
    ):
        should_build = self.use_b_updater or updater_name is not None or bool(updater_cfg)
        if not should_build:
            return None
        from src.models.updaters import build_sampling_updater

        merged_cfg = dict(self.b_updater_cfg)
        if updater_cfg:
            merged_cfg.update(dict(updater_cfg))
        return build_sampling_updater(
            model=self,
            updater_name=updater_name or self.b_updater_name,
            updater_cfg=merged_cfg,
            device=self.device,
        )
