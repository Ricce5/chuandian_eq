"""Neural-ETAS with fixed nonnegative basis kernels.

This module keeps the main :class:`NETAS` model while implementation details
for fixed basis kernels, Mamba encoding, and small sampling containers live in
focused sibling modules.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import src
from src.data.batch import Batch, get_mask
from src.data.sequence import Sequence
from src.utils.mask_utils import masked_select_per_row

from ..common.etas_utils import iter_chunks, resolve_chunk_size
from ..common.recurrent_blocks import RNNTPPBackbone
from .basis import FixedKernelBasis
from .encoder import MambaNETASEncoder
from .sampling import NETASSamplingMixin
from .types import (
    _HistorySelection,
    _MagnitudeSamplingParams,
    _ParentParameters,
)
from ..tpp_model import TPPModel

logger = logging.getLogger(__name__)


class NETAS(NETASSamplingMixin, TPPModel):
    """Neural-basis ETAS trigger model with fixed nonnegative kernels.

    The model parameterizes each parent event with a productivity ``eta_i`` and
    basis-mixture weights ``omega_i`` from an event encoder, then evaluates
    ``sum_i eta_i sum_l omega_il q_l(t - t_i)`` plus optional background
    intensity.  The basis kernels provide closed-form integrals for likelihood
    training and branching-process sampling.

    ``max_history_events`` and ``history_time_window`` are optional approximation
    controls for long catalogs.  They keep only recent parents by count and/or
    lag for trigger intensity and integral calculations while preserving the
    exact full-history behavior when left at ``0`` / ``None``.
    """

    def __init__(
        self,
        *,
        event_encoder: nn.Module,
        context_size: int,
        basis_family: str = "exponential",
        num_basis: int = 4,
        basis_rates: Optional[torch.Tensor] = None,
        basis_scales: Optional[torch.Tensor] = None,
        basis_shapes: Optional[torch.Tensor] = None,
        basis_rate_min: float = 1e-3,
        basis_rate_max: float = 1e1,
        basis_scale_min: float = 1e-2,
        basis_scale_max: float = 1e2,
        basis_lomax_shape: float = 0.35,
        basis_learnable: str = "fixed",
        basis_max_log_deviation: Optional[float] = 0.0,
        basis_learn_shapes: bool = False,
        base_rate_init: Union[float, torch.Tensor] = 0.02,
        productivity_alpha_init: float = 1.0,
        productivity_bias_init: float = 0.0,
        head_init_std: float = 1e-2,
        productivity_mode: str = "bounded",
        eta_max: float = 0.95,
        branching_penalty_weight: float = 0.0,
        branching_penalty_target: float = 0.95,
        richter_b: float = 1.0,
        mag_completeness: float = 2.0,
        mag_max: float = 10.0,
        device: Optional[torch.device] = None,
        bg_model=None,
        fix_mu: bool = False,
        fixed_mu_value: Optional[float] = None,
        loss_reduction: str = "per_time",
        query_chunk_size: int = 0,
        history_chunk_size: int = 0,
        max_history_events: int = 0,
        history_time_window: Optional[float] = None,
        loss_weights: Optional[dict[str, float]] = None,
    ) -> None:
        """Initialize NETAS.

        Args:
            event_encoder: Module that returns per-event contexts with shape
                ``[batch, seq_len, context_size]``.
            context_size: Dimension of event encoder contexts.
            basis_family: Fixed kernel family, currently ``"exponential"`` or
                ``"lomax"``.
            num_basis: Number of basis kernels.
            basis_rates: Optional explicit exponential rates.
            basis_scales: Optional explicit Lomax scales.
            basis_shapes: Optional explicit Lomax shapes.
            basis_rate_min: Minimum log-spaced exponential rate when
                ``basis_rates`` is omitted.
            basis_rate_max: Maximum log-spaced exponential rate when
                ``basis_rates`` is omitted.
            basis_scale_min: Minimum log-spaced Lomax scale when
                ``basis_scales`` is omitted.
            basis_scale_max: Maximum log-spaced Lomax scale when
                ``basis_scales`` is omitted.
            basis_lomax_shape: Default Lomax shape when explicit shapes are
                omitted.
            basis_learnable: Whether and how kernel parameters may adapt.
            basis_max_log_deviation: Bound for learnable log-parameter offsets.
            basis_learn_shapes: Whether learnable Lomax mode also adapts shapes.
            base_rate_init: Initial exogenous rate before optional fixing.
            productivity_alpha_init: Initial magnitude coefficient for
                productivity.
            productivity_bias_init: Initial productivity bias.
            head_init_std: Standard deviation for productivity/mixture head
                initialization.
            productivity_mode: Transform for productivity scores:
                ``"bounded"``, ``"softplus"``, or ``"exp"``.
            eta_max: Upper bound used by ``productivity_mode="bounded"``.
            branching_penalty_weight: Legacy direct branching penalty weight.
            branching_penalty_target: Target mean branching ratio for penalty.
            richter_b: Gutenberg-Richter magnitude parameter.
            mag_completeness: Magnitude completeness threshold.
            mag_max: Maximum magnitude used for truncated GR sampling.
            device: Device used by model tensors.
            bg_model: Optional background intensity model.
            fix_mu: If true, use ``fixed_mu_value`` rather than learned ``mu``.
            fixed_mu_value: Fixed exogenous rate when ``fix_mu=True``.
            loss_reduction: Default likelihood reduction mode.
            query_chunk_size: Number of query times per trigger chunk; ``0``
                means full query length unless history truncation is enabled.
            history_chunk_size: Number of history events per trigger chunk;
                ``0`` means full selected history length.
            max_history_events: If positive, only the most recent this many
                parents before each query contribute to trigger terms.
            history_time_window: If set, parents older than this lag are ignored.
            loss_weights: Optional weights for auxiliary loss terms.
        """
        super().__init__()
        if int(context_size) < 1:
            raise ValueError("context_size must be >= 1.")
        productivity_mode = str(productivity_mode).strip().lower()
        if productivity_mode not in {"bounded", "softplus", "exp"}:
            raise ValueError("productivity_mode must be one of ['bounded', 'softplus', 'exp'].")
        if eta_max <= 0.0:
            raise ValueError("eta_max must be positive.")
        if productivity_mode == "bounded" and eta_max >= 1.0:
            raise ValueError(
                "eta_max must lie in (0, 1) when productivity_mode='bounded'."
            )
        if branching_penalty_weight < 0.0:
            raise ValueError("branching_penalty_weight must be non-negative.")
        if branching_penalty_target <= 0.0:
            raise ValueError("branching_penalty_target must be positive.")
        if head_init_std < 0.0:
            raise ValueError("head_init_std must be non-negative.")
        if mag_max <= mag_completeness:
            raise ValueError("mag_max must be greater than mag_completeness.")
        if richter_b <= 0:
            raise ValueError("richter_b must be strictly positive.")
        if int(max_history_events) < 0:
            raise ValueError("max_history_events must be non-negative.")
        if history_time_window is not None and float(history_time_window) < 0.0:
            raise ValueError("history_time_window must be non-negative.")

        self.device = device if device is not None else torch.device("cpu")
        self.event_encoder = event_encoder
        self.context_size = int(context_size)
        self.bg_model = bg_model
        self.fix_mu = bool(fix_mu)
        self.reduction = str(loss_reduction)
        self.query_chunk_size = int(query_chunk_size)
        self.history_chunk_size = int(history_chunk_size)
        self.max_history_events = int(max_history_events)
        self.history_time_window = (
            None
            if history_time_window is None or float(history_time_window) == 0.0
            else float(history_time_window)
        )
        self.productivity_mode = productivity_mode
        self.branching_penalty_target = float(branching_penalty_target)

        self.basis = FixedKernelBasis(
            family=basis_family,
            num_basis=int(num_basis),
            rates=basis_rates,
            scales=basis_scales,
            shapes=basis_shapes,
            rate_min=float(basis_rate_min),
            rate_max=float(basis_rate_max),
            scale_min=float(basis_scale_min),
            scale_max=float(basis_scale_max),
            lomax_shape=float(basis_lomax_shape),
            learnable=basis_learnable,
            max_log_deviation=basis_max_log_deviation,
            learn_shapes=bool(basis_learn_shapes),
        )

        self.productivity_head = nn.Linear(self.context_size, 1, bias=False)
        self.productivity_alpha = nn.Parameter(
            torch.tensor(float(productivity_alpha_init), dtype=torch.float32)
        )
        self.productivity_bias = nn.Parameter(
            torch.tensor(float(productivity_bias_init), dtype=torch.float32)
        )
        self.mixture_head = nn.Linear(self.context_size, self.basis.num_basis, bias=False)
        if float(head_init_std) > 0.0:
            nn.init.normal_(
                self.productivity_head.weight,
                mean=0.0,
                std=float(head_init_std),
            )
            nn.init.normal_(
                self.mixture_head.weight,
                mean=0.0,
                std=float(head_init_std),
            )
        else:
            nn.init.zeros_(self.productivity_head.weight)
            nn.init.zeros_(self.mixture_head.weight)

        self.register_buffer("eta_max", torch.tensor(float(eta_max), dtype=torch.float32))
        self.register_buffer("M_c", torch.tensor(float(mag_completeness), dtype=torch.float32))
        self.register_buffer("M_m", torch.tensor(float(mag_max), dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(richter_b), dtype=torch.float32))

        base_rate_init_t = torch.as_tensor(
            base_rate_init,
            device=self.device,
            dtype=torch.get_default_dtype(),
        )
        tiny = torch.finfo(base_rate_init_t.dtype).tiny
        self.log_mu = nn.Parameter(base_rate_init_t.clamp_min(tiny).log())
        if self.fix_mu:
            self.log_mu.requires_grad = False
            mu_value = (
                torch.as_tensor(
                    fixed_mu_value,
                    device=self.device,
                    dtype=base_rate_init_t.dtype,
                )
                if fixed_mu_value is not None
                else torch.zeros(1, device=self.device, dtype=base_rate_init_t.dtype)
            )
            self.register_buffer("mu_fixed", mu_value)

        raw_weights = dict(loss_weights or {})
        self.bg_kl_weight = float(raw_weights.get("bg_kl_weight", 1.0))
        self.bg_norm_weight = float(raw_weights.get("bg_norm_weight", 0.0))
        self.branching_penalty_weight = float(
            raw_weights.get("branching_penalty_weight", branching_penalty_weight)
        )
        if self.bg_kl_weight < 0.0 or self.bg_norm_weight < 0.0:
            raise ValueError("bg_kl_weight and bg_norm_weight must be non-negative.")
        if self.branching_penalty_weight < 0.0:
            raise ValueError("branching_penalty_weight must be non-negative.")

        self.to(self.device)

    @property
    def mu(self) -> torch.Tensor:
        if self.fix_mu:
            return self.mu_fixed
        return torch.exp(self.log_mu)

    @property
    def num_basis(self) -> int:
        return self.basis.num_basis

    @property
    def _uses_history_truncation(self) -> bool:
        """Whether count-based or time-window history truncation is active."""
        return self.max_history_events > 0 or self.history_time_window is not None

    @staticmethod
    def _resolve_chunk_size(total: int, configured: int) -> int:
        """Resolve ``0`` as full length and positive values as capped chunks."""
        return resolve_chunk_size(total, configured)

    def _resolve_query_chunk_size(self, total: int) -> int:
        """Resolve query chunking, adding a safe default when truncation is active."""
        if self.query_chunk_size and self.query_chunk_size > 0:
            return self._resolve_chunk_size(total, self.query_chunk_size)
        if self._uses_history_truncation:
            return max(1, min(128, int(total)))
        return max(1, int(total))

    @staticmethod
    def _iter_chunks(total: int, chunk_size: int):
        """Yield ``[start, end)`` chunks covering ``total`` elements."""
        yield from iter_chunks(total, chunk_size)

    @staticmethod
    def _history_positions(event_mask_bool: torch.Tensor) -> torch.Tensor:
        """Map padded history columns to compact event positions per row."""
        return event_mask_bool.to(torch.long).cumsum(dim=-1) - 1

    @staticmethod
    def _previous_event_counts(
        *,
        t_query_chunk: torch.Tensor,
        t_history: torch.Tensor,
        event_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        """Count valid history events strictly before each query time."""
        counts = []
        for query_row, history_row, mask_row in zip(
            t_query_chunk,
            t_history,
            event_mask_bool,
        ):
            valid_history = history_row.masked_select(mask_row).contiguous()
            if valid_history.numel() == 0:
                counts.append(torch.zeros_like(query_row, dtype=torch.long))
            else:
                counts.append(
                    torch.searchsorted(
                        valid_history,
                        query_row.contiguous(),
                        right=False,
                    )
                )
        return torch.stack(counts, dim=0)

    def _history_cutoff_times(
        self,
        *,
        t_history: torch.Tensor,
        event_mask_bool: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Return per-parent end times implied by ``max_history_events``.

        For integral terms, a parent is active only until it would fall out of
        the recent-history window of future query times.  Entries that never
        expire are set to ``inf``.
        """
        if self.max_history_events <= 0:
            return None

        cutoff_rows = []
        for history_row, mask_row in zip(t_history, event_mask_bool):
            valid_history = history_row.masked_select(mask_row)
            cutoff_row = torch.full_like(history_row, float("inf"))
            if valid_history.numel() > self.max_history_events:
                cutoff_values = valid_history[self.max_history_events :]
                cutoff_row[: cutoff_values.numel()] = cutoff_values
            cutoff_rows.append(cutoff_row)
        return torch.stack(cutoff_rows, dim=0)

    def _select_history_for_query_chunk(
        self,
        *,
        t_query_chunk: torch.Tensor,
        t_history: torch.Tensor,
        event_mask_bool: torch.Tensor,
        history_positions: Optional[torch.Tensor] = None,
    ) -> _HistorySelection:
        """Select a coarse history slice that can contain contributing parents.

        The returned slice is a superset of the exact per-query valid parents.
        Fine-grained count and lag masks are still applied inside
        :meth:`_history_parent_mask`.
        """
        previous_event_counts = (
            self._previous_event_counts(
                t_query_chunk=t_query_chunk,
                t_history=t_history,
                event_mask_bool=event_mask_bool,
            )
            if self.max_history_events > 0
            else None
        )
        h_start = 0
        h_end = int(t_history.size(1))

        if self.max_history_events > 0:
            if previous_event_counts is None:
                raise ValueError(
                    "previous_event_counts is required when max_history_events > 0."
                )
            min_position = (previous_event_counts - self.max_history_events).clamp_min(0)
            h_start = max(h_start, int(min_position.min().item()))
            h_end = min(h_end, int(previous_event_counts.max().item()))

        if self.history_time_window is not None:
            if history_positions is None:
                raise ValueError(
                    "history_positions is required when history_time_window is set."
                )
            min_query = t_query_chunk.detach().min()
            max_query = t_query_chunk.detach().max()
            candidate_mask = (
                event_mask_bool
                & (t_history < max_query)
                & (t_history >= min_query - float(self.history_time_window))
            )
            if not bool(candidate_mask.any().item()):
                return _HistorySelection.empty_for(history_positions, previous_event_counts)
            candidate_positions = history_positions.masked_select(candidate_mask)
            h_start = max(h_start, int(candidate_positions.min().item()))
            h_end = min(h_end, int(candidate_positions.max().item()) + 1)

        if h_end <= h_start:
            return _HistorySelection.empty_for(history_positions, previous_event_counts)
        return _HistorySelection(h_start, h_end, history_positions, previous_event_counts)

    def _history_parent_mask(
        self,
        *,
        delta_t: torch.Tensor,
        event_mask_chunk: torch.Tensor,
        history_position_chunk: Optional[torch.Tensor],
        previous_event_counts: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build the exact parent mask for a query-history block."""
        parent_mask = (delta_t > 0.0) & event_mask_chunk.unsqueeze(-2)
        if self.history_time_window is not None:
            parent_mask = parent_mask & (delta_t <= self.history_time_window)
        if self.max_history_events > 0:
            if history_position_chunk is None or previous_event_counts is None:
                raise ValueError(
                    "history positions/counts are required when max_history_events > 0."
                )
            min_position = (previous_event_counts - self.max_history_events).clamp_min(0)
            recent_mask = (
                history_position_chunk.unsqueeze(1) >= min_position.unsqueeze(-1)
            )
            parent_mask = parent_mask & recent_mask
        return parent_mask

    def _integral_lag_bounds(
        self,
        *,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        t_history: torch.Tensor,
        event_mask_bool: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute lag bounds for closed-form trigger integrals.

        The upper bound is clipped by the optional count and time-window history
        truncation rules so the integrated intensity matches
        :meth:`trigger_intensity`.
        """
        effective_t_end = t_end.unsqueeze(-1)
        cutoff_times = self._history_cutoff_times(
            t_history=t_history,
            event_mask_bool=event_mask_bool,
        )
        if cutoff_times is not None:
            effective_t_end = torch.minimum(effective_t_end, cutoff_times.unsqueeze(-2))

        dt_start = (t_start.unsqueeze(-1) - t_history.unsqueeze(-2)).clamp_min(0.0)
        dt_end = (effective_t_end - t_history.unsqueeze(-2)).clamp_min(0.0)
        if self.history_time_window is not None:
            dt_end = torch.minimum(dt_end, dt_end.new_tensor(self.history_time_window))
        return dt_start, dt_end

    def _sampling_magnitude_params(self) -> _MagnitudeSamplingParams:
        return _MagnitudeSamplingParams(
            b=float(self.b.detach().cpu().item()),
            m_min=float(self.M_c.detach().cpu().item()),
            m_max=float(self.M_m.detach().cpu().item()),
        )

    def _event_mask(self, batch: Batch) -> torch.Tensor:
        return get_mask(
            batch.inter_times,
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx,
        )

    def _require_magnitude(self, batch: Batch) -> torch.Tensor:
        if not hasattr(batch, "mag") or batch.mag is None:
            raise ValueError("NETAS requires `batch.mag`.")
        return batch.mag.to(self.device)

    def _get_rnn_parent_context(self, batch: Batch, encoder: RNNTPPBackbone) -> torch.Tensor:
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling/context extraction currently supports no extra features.")
        features = encoder.build_features(batch)
        rnn_output, _ = encoder.rnn(features.contiguous())
        rnn_output = encoder._post_rnn_transform(rnn_output, features)
        return encoder.dropout(rnn_output * batch.input_mask[:, :, None])

    @staticmethod
    def _build_rnn_event_features(
        encoder: RNNTPPBackbone,
        *,
        inter_time: torch.Tensor,
        magnitude: torch.Tensor,
    ) -> torch.Tensor:
        feat_list = [encoder.encode_time(inter_time)]
        if encoder.input_magnitude:
            feat_list.append(encoder.encode_magnitude(magnitude))
        return torch.cat(feat_list, dim=-1).contiguous()

    def get_parent_context(self, batch: Batch) -> torch.Tensor:
        encoder = self.event_encoder
        if hasattr(encoder, "get_parent_context"):
            context = encoder.get_parent_context(batch)
        elif isinstance(encoder, RNNTPPBackbone):
            context = self._get_rnn_parent_context(batch, encoder)
        else:
            output = encoder(batch)
            context = output[0] if isinstance(output, tuple) else output

        if not torch.is_tensor(context) or context.ndim != 3:
            raise ValueError("Event encoder must return a tensor of shape [B, L, C].")
        if context.shape[0] != batch.batch_size or context.shape[1] != batch.seq_len:
            raise ValueError(
                "Parent context must be aligned with the padded event axis "
                f"(expected {(batch.batch_size, batch.seq_len)}, got {tuple(context.shape[:2])})."
            )
        if context.shape[-1] != self.context_size:
            raise ValueError(
                f"Parent context last dimension must equal context_size={self.context_size}."
            )
        return context.to(self.device)

    def _productivity_from_score(self, score: torch.Tensor) -> torch.Tensor:
        if self.productivity_mode == "bounded":
            return self.eta_max * torch.sigmoid(score)
        elif self.productivity_mode == "softplus":
            return F.softplus(score)
        elif self.productivity_mode == "exp":
            return torch.exp(score)
        raise RuntimeError(f"Unsupported productivity_mode={self.productivity_mode!r}.")

    def parent_params_from_context(
        self,
        parent_context: torch.Tensor,
        magnitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prod_score = self.productivity_head(parent_context).squeeze(-1)
        prod_score = prod_score + self.productivity_bias
        prod_score = prod_score + self.productivity_alpha * (magnitudes - self.M_c)
        eta = self._productivity_from_score(prod_score)
        omega = F.softmax(self.mixture_head(parent_context), dim=-1)
        return eta, omega

    def _make_parent_parameters(
        self,
        batch: Batch,
        *,
        parent_context: Optional[torch.Tensor] = None,
    ) -> _ParentParameters:
        magnitudes = self._require_magnitude(batch)
        event_mask = self._event_mask(batch).to(self.device)
        if parent_context is None:
            parent_context = self.get_parent_context(batch)
        eta, omega = self.parent_params_from_context(parent_context, magnitudes)
        eta = eta * event_mask
        omega = omega * event_mask.unsqueeze(-1)
        return _ParentParameters(
            context=parent_context,
            eta=eta,
            omega=omega,
            event_mask=event_mask,
        )

    @staticmethod
    def _coerce_parent_parameters(
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
    ) -> _ParentParameters:
        if isinstance(parent_params, _ParentParameters):
            return parent_params
        return _ParentParameters.from_mapping(parent_params)

    def parent_parameters(
        self,
        batch: Batch,
        *,
        parent_context: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        return self._make_parent_parameters(
            batch,
            parent_context=parent_context,
        ).as_dict()

    def _history_contrib(
        self,
        *,
        t_query_chunk: torch.Tensor,
        t_hist_chunk: torch.Tensor,
        eta_hist_chunk: torch.Tensor,
        omega_hist_chunk: torch.Tensor,
        event_mask_chunk: torch.Tensor,
        history_selection: _HistorySelection,
        history_start: int,
        history_end: int,
    ) -> torch.Tensor:
        """Accumulate trigger contribution from one history chunk."""
        delta_t = t_query_chunk.unsqueeze(-1) - t_hist_chunk.unsqueeze(-2)
        parent_mask = self._history_parent_mask(
            delta_t=delta_t,
            event_mask_chunk=event_mask_chunk,
            history_position_chunk=history_selection.position_chunk(
                history_start,
                history_end,
            ),
            previous_event_counts=history_selection.previous_counts,
        )
        basis_pdf = self.basis.pdf(delta_t)
        parent_weight = eta_hist_chunk.unsqueeze(-1) * omega_hist_chunk
        contrib = basis_pdf * parent_weight.unsqueeze(-3) * parent_mask.unsqueeze(-1)
        return contrib.sum(dim=(-2, -1))

    def _trigger_intensity_from_history(
        self,
        *,
        t_query: torch.Tensor,
        t_history: torch.Tensor,
        eta: torch.Tensor,
        omega: torch.Tensor,
        event_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate trigger intensity for query times from encoded parent history."""
        s_total = t_query.size(1)
        l_total = t_history.size(1)
        out = torch.zeros_like(t_query)
        query_chunk_size = self._resolve_query_chunk_size(s_total)
        history_chunk_size = self._resolve_chunk_size(l_total, self.history_chunk_size)
        history_positions = (
            self._history_positions(event_mask_bool)
            if self._uses_history_truncation
            else None
        )

        for q_start, q_end in self._iter_chunks(s_total, query_chunk_size):
            t_query_chunk = t_query[:, q_start:q_end]
            intensity_chunk = torch.zeros_like(t_query_chunk)
            history_selection = self._select_history_for_query_chunk(
                t_query_chunk=t_query_chunk,
                t_history=t_history,
                event_mask_bool=event_mask_bool,
                history_positions=history_positions,
            )
            if history_selection.empty:
                out[:, q_start:q_end] = intensity_chunk
                continue
            for h_start, h_end in history_selection.iter_chunks(history_chunk_size):
                intensity_chunk = intensity_chunk + self._history_contrib(
                    t_query_chunk=t_query_chunk,
                    t_hist_chunk=t_history[:, h_start:h_end],
                    eta_hist_chunk=eta[:, h_start:h_end],
                    omega_hist_chunk=omega[:, h_start:h_end, :],
                    event_mask_chunk=event_mask_bool[:, h_start:h_end],
                    history_selection=history_selection,
                    history_start=h_start,
                    history_end=h_end,
                )
            out[:, q_start:q_end] = intensity_chunk
        return out

    def trigger_intensity(
        self,
        batch: Batch,
        *,
        t_query: Optional[torch.Tensor] = None,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        """Return trigger-only intensity at ``t_query`` for a batch.

        Args:
            batch: Event batch providing history event times and marks.
            t_query: Query times with shape ``[batch, num_queries]``.  If omitted,
                event arrival times are used.
            parent_params: Optional cached parent parameters from
                :meth:`parent_parameters`.
        """
        if parent_params is None:
            parent_params = self._make_parent_parameters(batch)
        parent_params = self._coerce_parent_parameters(parent_params)
        t_history = batch.arrival_times.to(self.device)
        if t_query is None:
            t_query = t_history
        return self._trigger_intensity_from_history(
            t_query=t_query.to(self.device),
            t_history=t_history,
            eta=parent_params.eta,
            omega=parent_params.omega,
            event_mask_bool=parent_params.event_mask_bool,
        )

    def intensity(
        self,
        batch: Batch,
        *,
        t_query: Optional[torch.Tensor] = None,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        """Return total conditional intensity at ``t_query``."""
        trigger = self.trigger_intensity(batch, t_query=t_query, parent_params=parent_params)
        total = trigger + self.mu.to(trigger.dtype)
        if self.bg_model is not None:
            total = total + self.bg_model.intensity(batch, t_query=t_query)
        return total

    def trigger_integral_between(
        self,
        batch: Batch,
        *,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        """Integrate trigger intensity over ``[t_start, t_end]``.

        ``t_start`` and ``t_end`` may be shaped ``[batch]`` or
        ``[batch, num_intervals]``.  The result has the broadcast interval shape
        and excludes exogenous/background intensity.
        """
        if t_start.ndim == 1:
            t_start = t_start.unsqueeze(-1)
        if t_end.ndim == 1:
            t_end = t_end.unsqueeze(-1)
        if t_start.shape != t_end.shape:
            raise ValueError("t_start and t_end must have identical shapes.")
        t_start = t_start.to(self.device)
        t_end = t_end.to(self.device)

        if parent_params is None:
            parent_params = self._make_parent_parameters(batch)
        parent_params = self._coerce_parent_parameters(parent_params)

        t_history = batch.arrival_times.to(self.device)
        event_mask_bool = parent_params.event_mask_bool
        dt_start, dt_end = self._integral_lag_bounds(
            t_start=t_start,
            t_end=t_end,
            t_history=t_history,
            event_mask_bool=event_mask_bool,
        )
        mass = self.basis.interval_mass(dt_start, dt_end)
        parent_weight = parent_params.weighted_mixture
        masked_weight = (
            parent_weight.unsqueeze(1)
            * event_mask_bool.unsqueeze(1).unsqueeze(-1)
        )
        return (mass * masked_weight).sum(dim=(-2, -1))

    def compensator(
        self,
        batch: Batch,
        *,
        t_query: torch.Tensor,
        lower: Optional[torch.Tensor] = None,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        if t_query.ndim == 1:
            t_query = t_query.unsqueeze(-1)
        if lower is None:
            lower = batch.t_start
        if lower.ndim == 1:
            lower = lower.unsqueeze(-1).expand_as(t_query)
        trigger = self.trigger_integral_between(
            batch,
            t_start=lower,
            t_end=t_query,
            parent_params=parent_params,
        )
        total = trigger + self.mu.to(trigger.dtype) * (t_query - lower)
        if self.bg_model is not None:
            total = total + self.bg_model.intensity_integral_between(
                batch,
                t_start=lower,
                t_end=t_query,
            )
        return total

    def _nll_event_log_intensity(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = batch.arrival_times.to(self.device)
        t_select, intensity_mask = masked_select_per_row(
            t,
            batch.nll_event_mask.to(self.device),
        )
        intensity = self.intensity(
            batch,
            t_query=t_select,
            parent_params=parent_params,
        )
        log_intensity = torch.log(intensity.clamp_min(eps))
        return (log_intensity * intensity_mask).sum(dim=-1), t_select, intensity_mask

    def _nll_integral(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
    ) -> torch.Tensor:
        t_start = batch.t_nll_start.to(self.device).unsqueeze(-1)
        t_end = batch.t_end.to(self.device).unsqueeze(-1)
        integral = self.trigger_integral_between(
            batch,
            t_start=t_start,
            t_end=t_end,
            parent_params=parent_params,
        ).squeeze(-1)
        interval_length = (batch.t_end - batch.t_nll_start).to(
            device=integral.device,
            dtype=integral.dtype,
        )
        integral = integral + interval_length * self.mu.to(integral.dtype)
        if self.bg_model is not None:
            integral = integral + self.bg_model.intensity_integral(batch)
        return integral

    def _background_regularization_terms(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
        t_query: torch.Tensor,
        event_mask: torch.Tensor,
        eps: float,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        bg_kl = None
        if self.bg_model is not None and hasattr(self.bg_model, "kl_term"):
            bg_kl = self.bg_model.kl_term(batch, eps=eps)

        bg_norm = None
        if (
            self.bg_model is not None
            and self.bg_norm_weight > 0.0
            and hasattr(self.bg_model, "normalizing_term")
        ):
            trigger_intensity = self.trigger_intensity(
                batch,
                t_query=t_query,
                parent_params=parent_params,
            )
            bg_norm = self.bg_model.normalizing_term(
                batch,
                torch.log(trigger_intensity.clamp_min(eps)),
                eps=eps,
                t_query=t_query,
                event_mask=event_mask,
            )
        return bg_kl, bg_norm

    def _branching_penalty(
        self,
        batch: Batch,
        *,
        parent_params: _ParentParameters,
    ) -> Optional[torch.Tensor]:
        if self.branching_penalty_weight <= 0.0:
            return None

        event_mask = parent_params.event_mask.to(parent_params.eta.dtype)
        event_counts = event_mask.sum(dim=-1).clamp_min(1.0)
        mean_eta = (parent_params.eta * event_mask).sum(dim=-1) / event_counts
        target = torch.as_tensor(
            self.branching_penalty_target,
            device=mean_eta.device,
            dtype=mean_eta.dtype,
        )
        penalty = F.relu(mean_eta - target).pow(2)
        nll_event_counts = batch.nll_event_mask.to(
            device=penalty.device,
            dtype=penalty.dtype,
        ).sum(dim=-1).clamp_min(1.0)
        return penalty * nll_event_counts

    def nll_loss(
        self,
        batch: Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        reduction = self.reduction if reduction is None else reduction

        parent_params = self._make_parent_parameters(batch)
        log_intensity, t_select, intensity_mask = self._nll_event_log_intensity(
            batch,
            parent_params=parent_params,
            eps=eps,
        )
        integral = self._nll_integral(batch, parent_params=parent_params)
        nll_time = -log_intensity + integral
        bg_kl, bg_norm = self._background_regularization_terms(
            batch,
            parent_params=parent_params,
            t_query=t_select,
            event_mask=intensity_mask,
            eps=eps,
        )
        branching_penalty = self._branching_penalty(
            batch,
            parent_params=parent_params,
        )

        nll_total = nll_time
        if bg_kl is not None:
            nll_total = nll_total + self.bg_kl_weight * bg_kl
        if bg_norm is not None:
            nll_total = nll_total + self.bg_norm_weight * bg_norm
        if branching_penalty is not None:
            nll_total = nll_total + self.branching_penalty_weight * branching_penalty

        out_dict = {"time": nll_time, "total": nll_total}
        if bg_kl is not None:
            out_dict["bg_kl"] = bg_kl
        if bg_norm is not None:
            out_dict["bg_norm"] = bg_norm
        if branching_penalty is not None:
            out_dict["branching_penalty"] = branching_penalty

        out = self.reduce_nll_dict(out_dict, batch, reduction=reduction, eps=eps)
        if return_dict:
            return out
        return out["total"]

    def _evaluation_grid(
        self,
        sequence: Sequence,
        *,
        num_grid_points: int,
    ) -> tuple[Batch, torch.Tensor]:
        if int(num_grid_points) < 1:
            raise ValueError("num_grid_points must be >= 1.")
        batch = Batch.from_list([sequence]).to(self.device)
        seq_len = int(batch.end_idx[0].item()) + 1
        inter_times = batch.inter_times[0, :seq_len]
        interval_starts = torch.cat(
            [
                batch.t_start[:1].to(inter_times.dtype),
                batch.arrival_times[0, : seq_len - 1].to(inter_times.dtype),
            ],
            dim=0,
        )
        fractions = torch.linspace(
            1e-4,
            1.0,
            int(num_grid_points),
            device=self.device,
            dtype=inter_times.dtype,
        )
        grid = interval_starts[:, None] + inter_times[:, None] * fractions[None, :]
        return batch, grid.reshape(1, -1)

    def evaluate_intensity(
        self,
        sequence: src.data.Sequence,
        num_grid_points: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, grid = self._evaluation_grid(sequence, num_grid_points=num_grid_points)
        parent_params = self.parent_parameters(batch)
        intensity = self.intensity(batch, t_query=grid, parent_params=parent_params)
        return grid.squeeze(0), intensity.squeeze(0)

    def evaluate_compensator(
        self,
        sequence: src.data.Sequence,
        num_grid_points: int = 50,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, grid = self._evaluation_grid(sequence, num_grid_points=num_grid_points)
        parent_params = self.parent_parameters(batch)
        lower = batch.t_start.unsqueeze(-1).expand_as(grid)
        compensator = self.compensator(
            batch,
            t_query=grid,
            lower=lower,
            parent_params=parent_params,
        )
        return grid.squeeze(0), compensator.squeeze(0)


__all__ = [
    "FixedKernelBasis",
    "MambaNETASEncoder",
    "NETAS",
    "masked_select_per_row",
]
