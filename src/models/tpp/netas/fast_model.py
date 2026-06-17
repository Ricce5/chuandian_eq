"""Scalable NETAS variant for exponential triggering bases.

``FastNETAS`` keeps the neural productivity and mixture heads from ``NETAS`` but
changes likelihood/intensity evaluation to an exponential-prefix formulation.
For ``K`` exponential kernels and ``N`` events, trigger terms are evaluated in
``O(NK)`` memory/time instead of building pairwise ``O(N^2K)`` history tensors.
Lomax/Omori tails should be approximated by choosing enough log-spaced
exponential basis kernels in the config, not by switching this class to Lomax.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch

from src.data.batch import Batch
from src.utils.mask_utils import masked_select_per_row

from .model import NETAS
from .types import _ParentParameters


@dataclass(frozen=True)
class _ExponentialPrefix:
    """Prefix sums needed for exponential-state NETAS scans."""

    t_history: torch.Tensor
    time_origin: torch.Tensor
    event_mask_bool: torch.Tensor
    rates: torch.Tensor
    log_weighted_exp_prefix: torch.Tensor
    weight_prefix: torch.Tensor


class FastNETAS(NETAS):
    """NETAS with exact ``O((N + Q)K)`` exponential-basis scans.

    The original :class:`NETAS` supports arbitrary fixed basis families by
    materializing query-history blocks.  This subclass intentionally supports
    only exponential basis kernels, because their prefix-state identity avoids
    the pairwise history matrix and scales to catalogs with ``10^5`` events.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("basis_family", "exponential")
        super().__init__(*args, **kwargs)
        if self.basis.family != "exponential":
            raise ValueError(
                "FastNETAS requires netas_basis_family='exponential'. "
                "Use a log-spaced exponential mixture to approximate Lomax/Omori tails."
            )
        if self.max_history_events > 0 or self.history_time_window is not None:
            raise ValueError(
                "FastNETAS does not support history truncation controls; leave "
                "netas_max_history_events=0 and netas_history_time_window=null."
            )

    @staticmethod
    def _as_time_matrix(
        value: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
    ) -> torch.Tensor:
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 0:
            return value.reshape(1, 1).expand(batch_size, 1)
        if value.ndim == 1:
            if value.numel() == batch_size:
                return value.unsqueeze(-1)
            if batch_size == 1:
                return value.unsqueeze(0)
            raise ValueError(
                f"{name} with one dimension must have length batch_size={batch_size}."
            )
        if value.ndim != 2:
            raise ValueError(f"{name} must be a 1D or 2D tensor.")
        if value.shape[0] != batch_size:
            raise ValueError(
                f"{name} first dimension must equal batch_size={batch_size}."
            )
        return value

    @staticmethod
    def _previous_counts(
        *,
        t_history: torch.Tensor,
        event_mask_bool: torch.Tensor,
        t_query: torch.Tensor,
    ) -> torch.Tensor:
        """Count valid history events with ``t_history < t_query`` per row."""
        counts = [
            torch.searchsorted(
                history_row.masked_select(mask_row).contiguous(),
                query_row.contiguous(),
                right=False,
            )
            for history_row, mask_row, query_row in zip(
                t_history,
                event_mask_bool,
                t_query,
            )
        ]
        return torch.stack(counts, dim=0).clamp_(0, t_history.size(1)).long()

    @staticmethod
    def _gather_prefix(
        prefix: torch.Tensor,
        counts: torch.Tensor,
        *,
        fill_value: float,
    ) -> torch.Tensor:
        """Gather inclusive prefix values at exclusive event counts."""
        if prefix.ndim != 3:
            raise ValueError("prefix must have shape [batch, seq_len, num_basis].")
        index = (counts.clamp_min(1) - 1).unsqueeze(-1).expand(
            -1,
            -1,
            prefix.size(-1),
        )
        gathered = prefix.gather(dim=1, index=index)
        fill = torch.as_tensor(
            fill_value,
            device=prefix.device,
            dtype=prefix.dtype,
        )
        return torch.where(counts.unsqueeze(-1) > 0, gathered, fill)

    def _basis_rates(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self.basis.effective_rates.to(device=device, dtype=dtype)

    @staticmethod
    def _safe_exp(log_value: torch.Tensor) -> torch.Tensor:
        max_log = torch.as_tensor(
            math.log(torch.finfo(log_value.dtype).max),
            device=log_value.device,
            dtype=log_value.dtype,
        )
        safe_log_value = torch.nan_to_num(log_value, nan=-float("inf"))
        return torch.exp(safe_log_value.clamp_max(max_log))

    @staticmethod
    def _finite_nonnegative(value: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(
            value,
            nan=0.0,
            posinf=torch.finfo(value.dtype).max,
            neginf=0.0,
        ).clamp_min(0.0)

    def _build_exponential_prefix(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
    ) -> _ExponentialPrefix:
        parent_params = self._coerce_parent_parameters(parent_params)
        basis_weights = parent_params.weighted_mixture.to(self.device)
        dtype = basis_weights.dtype
        t_abs = batch.arrival_times.to(device=self.device, dtype=dtype)
        time_origin = batch.t_start.to(device=self.device, dtype=dtype).unsqueeze(-1)
        t_history = t_abs - time_origin
        event_mask_bool = parent_params.event_mask_bool.to(self.device)
        rates = self._basis_rates(device=self.device, dtype=dtype)
        valid_weights = basis_weights * event_mask_bool.unsqueeze(-1).to(dtype)

        valid_weight_mask = valid_weights > 0.0
        safe_weights = valid_weights.masked_fill(~valid_weight_mask, 1.0)
        log_terms = (
            safe_weights.log()
            + t_history.unsqueeze(-1) * rates.view(1, 1, -1)
        ).masked_fill(~valid_weight_mask, -float("inf"))
        return _ExponentialPrefix(
            t_history=t_history,
            time_origin=time_origin,
            event_mask_bool=event_mask_bool,
            rates=rates,
            log_weighted_exp_prefix=torch.logcumsumexp(log_terms, dim=1),
            weight_prefix=valid_weights.cumsum(dim=1),
        )

    def _log_weighted_exp_at_queries(
        self,
        prefix: _ExponentialPrefix,
        *,
        t_query: torch.Tensor,
    ) -> torch.Tensor:
        t_query = t_query - prefix.time_origin
        counts = self._previous_counts(
            t_history=prefix.t_history,
            event_mask_bool=prefix.event_mask_bool,
            t_query=t_query,
        )
        return self._gather_prefix(
            prefix.log_weighted_exp_prefix,
            counts,
            fill_value=-float("inf"),
        )

    def _trigger_intensity_from_prefix(
        self,
        prefix: _ExponentialPrefix,
        *,
        t_query: torch.Tensor,
    ) -> torch.Tensor:
        t_query_local = t_query - prefix.time_origin
        log_weighted_exp = self._log_weighted_exp_at_queries(prefix, t_query=t_query)
        log_state = (
            log_weighted_exp
            - t_query_local.unsqueeze(-1) * prefix.rates.view(1, 1, -1)
        )
        state = self._safe_exp(log_state)
        trigger = (state * prefix.rates.view(1, 1, -1)).sum(dim=-1)
        return self._finite_nonnegative(trigger)

    def _cumulative_trigger_from_prefix(
        self,
        prefix: _ExponentialPrefix,
        *,
        t_query: torch.Tensor,
    ) -> torch.Tensor:
        t_query_local = t_query - prefix.time_origin
        counts = self._previous_counts(
            t_history=prefix.t_history,
            event_mask_bool=prefix.event_mask_bool,
            t_query=t_query_local,
        )
        log_weighted_exp = self._gather_prefix(
            prefix.log_weighted_exp_prefix,
            counts,
            fill_value=-float("inf"),
        )
        weights = self._gather_prefix(
            prefix.weight_prefix,
            counts,
            fill_value=0.0,
        )
        decayed_state = self._safe_exp(
            log_weighted_exp
            - t_query_local.unsqueeze(-1) * prefix.rates.view(1, 1, -1)
        )
        cumulative_by_basis = (weights - decayed_state).clamp_min(0.0)
        return cumulative_by_basis.sum(dim=-1)

    def _trigger_integral_from_prefix(
        self,
        prefix: _ExponentialPrefix,
        *,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
    ) -> torch.Tensor:
        t_start, t_end = torch.broadcast_tensors(t_start, t_end)
        cumulative_end = self._cumulative_trigger_from_prefix(prefix, t_query=t_end)
        cumulative_start = self._cumulative_trigger_from_prefix(prefix, t_query=t_start)
        integral = cumulative_end - cumulative_start
        same_or_backward = t_end <= t_start
        integral = torch.where(same_or_backward, torch.zeros_like(integral), integral)
        return self._finite_nonnegative(integral)

    def trigger_intensity(
        self,
        batch: Batch,
        *,
        t_query: Optional[torch.Tensor] = None,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        if parent_params is None:
            parent_params = self._make_parent_parameters(batch)
        parent_params = self._coerce_parent_parameters(parent_params)
        prefix = self._build_exponential_prefix(batch, parent_params=parent_params)
        if t_query is None:
            t_query = prefix.t_history + prefix.time_origin
        else:
            t_query = self._as_time_matrix(
                t_query,
                batch_size=batch.batch_size,
                device=self.device,
                dtype=prefix.t_history.dtype,
                name="t_query",
            )
        return self._trigger_intensity_from_prefix(prefix, t_query=t_query)

    def trigger_integral_between(
        self,
        batch: Batch,
        *,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        parent_params: Optional[dict[str, torch.Tensor] | _ParentParameters] = None,
    ) -> torch.Tensor:
        if parent_params is None:
            parent_params = self._make_parent_parameters(batch)
        parent_params = self._coerce_parent_parameters(parent_params)
        prefix = self._build_exponential_prefix(batch, parent_params=parent_params)
        t_start = self._as_time_matrix(
            t_start,
            batch_size=batch.batch_size,
            device=self.device,
            dtype=prefix.t_history.dtype,
            name="t_start",
        )
        t_end = self._as_time_matrix(
            t_end,
            batch_size=batch.batch_size,
            device=self.device,
            dtype=prefix.t_history.dtype,
            name="t_end",
        )
        return self._trigger_integral_from_prefix(
            prefix,
            t_start=t_start,
            t_end=t_end,
        )

    def _nll_event_log_intensity_from_prefix(
        self,
        batch: Batch,
        *,
        prefix: _ExponentialPrefix,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        t_select, intensity_mask = masked_select_per_row(
            batch.arrival_times.to(self.device),
            batch.nll_event_mask.to(self.device),
        )
        t_select = t_select.to(device=self.device, dtype=prefix.t_history.dtype)
        trigger_intensity = self._trigger_intensity_from_prefix(
            prefix,
            t_query=t_select,
        )
        intensity = trigger_intensity + self.mu.to(trigger_intensity.dtype)
        if self.bg_model is not None:
            intensity = intensity + self.bg_model.intensity(batch, t_query=t_select)
        log_intensity = torch.log(intensity.clamp_min(eps))
        return (
            (log_intensity * intensity_mask.to(log_intensity.dtype)).sum(dim=-1),
            t_select,
            intensity_mask,
            trigger_intensity,
        )

    def _nll_integral_from_prefix(
        self,
        batch: Batch,
        *,
        prefix: _ExponentialPrefix,
    ) -> torch.Tensor:
        t_start = batch.t_nll_start.to(device=self.device, dtype=prefix.t_history.dtype)
        t_end = batch.t_end.to(device=self.device, dtype=prefix.t_history.dtype)
        trigger_integral = self._trigger_integral_from_prefix(
            prefix,
            t_start=t_start.unsqueeze(-1),
            t_end=t_end.unsqueeze(-1),
        ).squeeze(-1)
        interval_length = (t_end - t_start).to(trigger_integral.dtype)
        integral = trigger_integral + interval_length * self.mu.to(trigger_integral.dtype)
        if self.bg_model is not None:
            integral = integral + self.bg_model.intensity_integral(batch)
        return integral

    def _background_regularization_terms_from_prefix(
        self,
        batch: Batch,
        *,
        t_query: torch.Tensor,
        event_mask: torch.Tensor,
        trigger_intensity: torch.Tensor,
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
            bg_norm = self.bg_model.normalizing_term(
                batch,
                torch.log(trigger_intensity.clamp_min(eps)),
                eps=eps,
                t_query=t_query,
                event_mask=event_mask,
            )
        return bg_kl, bg_norm

    def _nll_event_log_intensity(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prefix = self._build_exponential_prefix(batch, parent_params=parent_params)
        log_intensity, t_select, intensity_mask, _ = (
            self._nll_event_log_intensity_from_prefix(
                batch,
                prefix=prefix,
                eps=eps,
            )
        )
        return log_intensity, t_select, intensity_mask

    def _nll_integral(
        self,
        batch: Batch,
        *,
        parent_params: dict[str, torch.Tensor] | _ParentParameters,
    ) -> torch.Tensor:
        prefix = self._build_exponential_prefix(batch, parent_params=parent_params)
        return self._nll_integral_from_prefix(batch, prefix=prefix)

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
        prefix = self._build_exponential_prefix(batch, parent_params=parent_params)
        log_intensity, t_select, intensity_mask, trigger_intensity = (
            self._nll_event_log_intensity_from_prefix(
                batch,
                prefix=prefix,
                eps=eps,
            )
        )
        integral = self._nll_integral_from_prefix(batch, prefix=prefix)
        nll_time = -log_intensity + integral
        bg_kl, bg_norm = self._background_regularization_terms_from_prefix(
            batch,
            t_query=t_select,
            event_mask=intensity_mask,
            trigger_intensity=trigger_intensity,
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


__all__ = ["FastNETAS"]
