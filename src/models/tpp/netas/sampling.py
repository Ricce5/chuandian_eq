"""Sampling and simulation routines for NETAS."""

from __future__ import annotations

import copy
import heapq
import itertools
import logging
import math
from typing import Any, Optional, Union

import numpy as np
import torch

import src
from src.data.batch import Batch
from src.data.sequence import Sequence

from ..common.recurrent_blocks import RNNTPPBackbone
from .encoder import MambaNETASEncoder
from .types import (
    _MagnitudeSamplingParams,
    _SequenceEventAccumulator,
    _empty_initial_event_arrays,
    _gen_mag,
    _to_numpy_array,
)

logger = logging.getLogger(__name__)


class NETASSamplingMixin:
    """Branching-process sampling helpers for :class:`NETAS`."""

    def _encoder_initial_state(self, *, batch_size: int, dtype: torch.dtype):
        encoder = self.event_encoder
        if hasattr(encoder, "initial_state"):
            return encoder.initial_state(
                batch_size=batch_size,
                device=self.device,
                dtype=dtype,
            )
        if isinstance(encoder, RNNTPPBackbone):
            if encoder.num_extra_features is not None:
                raise ValueError("NETAS sampling does not support extra encoder features.")
            return torch.zeros(
                encoder.num_rnn_layers,
                batch_size,
                encoder.context_size,
                device=self.device,
                dtype=dtype,
            )
        raise NotImplementedError(
            "Sampling requires either an RNNTPPBackbone encoder or a custom encoder "
            "implementing `initial_state(...)` and `step_event(...)`."
        )

    def _encoder_step_event(
        self,
        *,
        inter_time: float,
        magnitude: float,
        prev_state: Any,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, Any]:
        encoder = self.event_encoder
        inter_time_t = torch.tensor([[float(inter_time)]], device=self.device, dtype=dtype)
        magnitude_t = torch.tensor([[float(magnitude)]], device=self.device, dtype=dtype)

        if hasattr(encoder, "step_event"):
            context, next_state = encoder.step_event(
                inter_time=inter_time_t,
                magnitude=magnitude_t,
                prev_state=prev_state,
            )
            return context, next_state

        if isinstance(encoder, RNNTPPBackbone):
            rnn_input = self._build_rnn_event_features(
                encoder,
                inter_time=inter_time_t,
                magnitude=magnitude_t,
            )
            return encoder.step(rnn_input, prev_state)

        raise NotImplementedError(
            "Sampling requires either an RNNTPPBackbone encoder or a custom encoder "
            "implementing `step_event(...)`."
        )

    def _encoder_step_event_batch_rnn(
        self,
        *,
        inter_time: torch.Tensor,
        magnitude: torch.Tensor,
        prev_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.event_encoder
        if not isinstance(encoder, RNNTPPBackbone):
            raise TypeError("_encoder_step_event_batch_rnn requires RNNTPPBackbone.")
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling does not support extra encoder features.")

        rnn_input = self._build_rnn_event_features(
            encoder,
            inter_time=inter_time,
            magnitude=magnitude,
        )
        return encoder.step(rnn_input, prev_state)

    def _sample_background_times_np(
        self,
        *,
        t_start: float,
        t_end: float,
        rng: np.random.Generator,
        preset_bg_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        duration = float(t_end - t_start)
        all_times: list[np.ndarray] = []

        mu_value = float(self.mu.detach().cpu().item())
        if mu_value > 0.0:
            n_back = int(rng.poisson(mu_value * duration))
            if n_back > 0:
                times = t_start + np.sort(rng.random(n_back) * duration)
                all_times.append(times.astype(np.float64, copy=False))

        if preset_bg_times is not None:
            bg_times = np.asarray(preset_bg_times, dtype=np.float64)
            if bg_times.size > 0:
                all_times.append(np.sort(bg_times))
        elif self.bg_model is not None:
            times_list = self.bg_model.sample_nhpp_inverse(
                B=1,
                t0=torch.tensor([t_start], device=self.device),
                dt=torch.tensor([duration], device=self.device),
                sample_sequence=True,
                mu=0.0,
            )
            if len(times_list) != 1:
                raise ValueError("Expected a single sampled background sequence.")
            bg_times = times_list[0]
            if torch.is_tensor(bg_times):
                bg_times = bg_times.detach().cpu().numpy()
            bg_times = np.asarray(bg_times, dtype=np.float64)
            if bg_times.size > 0:
                all_times.append(np.sort(bg_times))

        if not all_times:
            return np.empty((0,), dtype=np.float64)
        return np.sort(np.concatenate(all_times, axis=0))

    def _bg_cache_covers_window(
        self,
        cached: Any,
        *,
        t_start: float,
        t_end: float,
    ) -> bool:
        if cached is None or not hasattr(cached, "time_series_times"):
            return False
        ts_times = torch.as_tensor(cached.time_series_times).reshape(-1)
        if ts_times.numel() == 0:
            return False
        ts_min = float(ts_times[0].item())
        ts_max = float(ts_times[-1].item())
        return ts_min <= float(t_start) and ts_max >= float(t_end)

    def _ensure_bg_sampling_cache(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
        bg_cache_seq: Optional[Sequence],
    ) -> None:
        if self.bg_model is None:
            return

        cached = getattr(self.bg_model, "ts_batch_cache", None)
        if cached is None and not hasattr(self.bg_model, "cache_batch"):
            return
        if self._bg_cache_covers_window(
            cached,
            t_start=float(t_start),
            t_end=float(t_end),
        ):
            return

        for source_seq in (bg_cache_seq, past_seq):
            if source_seq is None:
                continue
            time_series = getattr(source_seq, "time_series", None)
            time_series_times = getattr(source_seq, "time_series_times", None)
            if time_series is None or time_series_times is None:
                continue
            ts_times = torch.as_tensor(time_series_times).reshape(-1)
            if ts_times.numel() == 0:
                continue
            ts_min = float(ts_times[0].item())
            ts_max = float(ts_times[-1].item())
            if ts_min <= float(t_start) and ts_max >= float(t_end):
                self.bg_model.cache_batch(
                    time_series=torch.as_tensor(time_series).unsqueeze(0),
                    time_series_times=ts_times.unsqueeze(0),
                )
                return

        raise ValueError(
            "NETAS background sampling requires a cached time series covering "
            f"[{float(t_start):.4f}, {float(t_end):.4f}]. "
            "Pass `bg_cache_seq=full_seq` to `model.sample(...)` or pre-call "
            "`model.bg_model.cache_batch(...)` with the full forcing series."
        )

    def _prepare_batched_bg_times(
        self,
        *,
        batch_size: int,
        t_start: float,
        t_end: float,
    ) -> list[np.ndarray]:
        if self.bg_model is None:
            return [np.empty((0,), dtype=np.float64) for _ in range(int(batch_size))]

        duration = float(t_end - t_start)
        times_list = self.bg_model.sample_nhpp_inverse(
            B=int(batch_size),
            t0=torch.full((int(batch_size),), float(t_start), device=self.device),
            dt=torch.full((int(batch_size),), duration, device=self.device),
            sample_sequence=True,
            mu=0.0,
        )
        bg_times_batch: list[np.ndarray] = []
        for bg_times in times_list:
            if torch.is_tensor(bg_times):
                bg_times = bg_times.detach().cpu().numpy()
            bg_times = np.asarray(bg_times, dtype=np.float64)
            bg_times_batch.append(np.sort(bg_times) if bg_times.size > 0 else np.empty((0,), dtype=np.float64))
        if len(bg_times_batch) != int(batch_size):
            raise ValueError(
                f"Expected {int(batch_size)} background sequences, got {len(bg_times_batch)}."
            )
        return bg_times_batch

    def _sample_offspring_for_parent(
        self,
        *,
        parent_time: float,
        parent_mag: float,
        parent_context: torch.Tensor,
        lower: float,
        upper: float,
        rng: np.random.Generator,
        mag_params: Optional[_MagnitudeSamplingParams] = None,
    ) -> list[tuple[float, float]]:
        if upper <= lower:
            return []

        magnitude_t = torch.tensor(
            [[float(parent_mag)]],
            device=parent_context.device,
            dtype=parent_context.dtype,
        )
        eta_t, omega_t = self.parent_params_from_context(parent_context, magnitude_t)
        eta = float(eta_t.squeeze().detach().cpu().item())
        omega = _to_numpy_array(omega_t.squeeze(0).squeeze(0)).astype(np.float64, copy=False)
        interval_mass = self.basis.interval_mass_np(lower, upper)
        offspring_means = eta * omega * interval_mass
        offspring_counts = rng.poisson(np.clip(offspring_means, a_min=0.0, a_max=None))

        children: list[tuple[float, float]] = []
        if mag_params is None:
            mag_params = self._sampling_magnitude_params()
        for basis_idx, count in enumerate(offspring_counts.tolist()):
            count = int(count)
            if count <= 0:
                continue
            lags = self.basis.sample_truncated_np(
                component_idx=basis_idx,
                lower=lower,
                upper=upper,
                size=count,
                rng=rng,
            )
            mags = _gen_mag(
                rng,
                count,
                b=mag_params.b,
                m_min=mag_params.m_min,
                m_max=mag_params.m_max,
            )
            for lag, magnitude in zip(lags.tolist(), mags.tolist()):
                children.append((parent_time + float(lag), float(magnitude)))
        return children

    def _sample_offspring_from_cached_means(
        self,
        *,
        parent_time: float,
        lower: float,
        upper: float,
        offspring_means: np.ndarray,
        rng: np.random.Generator,
        mag_params: Optional[_MagnitudeSamplingParams] = None,
    ) -> list[tuple[float, float]]:
        if upper <= lower:
            return []

        offspring_counts = rng.poisson(np.clip(offspring_means, a_min=0.0, a_max=None))
        children: list[tuple[float, float]] = []
        if mag_params is None:
            mag_params = self._sampling_magnitude_params()
        for basis_idx, count in enumerate(offspring_counts.tolist()):
            count = int(count)
            if count <= 0:
                continue
            lags = self.basis.sample_truncated_np(
                component_idx=basis_idx,
                lower=lower,
                upper=upper,
                size=count,
                rng=rng,
            )
            mags = _gen_mag(
                rng,
                count,
                b=mag_params.b,
                m_min=mag_params.m_min,
                m_max=mag_params.m_max,
            )
            for lag, magnitude in zip(lags.tolist(), mags.tolist()):
                children.append((parent_time + float(lag), float(magnitude)))
        return children

    def _clone_sampling_state(self, state: Any) -> Any:
        try:
            return copy.deepcopy(state)
        except Exception:
            if torch.is_tensor(state):
                return state.clone()
            if isinstance(state, tuple):
                return tuple(self._clone_sampling_state(item) for item in state)
            if isinstance(state, list):
                return [self._clone_sampling_state(item) for item in state]
            if isinstance(state, dict):
                return {
                    key: self._clone_sampling_state(value)
                    for key, value in state.items()
                }
            raise

    def _prepare_rnn_history_sampling_cache(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
        encoder: RNNTPPBackbone,
    ) -> dict[str, Any]:
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling does not support extra encoder features.")

        dtype = self.M_c.dtype
        state = self._encoder_initial_state(batch_size=1, dtype=dtype)
        cached_parents: list[dict[str, Any]] = []

        if past_seq is None:
            return {
                "state": self._clone_sampling_state(state),
                "last_time": float(t_start),
                "parents": cached_parents,
            }

        if not hasattr(past_seq, "mag") or past_seq.mag is None:
            raise ValueError("Sampling with past_seq requires `past_seq.mag`.")

        past_times = _to_numpy_array(past_seq.arrival_times).astype(np.float64, copy=False)
        past_mags = _to_numpy_array(past_seq.mag).astype(np.float64, copy=False)
        if past_times.size == 0:
            return {
                "state": self._clone_sampling_state(state),
                "last_time": float(past_seq.t_end),
                "parents": cached_parents,
                "parent_times": np.empty((0,), dtype=np.float64),
                "parent_lower": np.empty((0,), dtype=np.float64),
                "parent_upper": np.empty((0,), dtype=np.float64),
                "parent_offspring_means": np.empty((0, self.num_basis), dtype=np.float64),
            }

        inter_times = np.diff(past_times, prepend=[float(past_seq.t_start)])
        inter_time_t = torch.as_tensor(
            inter_times,
            device=self.device,
            dtype=dtype,
        ).unsqueeze(0)
        magnitude_t = torch.as_tensor(
            past_mags,
            device=self.device,
            dtype=dtype,
        ).unsqueeze(0)

        features = self._build_rnn_event_features(
            encoder,
            inter_time=inter_time_t,
            magnitude=magnitude_t,
        )
        rnn_output, state = encoder.rnn(features, state.contiguous())
        context = encoder._post_rnn_transform(rnn_output, features)
        context = encoder.dropout(context)

        parent_lower = np.maximum(float(t_start) - past_times, 0.0).astype(
            np.float64,
            copy=False,
        )
        parent_upper = np.maximum(float(t_end) - past_times, 0.0).astype(
            np.float64,
            copy=False,
        )
        lower_t = torch.as_tensor(parent_lower, device=self.device, dtype=dtype)
        upper_t = torch.as_tensor(parent_upper, device=self.device, dtype=dtype)
        eta_t, omega_t = self.parent_params_from_context(context, magnitude_t)
        interval_mass = self.basis.interval_mass(lower_t, upper_t)
        valid = (upper_t > lower_t).to(dtype=interval_mass.dtype).unsqueeze(-1)
        parent_offspring_means_t = (
            eta_t.squeeze(0).unsqueeze(-1)
            * omega_t.squeeze(0)
            * interval_mass
            * valid
        )
        parent_offspring_means = _to_numpy_array(parent_offspring_means_t).astype(
            np.float64,
            copy=False,
        )

        for event_time, lower, upper, offspring_means in zip(
            past_times.tolist(),
            parent_lower.tolist(),
            parent_upper.tolist(),
            parent_offspring_means,
        ):
            cached_parents.append(
                {
                    "time": float(event_time),
                    "lower": float(lower),
                    "upper": float(upper),
                    "offspring_means": offspring_means,
                }
            )

        return {
            "state": self._clone_sampling_state(state),
            "last_time": float(past_times[-1]),
            "parents": cached_parents,
            "parent_times": past_times,
            "parent_lower": parent_lower,
            "parent_upper": parent_upper,
            "parent_offspring_means": parent_offspring_means,
        }

    def _prepare_history_sampling_cache(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
    ) -> dict[str, Any]:
        encoder = self.event_encoder
        if isinstance(encoder, RNNTPPBackbone):
            return self._prepare_rnn_history_sampling_cache(
                t_start=t_start,
                t_end=t_end,
                past_seq=past_seq,
                encoder=encoder,
            )

        dtype = self.M_c.dtype
        state = self._encoder_initial_state(batch_size=1, dtype=dtype)
        last_time = float(t_start)
        cached_parents: list[dict[str, Any]] = []

        if past_seq is None:
            return {
                "state": self._clone_sampling_state(state),
                "last_time": last_time,
                "parents": cached_parents,
            }

        if not hasattr(past_seq, "mag") or past_seq.mag is None:
            raise ValueError("Sampling with past_seq requires `past_seq.mag`.")

        last_time = float(past_seq.t_start)
        past_times = _to_numpy_array(past_seq.arrival_times).astype(np.float64, copy=False)
        past_mags = _to_numpy_array(past_seq.mag).astype(np.float64, copy=False)
        has_past_events = past_times.size > 0
        for event_time, event_mag in zip(past_times.tolist(), past_mags.tolist()):
            context, state = self._encoder_step_event(
                inter_time=float(event_time - last_time),
                magnitude=float(event_mag),
                prev_state=state,
                dtype=dtype,
            )
            last_time = float(event_time)
            lower = max(t_start - float(event_time), 0.0)
            upper = max(t_end - float(event_time), 0.0)
            if upper > lower:
                magnitude_t = torch.tensor(
                    [[float(event_mag)]],
                    device=context.device,
                    dtype=context.dtype,
                )
                eta_t, omega_t = self.parent_params_from_context(context, magnitude_t)
                eta = float(eta_t.squeeze().detach().cpu().item())
                omega = _to_numpy_array(omega_t.squeeze(0).squeeze(0)).astype(
                    np.float64,
                    copy=False,
                )
                interval_mass = self.basis.interval_mass_np(lower, upper)
                offspring_means = eta * omega * interval_mass
            else:
                offspring_means = np.zeros((self.num_basis,), dtype=np.float64)
            cached_parents.append(
                {
                    "time": float(event_time),
                    "lower": float(lower),
                    "upper": float(upper),
                    "offspring_means": offspring_means,
                }
            )
        if not has_past_events:
            last_time = float(past_seq.t_end)

        if cached_parents:
            parent_times = np.asarray(
                [payload["time"] for payload in cached_parents],
                dtype=np.float64,
            )
            parent_lower = np.asarray(
                [payload["lower"] for payload in cached_parents],
                dtype=np.float64,
            )
            parent_upper = np.asarray(
                [payload["upper"] for payload in cached_parents],
                dtype=np.float64,
            )
            parent_offspring_means = np.stack(
                [payload["offspring_means"] for payload in cached_parents],
                axis=0,
            ).astype(np.float64, copy=False)
        else:
            parent_times = np.empty((0,), dtype=np.float64)
            parent_lower = np.empty((0,), dtype=np.float64)
            parent_upper = np.empty((0,), dtype=np.float64)
            parent_offspring_means = np.empty(
                (0, self.num_basis),
                dtype=np.float64,
            )

        return {
            "state": self._clone_sampling_state(state),
            "last_time": last_time,
            "parents": cached_parents,
            "parent_times": parent_times,
            "parent_lower": parent_lower,
            "parent_upper": parent_upper,
            "parent_offspring_means": parent_offspring_means,
        }

    def _resolve_sampling_parent_chunk_size(
        self,
        *,
        batch_size: int,
        num_parents: int,
    ) -> int:
        if int(num_parents) <= 0:
            return 0
        target_count_elements = 1_000_000
        denom = max(1, int(batch_size) * int(self.num_basis))
        chunk_size = max(32, target_count_elements // denom)
        return min(int(num_parents), max(1, int(chunk_size)))

    def _build_batched_initial_events(
        self,
        *,
        batch_size: int,
        t_start: float,
        t_end: float,
        history_cache: dict[str, Any],
        init_rng: np.random.Generator,
        batched_bg_times: Optional[list[np.ndarray]] = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        initial_time_chunks: list[list[np.ndarray]] = [[] for _ in range(int(batch_size))]
        initial_mag_chunks: list[list[np.ndarray]] = [[] for _ in range(int(batch_size))]

        mag_params = self._sampling_magnitude_params()

        parent_times = np.asarray(history_cache.get("parent_times", []), dtype=np.float64)
        parent_lower = np.asarray(history_cache.get("parent_lower", []), dtype=np.float64)
        parent_upper = np.asarray(history_cache.get("parent_upper", []), dtype=np.float64)
        parent_offspring_means = np.asarray(
            history_cache.get("parent_offspring_means", []),
            dtype=np.float64,
        )

        num_parents = int(parent_times.shape[0])
        if num_parents > 0:
            parent_chunk_size = self._resolve_sampling_parent_chunk_size(
                batch_size=int(batch_size),
                num_parents=num_parents,
            )
            for parent_start in range(0, num_parents, parent_chunk_size):
                parent_end = min(num_parents, parent_start + parent_chunk_size)
                chunk_times = parent_times[parent_start:parent_end]
                chunk_lower = parent_lower[parent_start:parent_end]
                chunk_upper = parent_upper[parent_start:parent_end]
                chunk_means = parent_offspring_means[parent_start:parent_end]
                if (
                    chunk_means.size == 0
                    or not np.any(chunk_upper > chunk_lower)
                    or not np.any(chunk_means > 0.0)
                ):
                    continue

                counts_cube = init_rng.poisson(
                    chunk_means[None, :, :],
                    size=(int(batch_size), chunk_means.shape[0], chunk_means.shape[1]),
                )
                if counts_cube.size == 0 or int(counts_cube.sum()) == 0:
                    continue

                for basis_idx in range(chunk_means.shape[1]):
                    counts_bp = counts_cube[:, :, basis_idx]
                    flat_counts = counts_bp.reshape(-1)
                    nonzero_mask = flat_counts > 0
                    if not np.any(nonzero_mask):
                        continue

                    contributing_pairs = np.nonzero(nonzero_mask)[0]
                    pair_counts = flat_counts[nonzero_mask].astype(np.int64, copy=False)
                    total_count = int(pair_counts.sum())
                    if total_count <= 0:
                        continue

                    chunk_parent_count = chunk_means.shape[0]
                    seq_ids = contributing_pairs // chunk_parent_count
                    parent_ids = contributing_pairs % chunk_parent_count
                    expanded_seq_ids = np.repeat(seq_ids, pair_counts)
                    expanded_parent_ids = np.repeat(parent_ids, pair_counts)

                    lags = self.basis.sample_truncated_vectorized_np(
                        component_idx=basis_idx,
                        lower=chunk_lower[expanded_parent_ids],
                        upper=chunk_upper[expanded_parent_ids],
                        rng=init_rng,
                    )
                    event_times = chunk_times[expanded_parent_ids] + lags
                    event_mags = _gen_mag(
                        init_rng,
                        total_count,
                        b=mag_params.b,
                        m_min=mag_params.m_min,
                        m_max=mag_params.m_max,
                    ).astype(np.float32, copy=False)

                    order = np.argsort(expanded_seq_ids, kind="stable")
                    expanded_seq_ids = expanded_seq_ids[order]
                    event_times = event_times[order]
                    event_mags = event_mags[order]

                    unique_seq_ids, group_starts = np.unique(
                        expanded_seq_ids,
                        return_index=True,
                    )
                    group_ends = np.concatenate(
                        [group_starts[1:], np.asarray([expanded_seq_ids.shape[0]])]
                    )
                    for seq_idx, start_idx, end_idx in zip(
                        unique_seq_ids.tolist(),
                        group_starts.tolist(),
                        group_ends.tolist(),
                    ):
                        initial_time_chunks[int(seq_idx)].append(event_times[start_idx:end_idx])
                        initial_mag_chunks[int(seq_idx)].append(event_mags[start_idx:end_idx])

        mu_value = float(self.mu.detach().cpu().item())
        duration = float(t_end - t_start)
        if mu_value > 0.0:
            counts_bg = init_rng.poisson(mu_value * duration, size=int(batch_size))
            total_count = int(counts_bg.sum())
            if total_count > 0:
                bg_times = t_start + np.sort(init_rng.random(total_count) * duration)
                bg_mags = _gen_mag(
                    init_rng,
                    total_count,
                    b=mag_params.b,
                    m_min=mag_params.m_min,
                    m_max=mag_params.m_max,
                )
                offset = 0
                for seq_idx, seq_count in enumerate(counts_bg.tolist()):
                    seq_count = int(seq_count)
                    if seq_count <= 0:
                        continue
                    seq_times = bg_times[offset : offset + seq_count]
                    seq_mags = bg_mags[offset : offset + seq_count]
                    initial_time_chunks[seq_idx].append(seq_times)
                    initial_mag_chunks[seq_idx].append(
                        seq_mags.astype(np.float32, copy=False)
                    )
                    offset += seq_count

        if batched_bg_times is not None:
            for seq_idx, bg_times in enumerate(batched_bg_times):
                if bg_times.size == 0:
                    continue
                bg_mags = _gen_mag(
                    init_rng,
                    int(bg_times.size),
                    b=mag_params.b,
                    m_min=mag_params.m_min,
                    m_max=mag_params.m_max,
                )
                initial_time_chunks[seq_idx].append(np.asarray(bg_times, dtype=np.float64))
                initial_mag_chunks[seq_idx].append(bg_mags.astype(np.float32, copy=False))

        initial_events: list[tuple[np.ndarray, np.ndarray]] = []
        for time_parts, mag_parts in zip(initial_time_chunks, initial_mag_chunks):
            if not time_parts:
                initial_events.append(_empty_initial_event_arrays())
                continue
            seq_times = np.concatenate(time_parts, axis=0).astype(np.float64, copy=False)
            seq_mags = np.concatenate(mag_parts, axis=0).astype(np.float32, copy=False)
            order = np.argsort(seq_times, kind="mergesort")
            initial_events.append((seq_times[order], seq_mags[order]))
        return initial_events

    def _sample_single_sequence(
        self,
        *,
        t_start: float,
        t_end: float,
        past_seq: Optional[Sequence],
        rng: np.random.Generator,
        max_length: int,
        history_cache: Optional[dict[str, Any]] = None,
        preset_bg_times: Optional[np.ndarray] = None,
        preset_initial_events: Optional[tuple[np.ndarray, np.ndarray]] = None,
    ) -> Sequence:
        dtype = self.M_c.dtype
        if history_cache is None:
            history_cache = self._prepare_history_sampling_cache(
                t_start=t_start,
                t_end=t_end,
                past_seq=past_seq,
            )
        mag_params = self._sampling_magnitude_params()
        state = self._clone_sampling_state(history_cache["state"])
        last_time = float(history_cache["last_time"])
        event_heap: list[tuple[float, int, float]] = []
        event_counter = itertools.count()
        future_events = _SequenceEventAccumulator()
        pending_initial_times: Optional[np.ndarray] = None
        pending_initial_magnitudes: Optional[np.ndarray] = None
        pending_initial_idx = 0

        if preset_initial_events is not None:
            pending_initial_times = np.asarray(
                preset_initial_events[0],
                dtype=np.float64,
            )
            pending_initial_magnitudes = np.asarray(
                preset_initial_events[1],
                dtype=np.float32,
            )
            if pending_initial_times.shape != pending_initial_magnitudes.shape:
                raise ValueError(
                    "preset_initial_events time and magnitude arrays must have identical shapes."
                )
        else:
            for parent_payload in history_cache["parents"]:
                children = self._sample_offspring_from_cached_means(
                    parent_time=float(parent_payload["time"]),
                    lower=float(parent_payload["lower"]),
                    upper=float(parent_payload["upper"]),
                    offspring_means=parent_payload["offspring_means"],
                    rng=rng,
                    mag_params=mag_params,
                )
                for child_time, child_mag in children:
                    heapq.heappush(
                        event_heap,
                        (float(child_time), next(event_counter), float(child_mag)),
                    )

            background_times = self._sample_background_times_np(
                t_start=t_start,
                t_end=t_end,
                rng=rng,
                preset_bg_times=preset_bg_times,
            )
            if background_times.size > 0:
                background_mags = _gen_mag(
                    rng,
                    int(background_times.size),
                    b=mag_params.b,
                    m_min=mag_params.m_min,
                    m_max=mag_params.m_max,
                )
                for event_time, event_mag in zip(background_times.tolist(), background_mags.tolist()):
                    heapq.heappush(
                        event_heap,
                        (float(event_time), next(event_counter), float(event_mag)),
                    )

        if past_seq is None:
            last_time = float(t_start)

        while (
            event_heap
            or (
                pending_initial_times is not None
                and pending_initial_idx < int(pending_initial_times.shape[0])
            )
        ):
            next_pending_time = math.inf
            if (
                pending_initial_times is not None
                and pending_initial_idx < int(pending_initial_times.shape[0])
            ):
                next_pending_time = float(pending_initial_times[pending_initial_idx])

            next_heap_time = event_heap[0][0] if event_heap else math.inf
            if next_pending_time <= next_heap_time:
                event_time = next_pending_time
                assert pending_initial_magnitudes is not None
                event_mag = float(pending_initial_magnitudes[pending_initial_idx])
                pending_initial_idx += 1
            else:
                event_time, _, event_mag = heapq.heappop(event_heap)
            if event_time > t_end:
                continue

            context, state = self._encoder_step_event(
                inter_time=float(event_time - last_time),
                magnitude=float(event_mag),
                prev_state=state,
                dtype=dtype,
            )
            last_time = float(event_time)
            future_events.append(event_time, event_mag)
            if len(future_events.times) > int(max_length):
                raise RuntimeError(
                    f"NETAS sampling exceeded max_length={max_length}; "
                    "consider lowering eta_max or shortening the horizon."
                )

            children = self._sample_offspring_for_parent(
                parent_time=float(event_time),
                parent_mag=float(event_mag),
                parent_context=context,
                lower=0.0,
                upper=max(t_end - float(event_time), 0.0),
                rng=rng,
                mag_params=mag_params,
            )
            for child_time, child_mag in children:
                heapq.heappush(
                    event_heap,
                    (float(child_time), next(event_counter), float(child_mag)),
                )

        return future_events.to_sequence(t_start=t_start, t_end=t_end)

    def _can_use_batched_rnn_sampling(self, *, batch_size: int) -> bool:
        encoder = self.event_encoder
        return (
            int(batch_size) > 1
            and isinstance(encoder, RNNTPPBackbone)
            and encoder.num_extra_features is None
        )

    def _sample_batch_sequences_rnn(
        self,
        *,
        batch_size: int,
        t_start: float,
        t_end: float,
        child_seeds: list[np.random.SeedSequence],
        max_length: int,
        history_cache: dict[str, Any],
        preset_initial_events_batch: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[Sequence]:
        encoder = self.event_encoder
        if not isinstance(encoder, RNNTPPBackbone):
            raise TypeError("_sample_batch_sequences_rnn requires RNNTPPBackbone.")
        if encoder.num_extra_features is not None:
            raise ValueError("NETAS sampling does not support extra encoder features.")

        dtype = self.M_c.dtype
        mag_params = self._sampling_magnitude_params()
        base_state = history_cache["state"]
        if not torch.is_tensor(base_state) or base_state.ndim != 3 or base_state.shape[1] != 1:
            raise ValueError("RNN sampling state must have shape [num_layers, 1, context_size].")
        state = base_state.repeat(1, int(batch_size), 1).contiguous()

        rngs = [np.random.default_rng(seed) for seed in child_seeds]
        event_heaps: list[list[tuple[float, int, float]]] = [[] for _ in range(int(batch_size))]
        event_counters = [itertools.count() for _ in range(int(batch_size))]
        future_events = [
            _SequenceEventAccumulator()
            for _ in range(int(batch_size))
        ]
        last_times = np.full(
            int(batch_size),
            float(history_cache["last_time"]),
            dtype=np.float64,
        )

        pending_initial_times: list[np.ndarray] = []
        pending_initial_magnitudes: list[np.ndarray] = []
        pending_initial_idx = np.zeros(int(batch_size), dtype=np.int64)
        for event_times, event_mags in preset_initial_events_batch:
            event_times = np.asarray(event_times, dtype=np.float64)
            event_mags = np.asarray(event_mags, dtype=np.float32)
            if event_times.shape != event_mags.shape:
                raise ValueError(
                    "preset_initial_events time and magnitude arrays must have identical shapes."
                )
            pending_initial_times.append(event_times)
            pending_initial_magnitudes.append(event_mags)

        while True:
            active_indices: list[int] = []
            active_times: list[float] = []
            active_magnitudes: list[float] = []

            for seq_idx in range(int(batch_size)):
                next_pending_time = math.inf
                if pending_initial_idx[seq_idx] < int(pending_initial_times[seq_idx].shape[0]):
                    next_pending_time = float(
                        pending_initial_times[seq_idx][pending_initial_idx[seq_idx]]
                    )

                next_heap_time = (
                    event_heaps[seq_idx][0][0] if event_heaps[seq_idx] else math.inf
                )
                if math.isinf(next_pending_time) and math.isinf(next_heap_time):
                    continue

                if next_pending_time <= next_heap_time:
                    event_time = next_pending_time
                    event_mag = float(
                        pending_initial_magnitudes[seq_idx][pending_initial_idx[seq_idx]]
                    )
                    pending_initial_idx[seq_idx] += 1
                else:
                    event_time, _, event_mag = heapq.heappop(event_heaps[seq_idx])

                if event_time > t_end:
                    continue

                active_indices.append(seq_idx)
                active_times.append(float(event_time))
                active_magnitudes.append(float(event_mag))

            if not active_indices:
                break

            active_indices_t = torch.as_tensor(
                active_indices,
                device=self.device,
                dtype=torch.long,
            )
            active_times_np = np.asarray(active_times, dtype=np.float64)
            active_magnitudes_np = np.asarray(active_magnitudes, dtype=np.float64)
            active_indices_np = np.asarray(active_indices, dtype=np.int64)
            inter_times_np = active_times_np - last_times[active_indices_np]
            inter_time_t = torch.as_tensor(
                inter_times_np,
                device=self.device,
                dtype=dtype,
            ).unsqueeze(-1)
            magnitude_t = torch.as_tensor(
                active_magnitudes_np,
                device=self.device,
                dtype=dtype,
            ).unsqueeze(-1)

            prev_state = state.index_select(1, active_indices_t).contiguous()
            context, next_state = self._encoder_step_event_batch_rnn(
                inter_time=inter_time_t,
                magnitude=magnitude_t,
                prev_state=prev_state,
            )
            state.index_copy_(1, active_indices_t, next_state)

            for seq_idx, event_time, event_mag in zip(
                active_indices,
                active_times,
                active_magnitudes,
            ):
                last_times[seq_idx] = float(event_time)
                future_events[seq_idx].append(event_time, event_mag)
                if len(future_events[seq_idx].times) > int(max_length):
                    raise RuntimeError(
                        f"NETAS sampling exceeded max_length={max_length}; "
                        "consider lowering eta_max or shortening the horizon."
                    )

            event_time_t = torch.as_tensor(
                active_times_np,
                device=self.device,
                dtype=dtype,
            )
            upper_t = (
                torch.as_tensor(float(t_end), device=self.device, dtype=dtype)
                - event_time_t
            ).clamp_min(0.0)
            interval_mass = self.basis.interval_mass(torch.zeros_like(upper_t), upper_t)
            eta_t, omega_t = self.parent_params_from_context(context, magnitude_t)
            offspring_means = (
                eta_t.squeeze(1).unsqueeze(-1)
                * omega_t.squeeze(1)
                * interval_mass
            )
            offspring_means_np = _to_numpy_array(offspring_means).astype(
                np.float64,
                copy=False,
            )

            for local_idx, seq_idx in enumerate(active_indices):
                upper = max(float(t_end) - float(active_times[local_idx]), 0.0)
                children = self._sample_offspring_from_cached_means(
                    parent_time=float(active_times[local_idx]),
                    lower=0.0,
                    upper=upper,
                    offspring_means=offspring_means_np[local_idx],
                    rng=rngs[seq_idx],
                    mag_params=mag_params,
                )
                for child_time, child_mag in children:
                    heapq.heappush(
                        event_heaps[seq_idx],
                        (
                            float(child_time),
                            next(event_counters[seq_idx]),
                            float(child_mag),
                        ),
                    )

        return [
            events.to_sequence(t_start=t_start, t_end=t_end)
            for events in future_events
        ]

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        bg_cache_seq: Optional[src.data.Sequence] = None,
        random_state: int = 123,
        max_length: int = 50_000,
        return_sequences: bool = False,
    ) -> Union[Batch, list[Sequence]]:
        if int(batch_size) < 1:
            raise ValueError("batch_size must be >= 1.")
        if float(duration) <= 0.0:
            raise ValueError("duration must be positive.")
        if past_seq is not None:
            t_start = float(past_seq.t_end)

        t_end = float(t_start + duration)
        self._ensure_bg_sampling_cache(
            t_start=float(t_start),
            t_end=t_end,
            past_seq=past_seq,
            bg_cache_seq=bg_cache_seq,
        )
        seed_sequence = np.random.SeedSequence(int(random_state))
        child_seeds = seed_sequence.spawn(int(batch_size))
        history_cache = self._prepare_history_sampling_cache(
            t_start=float(t_start),
            t_end=t_end,
            past_seq=past_seq,
        )
        batched_bg_times = self._prepare_batched_bg_times(
            batch_size=int(batch_size),
            t_start=float(t_start),
            t_end=t_end,
        )
        init_rng = np.random.default_rng(seed_sequence.spawn(1)[0])
        preset_initial_events_batch = self._build_batched_initial_events(
            batch_size=int(batch_size),
            t_start=float(t_start),
            t_end=t_end,
            history_cache=history_cache,
            init_rng=init_rng,
            batched_bg_times=batched_bg_times,
        )

        if past_seq is not None:
            estimated_history_work = int(batch_size) * int(past_seq.num_events)
            if estimated_history_work >= 20_000:
                logger.warning(
                    "NETAS sampling may be slow: batch_size=%s, past_events=%s, "
                    "estimated parent-history work=%s. Consider reducing "
                    "`samples_per_batch` or forecast horizon.",
                    int(batch_size),
                    int(past_seq.num_events),
                    estimated_history_work,
                )

        progress_stride = 0
        if int(batch_size) >= 200:
            progress_stride = max(1, int(batch_size) // 10)

        if self._can_use_batched_rnn_sampling(batch_size=int(batch_size)):
            sequences = self._sample_batch_sequences_rnn(
                batch_size=int(batch_size),
                t_start=float(t_start),
                t_end=t_end,
                child_seeds=child_seeds,
                max_length=int(max_length),
                history_cache=history_cache,
                preset_initial_events_batch=preset_initial_events_batch,
            )
        else:
            sequences = []
            for sample_idx, child_seed in enumerate(child_seeds, start=1):
                rng = np.random.default_rng(child_seed)
                sequences.append(
                    self._sample_single_sequence(
                        t_start=float(t_start),
                        t_end=t_end,
                        past_seq=past_seq,
                        rng=rng,
                        max_length=int(max_length),
                        history_cache=history_cache,
                        preset_bg_times=batched_bg_times[sample_idx - 1],
                        preset_initial_events=preset_initial_events_batch[sample_idx - 1],
                    )
                )
                if progress_stride and (
                    sample_idx % progress_stride == 0 or sample_idx == int(batch_size)
                ):
                    logger.info(
                        "NETAS sampling progress: %s/%s sequences completed.",
                        sample_idx,
                        int(batch_size),
                    )

        if return_sequences:
            return sequences
        return Batch.from_list(sequences).to(self.device)


__all__ = ["NETASSamplingMixin"]
