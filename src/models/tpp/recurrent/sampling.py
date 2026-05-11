from __future__ import annotations

import inspect
from typing import Any, List, Optional, Tuple, Union

import torch

import src
import src.distributions as dist
from ..common.sequence_ops import build_sample_batch, evaluate_compensator_from_model

_VALID_B_SAMPLING_MODES = {"model", "updater"}


class RecurrentTPPSamplingMixin:
    """Shared sampling/evaluation logic for recurrent-style TPP models."""

    def _sampling_state_step(
        self,
        rnn_input: torch.Tensor,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Advance autoregressive state by one step."""
        raise NotImplementedError

    def sample_next_inter_time(
        self,
        inter_time_dist: dist.MixtureSameFamily,
        t_last_event: Optional[torch.Tensor] = None,
        lower_bound: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if lower_bound is None:
            inter_time_h = inter_time_dist.sample()
        else:
            inter_time_h = inter_time_dist.sample_conditional(lower_bound=lower_bound) - lower_bound

        if getattr(self, "bg_model", None) is None:
            return inter_time_h

        assert t_last_event is not None, "t_last_event must be provided when using bg_model"
        dt = inter_time_h.squeeze(-1)
        if lower_bound is None:
            t0 = t_last_event
            inter_time = self.bg_model.sample_nhpp_inverse(inter_time_h.shape[0], t0=t0, dt=dt)
        else:
            t0 = t_last_event + lower_bound
            inter_time = self.bg_model.sample_nhpp_inverse(inter_time_h.shape[0], t0=t0, dt=dt)

        if (inter_time < 0.0).any():
            raise ValueError(
                "Sampled inter-event time should be non-negative. "
                f"Got minimum value {inter_time.min()}"
            )
        return inter_time.unsqueeze(-1)

    def _build_sampling_b_updater(
        self,
        *,
        updater_name: str | None = None,
        updater_cfg: Optional[dict[str, Any]] = None,
    ):
        """Optional hook for models to provide a default updater instance."""
        del updater_name, updater_cfg
        return None

    def _normalize_b_sampling_mode(
        self,
        *,
        b_sampling: Optional[str] = None,
    ) -> str:
        mode = "model" if b_sampling is None else str(b_sampling).strip().lower()
        if mode not in _VALID_B_SAMPLING_MODES:
            raise ValueError(
                "b_sampling must be one of ['model', 'updater'] "
                f"(got {b_sampling!r})."
            )
        return mode

    def _resolve_sampling_updater(
        self,
        *,
        b_sampling: str,
        updater: Any = None,
        updater_name: Optional[str] = None,
        updater_cfg: Optional[dict[str, Any]] = None,
    ):
        if b_sampling == "model":
            return None

        from src.models.updaters import resolve_sampling_updater

        return resolve_sampling_updater(
            model=self,
            updater=updater,
            updater_name=updater_name,
            updater_cfg=updater_cfg,
            device=getattr(self, "device", None),
            prefer_model_hook=True,
        )

    def _prepare_sampling_updater(
        self,
        *,
        batch_size: int,
        past_seq: Optional[src.data.Sequence],
        b_sampling: str,
        updater: Any = None,
        updater_name: Optional[str] = None,
        updater_cfg: Optional[dict[str, Any]] = None,
    ):
        """Initialize updater from history and expand its state to batch size."""
        if b_sampling == "model":
            return None

        resolved_updater = self._resolve_sampling_updater(
            b_sampling=b_sampling,
            updater=updater,
            updater_name=updater_name,
            updater_cfg=updater_cfg,
        )
        if resolved_updater is None:
            raise ValueError(
                "Updater-based sampling requires updater config. "
                "Pass `updater=...`, `updater_name=...`, or implement `_build_sampling_b_updater()`."
            )

        required = ("fit", "sample_b_value", "update_one")
        for name in required:
            if not callable(getattr(resolved_updater, name, None)):
                raise ValueError(f"Updater must implement callable `{name}`.")

        if past_seq is not None:
            prev_write_back = None
            if hasattr(resolved_updater, "write_back"):
                prev_write_back = bool(resolved_updater.write_back)
                resolved_updater.write_back = False
            try:
                resolved_updater.fit(past_seq, prefix="")
            except TypeError as exc:
                raise ValueError(
                    "Updater `fit` must support signature `fit(seq, prefix='')`."
                ) from exc
            finally:
                if prev_write_back is not None:
                    resolved_updater.write_back = prev_write_back

        if batch_size == 1:
            return resolved_updater

        expand_fn = getattr(resolved_updater, "expand_state", None)
        if callable(expand_fn):
            try:
                expand_fn(batch_size)
            except NotImplementedError:
                pass
            else:
                return resolved_updater

        clone_fn = getattr(resolved_updater, "clone_for_batch", None)
        if callable(clone_fn):
            cloned = clone_fn(batch_size)
            if not isinstance(cloned, list) or len(cloned) != batch_size:
                raise ValueError(
                    "`clone_for_batch(batch_size)` must return a list of length batch_size."
                )
            for idx, one in enumerate(cloned):
                if not callable(getattr(one, "sample_b_value", None)):
                    raise ValueError(f"Cloned updater at index {idx} missing `sample_b_value`.")
                if not callable(getattr(one, "update_one", None)):
                    raise ValueError(f"Cloned updater at index {idx} missing `update_one`.")
            return cloned

        raise ValueError(
            "For batch sampling, updater must implement either "
            "`expand_state(batch_size)` or `clone_for_batch(batch_size)`."
        )

    def _sample_b_from_updater(
        self,
        updater_state,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Read per-batch b values from updater state."""
        mode = "mean"

        if isinstance(updater_state, list):
            vals = []
            for updater in updater_state:
                val = updater.sample_b_value(mode=mode)
                if isinstance(val, torch.Tensor):
                    if val.numel() != 1:
                        raise ValueError("Per-sequence updater must return scalar b value.")
                    vals.append(float(val.reshape(-1)[0].item()))
                else:
                    vals.append(float(val))
            b = torch.tensor(vals, device=self.device, dtype=dtype)
        else:
            b = updater_state.sample_b_value(mode=mode)
            if not isinstance(b, torch.Tensor):
                b = torch.tensor(float(b), device=self.device, dtype=dtype)
            else:
                b = b.to(device=self.device, dtype=dtype)
            if b.ndim == 0:
                b = b.reshape(1).expand(batch_size)
            elif b.ndim != 1:
                raise ValueError(f"Updater sampled b must be 1D [B] or scalar, got {tuple(b.shape)}.")
            if b.shape[0] != batch_size:
                raise ValueError(
                    f"Updater sampled b shape mismatch: expected batch={batch_size}, got {tuple(b.shape)}."
                )
        return b.unsqueeze(-1)

    def _update_updater_state(
        self,
        updater_state,
        *,
        mags: torch.Tensor,
        times: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> None:
        """Update updater state with newly sampled magnitudes and event times."""
        if active_mask.ndim != 1 or active_mask.shape[0] != mags.shape[0]:
            raise ValueError("active_mask must have shape [B] matching mags/times batch dimension.")

        if isinstance(updater_state, list):
            for i, updater in enumerate(updater_state):
                if bool(active_mask[i].item()):
                    updater.update_one(mag=mags[i, 0], time=times[i])
            return

        mag_vec = mags.squeeze(-1)
        updater_state.update_one(mag=mag_vec, time=times, active_mask=active_mask)

    def _clamp_updater_b_values(self, b_values: torch.Tensor) -> torch.Tensor:
        """Clamp updater-provided b values to model range when available."""
        if b_values.ndim != 2 or b_values.shape[1] != 1:
            raise ValueError(
                "Updater b values must have shape [B, 1] before clamping, "
                f"got {tuple(b_values.shape)}."
            )

        lower = getattr(self, "b_min", None)
        upper = getattr(self, "b_max", None)
        if lower is None and upper is None:
            return b_values

        if lower is not None and upper is not None and float(lower) > float(upper):
            raise ValueError(
                f"Invalid b clamp range: b_min={float(lower)} > b_max={float(upper)}."
            )

        min_val = float(lower) if lower is not None else None
        max_val = float(upper) if upper is not None else None
        return torch.clamp(b_values, min=min_val, max=max_val)

    def _infer_sampling_dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except Exception:
            return torch.get_default_dtype()

    def _sampling_state_transition(
        self,
        *,
        rnn_input: torch.Tensor,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        step_out = self._sampling_state_step(rnn_input, current_hidden)
        if isinstance(step_out, tuple):
            if len(step_out) != 2:
                raise ValueError(
                    "_sampling_state_step must return `(next_state, next_hidden)` "
                    "or `next_state`."
                )
            next_state, next_hidden = step_out
        else:
            next_state, next_hidden = step_out, current_hidden
        return next_state, next_hidden

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        return_sequences: bool = False,
        *,
        predict_b: Optional[bool] = None,
        b_sampling: str = "model",
        updater: Any = None,
        updater_name: Optional[str] = None,
        updater_cfg: Optional[dict[str, Any]] = None,
        return_b_values: bool = False,
    ) -> Union[src.data.Batch, List[src.data.Sequence]]:
        """
        Sample future events from recurrent TPP model.

        Primary updater-related args:
            b_sampling: one of ['model', 'updater']
            updater: explicit updater instance
            updater_name/updater_cfg: updater registry spec
        """
        use_magnitude = bool(getattr(self, "input_magnitude", True))
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")
        predict_b = getattr(self, "predict_b", False) if predict_b is None else bool(predict_b)

        b_sampling = self._normalize_b_sampling_mode(b_sampling=b_sampling)

        if past_seq is not None:
            t_start = past_seq.t_end
            past_batch = src.data.Batch.from_list([past_seq])
            context_hidden_fn = getattr(self, "_get_context_and_hidden", None)
            if callable(context_hidden_fn):
                context, hidden = context_hidden_fn(past_batch)
            elif hasattr(self, "backbone") and hasattr(self.backbone, "get_context_and_hidden"):
                context, hidden = self.backbone.get_context_and_hidden(past_batch)
            else:
                raise ValueError(
                    "Sampling with `past_seq` requires `_get_context_and_hidden(batch)` "
                    "or `self.backbone.get_context_and_hidden(batch)`."
                )
            current_state = context[:, [-1], :].expand(batch_size, -1, -1).contiguous()
            current_hidden = hidden.expand(-1, batch_size, -1).contiguous()
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
        else:
            target_dtype = self._infer_sampling_dtype()
            current_state = torch.zeros(
                batch_size,
                1,
                self.context_size,
                device=self.device,
                dtype=target_dtype,
            )
            if hasattr(self, "backbone") and hasattr(self.backbone, "num_rnn_layers"):
                num_rnn_layers = int(self.backbone.num_rnn_layers)
            else:
                num_rnn_layers = int(getattr(self, "num_rnn_layers", 1))
            current_hidden = torch.zeros(
                num_rnn_layers,
                batch_size,
                self.context_size,
                device=self.device,
                dtype=target_dtype,
            )
            time_remaining = None

        t_end = t_start + duration
        inter_time_list: list[torch.Tensor] = []
        mag_list: list[torch.Tensor] = []
        b_list: list[torch.Tensor] = []
        total_time = torch.zeros(batch_size, device=self.device, dtype=current_state.dtype)
        updater_state = self._prepare_sampling_updater(
            batch_size=batch_size,
            past_seq=past_seq,
            b_sampling=b_sampling,
            updater=updater,
            updater_name=updater_name,
            updater_cfg=updater_cfg,
        )

        mag_sig = inspect.signature(self.get_magnitude_dist)
        supports_predict_b_arg = "predict_b" in mag_sig.parameters
        duration_t = current_state.new_tensor(float(t_end - t_start))
        while True:
            active_mask = total_time < duration_t
            if not bool(active_mask.any().item()):
                break

            if time_remaining is None:
                t_last_event = t_start + total_time
            else:
                t_last_event = past_seq.arrival_times[-1].repeat(batch_size)
            time_context = current_state
            time_context_hook = getattr(self, "_get_sampling_time_context", None)
            if callable(time_context_hook):
                time_context = time_context_hook(
                    current_state=current_state,
                    t_last_event=t_last_event,
                    lower_bound=time_remaining,
                )
            inter_time_dist = self.get_inter_time_dist(time_context)

            next_inter_times = self.sample_next_inter_time(
                inter_time_dist,
                t_last_event=t_last_event,
                lower_bound=time_remaining,
            )
            if isinstance(next_inter_times, tuple):
                next_inter_times = next_inter_times[0]
            time_remaining = None
            next_inter_times.clamp_max_(t_end - t_start)
            next_inter_times[~active_mask, 0] = 0.0
            inter_time_list.append(next_inter_times)

            rnn_input_list = [self.encode_time(next_inter_times)]
            if use_magnitude:
                if b_sampling == "model":
                    if supports_predict_b_arg:
                        mag_dist = self.get_magnitude_dist(current_state, predict_b=predict_b)
                    else:
                        mag_dist = self.get_magnitude_dist(current_state)
                    next_mag = mag_dist.sample()
                else:
                    b_step = self._sample_b_from_updater(
                        updater_state,
                        batch_size=batch_size,
                        dtype=current_state.dtype,
                    )
                    b_step = self._clamp_updater_b_values(b_step)
                    if return_b_values:
                        b_list.append(b_step)
                    mag_min = current_state.new_full(b_step.shape, float(self.mag_completeness))
                    mag_max = torch.as_tensor(
                        self.M_m,
                        dtype=current_state.dtype,
                        device=current_state.device,
                    )
                    if torch.any(mag_max <= mag_min):
                        raise ValueError("M_m must be greater than mag_completeness for sampling.")
                    mag_dist = dist.GutenbergRichter(
                        b=b_step,
                        mag_min=mag_min,
                        mag_max=mag_max,
                    )
                    next_mag = mag_dist.sample()
                    next_mag[~active_mask, 0] = float(self.mag_completeness)
                    t_event = t_start + total_time + next_inter_times.squeeze(-1)
                    self._update_updater_state(
                        updater_state,
                        mags=next_mag if next_mag.ndim == 2 else next_mag.unsqueeze(-1),
                        times=t_event,
                        active_mask=active_mask,
                    )

                if next_mag.ndim == 1:
                    next_mag = next_mag.unsqueeze(-1)
                mag_list.append(next_mag)
                rnn_input_list.append(self.encode_magnitude(next_mag))

            rnn_input = torch.cat(rnn_input_list, dim=-1).contiguous()
            next_state, next_hidden = self._sampling_state_transition(
                rnn_input=rnn_input,
                current_hidden=current_hidden,
            )
            next_state = next_state.detach()
            next_hidden = next_hidden.detach()
            state_mask = active_mask.view(batch_size, 1, 1)
            hidden_mask = active_mask.view(1, batch_size, 1)
            current_state = torch.where(state_mask, next_state, current_state)
            current_hidden = torch.where(hidden_mask, next_hidden, current_hidden)

            total_time = total_time + next_inter_times.squeeze(-1)

        inter_times = torch.cat(inter_time_list, dim=1)
        magnitudes = torch.cat(mag_list, dim=1) if use_magnitude else None

        duration = t_end - t_start
        unclipped_arrival_times = inter_times.cumsum(-1)
        epsilon = 1e-5
        padding_mask = unclipped_arrival_times > duration - epsilon
        inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
        end_idx = (1 - padding_mask.long()).sum(-1)
        last_surv_time = (duration - inter_times.sum(-1)).clamp_min(0.0)
        inter_times[torch.arange(batch_size, device=self.device), end_idx] = last_surv_time

        batch = build_sample_batch(
            inter_times=inter_times,
            t_start=t_start,
            t_end=t_end,
            device=self.device,
            magnitudes=magnitudes,
            time_dtype=torch.float32,
            epsilon=epsilon,
            clamp_last_surv_time=True,
        )
        if return_b_values and len(b_list) > 0:
            b_values = torch.cat(b_list, dim=1)
            b_values = torch.masked_fill(b_values, padding_mask, float("nan"))
            batch["b_sample"] = b_values
        return batch.to_list() if return_sequences else batch

    def evaluate_compensator(
        self,
        sequence: src.data.Sequence,
        num_grid_points: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return evaluate_compensator_from_model(
            model=self,
            sequence=sequence,
            num_grid_points=num_grid_points,
        )
