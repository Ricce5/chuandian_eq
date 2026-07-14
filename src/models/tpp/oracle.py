from __future__ import annotations

import math
import warnings
from typing import Iterable

import numpy as np
import torch

import src
import src.distributions as dist

from .common.oracle_blocks import OracleDistDecoder, OracleFCNDecoder, OracleRNNEncoder
from .tpp_model import TPPModel


class Oracle(TPPModel):
    """Oracle baseline ported from ORACLE-main with minimal integration changes."""

    REQUIRED_EVENT_MARKS = ("aRs", "vm", "dVc", "sv", "dTS")
    BASE_MARK_ORDER = ("time", "aRs", "mag", "Mc", "vm", "dVc", "sv", "dTS")
    DEFAULT_FUTURE_FEATURE_NAMES = ("vm", "sv", "dTS", "Mc")
    VALID_LOSS_MODES = ("causal", "legacy_fit", "legacy_forecast")
    VALID_SAMPLING_MODES = ("autoregressive",)
    _INTER_TIME_MIN = 1e-10
    _INTER_TIME_MAX = 1e10
    _MARK_MIN = -10.0
    _MARK_MAX = 10.0

    def __init__(self, args, device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.reduction = getattr(args, "loss_reduction", "per_event")

        self.input_magnitude = bool(getattr(args, "input_magnitude", True))
        self.input_injection = bool(getattr(args, "input_injection", True))
        self.train_to_forecast = bool(getattr(args, "train_to_forecast", False))
        self.oracle_loss_mode = self._resolve_loss_mode(args)
        self.oracle_forecast_count = int(getattr(args, "oracle_forecast_count", 5))
        if self.oracle_forecast_count < 0:
            raise ValueError("oracle_forecast_count must be >= 0.")
        self.oracle_sampling_mode = self._resolve_sampling_mode(args)
        self.sampling_mag_b = float(
            self._first_not_none_from_args(
                args,
                ("sampling_mag_b", "richter_b_mle", "richter_b", "b_value"),
                default=1.0,
            )
        )
        self.sampling_mag_max = float(
            self._first_not_none_from_args(args, ("mag_max",), default=10.0)
        )

        self.supplementary_mark_list = self._parse_mark_list(
            getattr(args, "supplementary_mark_list", ())
        )
        self.base_mark_order = self._parse_base_mark_order(
            getattr(args, "oracle_base_mark_order", self.BASE_MARK_ORDER)
        )
        self.base_mark_index = {name: idx for idx, name in enumerate(self.base_mark_order)}
        self.future_feature_names = self._parse_future_feature_names(
            getattr(args, "oracle_future_feature_names", self.DEFAULT_FUTURE_FEATURE_NAMES)
        )
        self._validate_feature_gate_flags()
        self.num_dist_components = int(getattr(args, "num_components", 32))
        if self.num_dist_components < 1:
            raise ValueError("num_components must be >= 1 for Oracle.")

        tau_mean = max(float(getattr(args, "tau_mean", 1.0)), 1e-10)
        log_tau_mean = math.log10(tau_mean)
        if hasattr(args, "log_tau_std"):
            # preparation.py computes std in ln-space; convert to log10-space here.
            raw_log_tau_std = float(args.log_tau_std)
            if not math.isfinite(raw_log_tau_std):
                raise ValueError(f"args.log_tau_std must be finite, got {raw_log_tau_std}.")
            if raw_log_tau_std <= 0:
                raise ValueError(f"args.log_tau_std must be > 0, got {raw_log_tau_std}.")
            log_tau_std = max(raw_log_tau_std / math.log(10.0), 1e-8)
        else:
            log_tau_std = float(getattr(args, "oracle_log_tau_std", 2.0))
        if log_tau_std <= 0:
            raise ValueError("oracle_log_tau_std must be > 0.")
        self.register_buffer("log_tau_mean", torch.tensor(log_tau_mean, dtype=torch.float32))
        self.register_buffer("log_tau_std", torch.tensor(log_tau_std, dtype=torch.float32))
        self.register_buffer("mag_mean", torch.tensor(float(getattr(args, "mag_mean", 0.0)), dtype=torch.float32))
        self.register_buffer(
            "mag_completeness",
            torch.tensor(float(getattr(args, "mag_completeness", 0.0)), dtype=torch.float32),
        )

        # Mark order follows configured base layout:
        # [time, aRs, mag, Mc, vm, dVc, sv, dTS] by default, then supplementary marks.
        self.num_marks = len(self.base_mark_order) + len(self.supplementary_mark_list)
        future_feature_idx = self._resolve_feature_indices(self.future_feature_names)
        encoder_type = str(getattr(args, "oracle_encoder_type", "GRU")).strip()
        if encoder_type.lower() == "none":
            self.encoder = None
            self.context_size = self.num_marks
        else:
            self.encoder = OracleRNNEncoder(
                rnn_type=encoder_type,
                d_model_in=self.num_marks,
                num_layers=int(getattr(args, "num_rnn_layers", 1)),
                dropout_prob=float(getattr(args, "rnn_dropout", 0.2)),
            )
            self.context_size = 2 * self.num_marks

        decoder_type = str(getattr(args, "oracle_decoder_type", "FCN")).strip().upper()
        if decoder_type != "FCN":
            raise ValueError("Only oracle_decoder_type='FCN' is supported in this integration.")
        self.decoder = OracleFCNDecoder(
            d_model_in=self.context_size,
            d_model_ff=self.context_size,
            d_model_out=3 * self.num_dist_components,
            lookback_size=int(getattr(args, "oracle_lookback_size", 1)),
            num_hidden_layers=int(getattr(args, "oracle_decoder_hidden_layers", 1)),
            dropout_prob=float(getattr(args, "rnn_dropout", 0.2)),
            future_feature_idx=future_feature_idx,
        )
        self.dist_decoder = OracleDistDecoder(num_dist_components=self.num_dist_components)
        event_feature_cfg = self._event_feature_cfg_from_args(args)
        self.oracle_sampling_smoothing_window = int(event_feature_cfg.get("smoothing_window", 10))
        self.oracle_sampling_activity_epsilon = float(event_feature_cfg.get("activity_epsilon", 0.0))
        self.oracle_sampling_log_floor = float(event_feature_cfg.get("log_floor", -10.0))
        self.oracle_sampling_time_unit_minutes = self._time_unit_minutes_from_args(args)
        self.to(self.device)

    @staticmethod
    def _has_arg(args, name: str) -> bool:
        try:
            return hasattr(args, name)
        except Exception:
            return False

    @classmethod
    def _normalize_loss_mode(cls, value) -> str:
        mode = str(value).strip().lower()
        aliases = {
            "fit": "legacy_fit",
            "legacy": "legacy_fit",
            "forecast": "legacy_forecast",
            "train_to_forecast": "legacy_forecast",
        }
        mode = aliases.get(mode, mode)
        if mode not in cls.VALID_LOSS_MODES:
            raise ValueError(
                f"oracle_loss_mode must be one of {cls.VALID_LOSS_MODES}, got {value!r}."
            )
        return mode

    def _resolve_loss_mode(self, args) -> str:
        explicit_mode = self._has_arg(args, "oracle_loss_mode") and getattr(
            args,
            "oracle_loss_mode",
        ) is not None
        if explicit_mode:
            mode = self._normalize_loss_mode(getattr(args, "oracle_loss_mode"))
            if self.train_to_forecast and mode != "legacy_forecast":
                warnings.warn(
                    "train_to_forecast is deprecated and ignored when oracle_loss_mode "
                    f"is explicitly set to {mode!r}. Use oracle_loss_mode='legacy_forecast' "
                    "only for original-ORACLE replication/ablation.",
                    UserWarning,
                    stacklevel=3,
                )
            return mode

        if self.train_to_forecast:
            warnings.warn(
                "train_to_forecast=True is deprecated. Mapping it to "
                "oracle_loss_mode='legacy_forecast' for compatibility; use "
                "oracle_loss_mode='causal' for the main likelihood/count-forecast comparison.",
                UserWarning,
                stacklevel=3,
            )
            return "legacy_forecast"
        return "causal"

    @classmethod
    def _normalize_sampling_mode(cls, value) -> str:
        mode = str(value).strip().lower()
        aliases = {
            "ar": "autoregressive",
            "auto": "autoregressive",
        }
        mode = aliases.get(mode, mode)
        if mode not in cls.VALID_SAMPLING_MODES:
            raise ValueError(
                "Oracle only supports oracle_sampling_mode='autoregressive' for fair "
                "forward count forecasts. Original open-loop/update_history=False sampling "
                f"is intentionally not exposed (got {value!r})."
            )
        return mode

    def _resolve_sampling_mode(self, args) -> str:
        return self._normalize_sampling_mode(
            getattr(args, "oracle_sampling_mode", "autoregressive")
        )

    @staticmethod
    def _parse_mark_list(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [token.strip() for token in value.split(",") if token.strip()]
        if isinstance(value, Iterable):
            return [str(v).strip() for v in value if str(v).strip()]
        raise TypeError(f"Unsupported supplementary_mark_list type: {type(value)}")

    @staticmethod
    def _find_duplicates(names: Iterable[str]) -> list[str]:
        seen = set()
        duplicates = []
        for name in names:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        return duplicates

    def _parse_base_mark_order(self, value) -> tuple[str, ...]:
        names = tuple(self._parse_mark_list(value))
        if not names:
            raise ValueError("oracle_base_mark_order cannot be empty.")

        duplicates = self._find_duplicates(names)
        if duplicates:
            raise ValueError(
                f"oracle_base_mark_order contains duplicates: {duplicates}."
            )

        unknown = [name for name in names if name not in self.BASE_MARK_ORDER]
        if unknown:
            raise ValueError(
                "oracle_base_mark_order must only use supported base marks. "
                f"Unknown: {unknown}, supported: {self.BASE_MARK_ORDER}."
            )
        return names

    def _parse_future_feature_names(self, value) -> tuple[str, ...]:
        names = tuple(self._parse_mark_list(value))
        duplicates = self._find_duplicates(names)
        if duplicates:
            raise ValueError(
                f"oracle_future_feature_names contains duplicates: {duplicates}."
            )
        unknown = [name for name in names if name not in self.base_mark_index]
        if unknown:
            raise ValueError(
                f"oracle_future_feature_names contains unknown marks: {unknown}. "
                f"Available base features: {self.base_mark_order}."
            )
        return names

    def _validate_feature_gate_flags(self) -> None:
        uses_magnitude_marks = "mag" in self.base_mark_index
        uses_injection_marks = any(
            name in self.base_mark_index for name in self.REQUIRED_EVENT_MARKS
        )
        if uses_magnitude_marks and not self.input_magnitude:
            raise ValueError(
                "Selected base marks include 'mag', so input_magnitude must be True."
            )
        if uses_injection_marks and not self.input_injection:
            raise ValueError(
                "Selected base marks include injection-driven marks "
                f"{self.REQUIRED_EVENT_MARKS}, so input_injection must be True."
            )

    @staticmethod
    def _mapping_get(value, key: str, default=None):
        if value is None:
            return default
        if hasattr(value, "get"):
            return value.get(key, default)
        return getattr(value, key, default)

    @staticmethod
    def _first_not_none_from_args(args, names: Iterable[str], *, default=None):
        for name in names:
            value = getattr(args, name, None)
            if value is not None:
                return value
        return default

    @classmethod
    def _event_feature_cfg_from_args(cls, args) -> dict:
        catalog_cfg = cls._mapping_get(args, "catalog_cfg", {}) or {}
        event_feature_cfg = cls._mapping_get(catalog_cfg, "event_feature_cfg", {}) or {}
        if hasattr(event_feature_cfg, "items"):
            return dict(event_feature_cfg.items())
        return dict(event_feature_cfg)

    @classmethod
    def _time_unit_minutes_from_args(cls, args) -> float:
        catalog_cfg = cls._mapping_get(args, "catalog_cfg", {}) or {}
        freq = cls._mapping_get(catalog_cfg, "freq", None)
        if freq is None:
            return 1.0
        try:
            import pandas as pd

            return float(pd.Timedelta(freq) / pd.Timedelta(minutes=1))
        except Exception:
            return 1.0

    def _resolve_feature_indices(self, feature_names: Iterable[str]) -> tuple[int, ...]:
        names = [str(name).strip() for name in feature_names if str(name).strip()]
        missing = [name for name in names if name not in self.base_mark_index]
        if missing:
            raise ValueError(
                f"Unknown Oracle feature names: {missing}. "
                f"Available base features: {self.base_mark_order}."
            )
        return tuple(self.base_mark_index[name] for name in names)

    def prep_time_marks(self, inter_times: torch.Tensor) -> torch.Tensor:
        log_tau = torch.log10(inter_times.clamp(self._INTER_TIME_MIN, self._INTER_TIME_MAX))
        return (log_tau.unsqueeze(-1) - self.log_tau_mean) / self.log_tau_std

    def prep_mag_marks(self, mag: torch.Tensor) -> torch.Tensor:
        mag = mag.clamp(self._MARK_MIN, self._MARK_MAX)
        return mag.unsqueeze(-1) - self.mag_mean

    def prep_aux_marks(self, mark: torch.Tensor) -> torch.Tensor:
        mark = mark.clamp(self._MARK_MIN, self._MARK_MAX)
        return mark.unsqueeze(-1)

    @staticmethod
    def _as_1d_numpy(value, *, name: str) -> np.ndarray:
        if value is None:
            raise ValueError(f"{name} must be provided.")
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        arr = np.ravel(arr)
        if arr.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional after flattening.")
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values.")
        return arr

    def _resolve_sampling_injection_series(
        self,
        *,
        past_seq: src.data.Sequence | None,
        bg_cache_seq: src.data.Sequence | None = None,
        injection_times=None,
        injection_rates=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if injection_times is not None or injection_rates is not None:
            inj_t = self._as_1d_numpy(injection_times, name="injection_times")
            inj_rate = self._as_1d_numpy(injection_rates, name="injection_rates")
        else:
            source_seq = bg_cache_seq if bg_cache_seq is not None else past_seq
            if source_seq is None:
                raise ValueError(
                    "Oracle sampling needs injection_times/injection_rates, "
                    "or a past_seq/bg_cache_seq carrying time_series and time_series_times."
                )
            raw_series = getattr(source_seq, "raw_time_series", None)
            raw_times = getattr(source_seq, "raw_time_series_times", None)
            if raw_series is not None:
                inj_t = self._as_1d_numpy(
                    raw_times
                    if raw_times is not None
                    else getattr(source_seq, "time_series_times", None),
                    name="raw_time_series_times",
                )
                inj_rate = self._as_1d_numpy(raw_series, name="raw_time_series")
            elif (
                getattr(source_seq, "time_series", None) is None
                or getattr(source_seq, "time_series_times", None) is None
            ):
                raise ValueError(
                    "Oracle sampling needs injection covariates. Provide injection_times/injection_rates "
                    "or pass a sequence with time_series and time_series_times."
                )
            else:
                inj_t = self._as_1d_numpy(source_seq.time_series_times, name="time_series_times")
                inj_rate = self._as_1d_numpy(source_seq.time_series, name="time_series")

        if inj_t.shape[0] != inj_rate.shape[0]:
            raise ValueError(
                f"injection_times and injection_rates must have the same length "
                f"(got {inj_t.shape[0]} and {inj_rate.shape[0]})."
            )
        if inj_t.shape[0] < 2:
            raise ValueError("Oracle sampling requires at least two injection samples.")

        order = np.argsort(inj_t, kind="stable")
        inj_t = inj_t[order]
        inj_rate = inj_rate[order]
        if np.any(np.diff(inj_t) < 0):
            raise ValueError("injection_times must be non-decreasing.")
        return inj_t, inj_rate

    @staticmethod
    def _legacy_elapsed_injection_time(
        inj_t: np.ndarray,
        inj_rate: np.ndarray,
        activity_epsilon: float,
    ) -> np.ndarray:
        active = np.abs(inj_rate) > activity_epsilon
        elapsed = np.zeros_like(inj_t, dtype=np.float64)
        last_active_t = np.nan
        for idx in range(len(inj_t)):
            if active[idx]:
                last_active_t = inj_t[idx]
                elapsed[idx] = 0.0
            elif np.isnan(last_active_t):
                elapsed[idx] = 0.0
            else:
                elapsed[idx] = inj_t[idx] - last_active_t
        return elapsed

    @staticmethod
    def _smoothed_log10_rate_numpy(
        event_times: np.ndarray,
        *,
        time_unit_minutes: float,
        window: int,
        floor: float,
    ) -> np.ndarray:
        inter_times_min = np.diff(event_times, prepend=0.0) * time_unit_minutes
        if np.any(inter_times_min < 0):
            raise ValueError("event_times must be non-decreasing for Oracle sampling.")

        with np.errstate(divide="ignore", invalid="ignore"):
            log_rates = np.log10(np.reciprocal(inter_times_min))

        history = np.full(int(window), float(floor), dtype=np.float64)
        out = np.empty_like(log_rates, dtype=np.float64)
        for idx, value in enumerate(log_rates):
            history = np.concatenate([history[1:], np.array([value], dtype=np.float64)])
            out[idx] = history.mean()
        return out

    def _strictly_increasing_event_times(
        self,
        event_times,
        *,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> np.ndarray:
        arr = np.asarray(event_times, dtype=np.float64).copy()
        if arr.size <= 1:
            return arr
        if not np.isfinite(arr).all():
            raise ValueError("Oracle sampled event times contain non-finite values.")
        if np.any(np.diff(arr) < 0):
            raise ValueError("Oracle sampled event times must be non-decreasing.")

        lo = None if t_start is None else float(t_start)
        hi = None if t_end is None else float(t_end)
        if lo is not None and hi is not None and hi < lo:
            raise ValueError("t_end must be >= t_start for Oracle sampling.")

        span_ref = max(float(np.ptp(arr)), abs(float(arr[-1])), 1.0)
        eps = max(self._INTER_TIME_MIN, 1e-9 * span_ref)
        if lo is not None and hi is not None and arr.size > 1:
            span = hi - lo
            if span <= 0:
                raise ValueError(
                    "Oracle sampling cannot place multiple events in a zero-duration context."
                )
            eps = min(eps, 0.25 * span / arr.size)

        if lo is not None:
            arr = np.maximum(arr, lo)
        if hi is not None:
            arr = np.minimum(arr, hi)

        for idx in range(1, arr.size):
            if arr[idx] <= arr[idx - 1]:
                arr[idx] = arr[idx - 1] + eps

        if hi is not None and arr[-1] > hi:
            arr -= arr[-1] - hi
        if lo is not None and arr[0] < lo:
            arr += lo - arr[0]

        if (lo is not None and arr[0] < lo) or (hi is not None and arr[-1] > hi):
            if lo is not None and hi is not None and hi > lo:
                arr = np.linspace(lo, hi, arr.size, dtype=np.float64)
            else:
                raise ValueError("Oracle sampling could not repair duplicate event times.")

        if np.any(np.diff(arr) <= 0):
            raise ValueError("Oracle sampling could not repair duplicate event times.")
        return arr

    def _build_sampling_event_features(
        self,
        *,
        event_times: np.ndarray,
        injection_times: np.ndarray,
        injection_rates: np.ndarray,
        time_unit_minutes: float,
    ) -> dict[str, torch.Tensor]:
        event_times = np.asarray(event_times, dtype=np.float64)
        if event_times.size == 0:
            empty = torch.empty(0, dtype=torch.float32, device=self.device)
            return {"vm": empty, "dVc": empty, "sv": empty, "dTS": empty, "aRs": empty}
        if np.any(np.diff(event_times) <= 0):
            raise ValueError("Oracle sampled event times must be strictly increasing.")

        dt_inj_min = np.diff(injection_times, prepend=injection_times[0]) * time_unit_minutes
        cumulative_volume = np.cumsum(injection_rates * dt_inj_min)

        vm_lin = np.interp(event_times, injection_times, injection_rates)
        vm_cum = np.interp(event_times, injection_times, cumulative_volume)
        dvm = np.diff(vm_cum, prepend=0.0)
        sv = np.sign(dvm)

        elapsed_inj = self._legacy_elapsed_injection_time(
            injection_times,
            injection_rates,
            self.oracle_sampling_activity_epsilon,
        )
        elapsed_event = np.interp(event_times, injection_times, elapsed_inj)
        ars = self._smoothed_log10_rate_numpy(
            event_times,
            time_unit_minutes=time_unit_minutes,
            window=self.oracle_sampling_smoothing_window,
            floor=self.oracle_sampling_log_floor,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            vm = np.log10(np.abs(vm_lin))
            dvc = np.log10(np.abs(dvm))
            dts = np.log10(elapsed_event)

        return {
            "vm": torch.tensor(vm, dtype=torch.float32, device=self.device),
            "dVc": torch.tensor(dvc, dtype=torch.float32, device=self.device),
            "sv": torch.tensor(sv, dtype=torch.float32, device=self.device),
            "dTS": torch.tensor(dts, dtype=torch.float32, device=self.device),
            "aRs": torch.tensor(ars, dtype=torch.float32, device=self.device),
        }

    def _build_sampling_sequence(
        self,
        *,
        event_times: list[float],
        magnitudes: list[float],
        t_start: float,
        t_end: float,
        injection_times: np.ndarray,
        injection_rates: np.ndarray,
        time_unit_minutes: float,
        include_time_series: bool = True,
        feature_event_times: list[float] | None = None,
        feature_indices: np.ndarray | None = None,
    ) -> src.data.Sequence:
        if len(event_times) != len(magnitudes):
            raise ValueError("event_times and magnitudes must have the same length.")

        if event_times:
            event_arr = self._strictly_increasing_event_times(
                event_times,
                t_start=t_start,
                t_end=t_end,
            )
            boundaries = np.concatenate([[float(t_start)], event_arr, [float(t_end)]])
            inter_times = np.diff(boundaries)
        else:
            event_arr = np.empty(0, dtype=np.float64)
            inter_times = np.asarray([float(t_end) - float(t_start)], dtype=np.float64)

        feature_arr = (
            event_arr
            if feature_event_times is None
            else self._strictly_increasing_event_times(feature_event_times)
        )
        features = self._build_sampling_event_features(
            event_times=feature_arr,
            injection_times=injection_times,
            injection_rates=injection_rates,
            time_unit_minutes=time_unit_minutes,
        )
        if feature_indices is not None:
            idx_tensor = torch.as_tensor(feature_indices, dtype=torch.long, device=self.device)
            features = {key: value[idx_tensor] for key, value in features.items()}
        if any(value.shape[0] != len(event_times) for value in features.values()):
            raise ValueError("Oracle sampling feature length does not match event_times length.")
        kwargs = {
            "mag": torch.tensor(magnitudes, dtype=torch.float32, device=self.device),
            **features,
        }

        if include_time_series:
            ts_mask = (injection_times >= float(t_start)) & (injection_times <= float(t_end))
            if not np.any(ts_mask):
                left = max(0, np.searchsorted(injection_times, float(t_start), side="right") - 1)
                right = min(injection_times.shape[0], left + 2)
                ts_mask = np.zeros_like(injection_times, dtype=bool)
                ts_mask[left:right] = True
            kwargs["time_series"] = torch.tensor(
                injection_rates[ts_mask, None],
                dtype=torch.float32,
                device=self.device,
            )
            kwargs["time_series_times"] = torch.tensor(
                injection_times[ts_mask],
                dtype=torch.float32,
                device=self.device,
            )

        return src.data.Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32, device=self.device),
            t_start=float(t_start),
            **kwargs,
        )

    def _build_sampling_sequence_from_feature_state(
        self,
        *,
        event_times: list[float],
        magnitudes: list[float],
        initial_feature_state: dict[str, object],
        t_start: float,
        t_end: float,
        injection_times: np.ndarray,
        injection_rates: np.ndarray,
    ) -> src.data.Sequence:
        if len(event_times) != len(magnitudes):
            raise ValueError("event_times and magnitudes must have the same length.")

        if event_times:
            event_arr = self._strictly_increasing_event_times(
                event_times,
                t_start=t_start,
                t_end=t_end,
            )
            boundaries = np.concatenate([[float(t_start)], event_arr, [float(t_end)]])
            inter_times = np.diff(boundaries)
        else:
            inter_times = np.asarray([float(t_end) - float(t_start)], dtype=np.float64)

        num_events = len(event_times)
        kwargs = {
            "mag": torch.tensor(magnitudes, dtype=torch.float32, device=self.device),
        }
        feature_state = self._clone_sampling_feature_state(initial_feature_state)
        features = {name: [] for name in self.REQUIRED_EVENT_MARKS}
        feature_event_times = event_arr.tolist() if event_times else []
        for event_time in feature_event_times:
            event_features = self._advance_sampling_feature_state(
                feature_state,
                float(event_time),
            )
            for name in self.REQUIRED_EVENT_MARKS:
                features[name].append(event_features[name])

        for name in self.REQUIRED_EVENT_MARKS:
            values = features[name]
            if len(values) != num_events:
                raise ValueError(
                    f"Oracle sampling feature {name!r} has length {len(values)}; "
                    f"expected {num_events}."
                )
            kwargs[name] = torch.tensor(values, dtype=torch.float32, device=self.device)

        ts_mask = (injection_times >= float(t_start)) & (injection_times <= float(t_end))
        if not np.any(ts_mask):
            left = max(0, np.searchsorted(injection_times, float(t_start), side="right") - 1)
            right = min(injection_times.shape[0], left + 2)
            ts_mask = np.zeros_like(injection_times, dtype=bool)
            ts_mask[left:right] = True
        kwargs["time_series"] = torch.tensor(
            injection_rates[ts_mask, None],
            dtype=torch.float32,
            device=self.device,
        )
        kwargs["time_series_times"] = torch.tensor(
            injection_times[ts_mask],
            dtype=torch.float32,
            device=self.device,
        )

        return src.data.Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float64, device=self.device),
            t_start=float(t_start),
            **kwargs,
        )

    def _sample_magnitudes_fixed_b(
        self,
        *,
        batch_size: int,
        b_value: float | torch.Tensor,
        mag_max: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = torch.as_tensor(b_value, dtype=torch.float32, device=self.device)
        if torch.any(b <= 0):
            raise ValueError("b_value must be positive for Oracle magnitude sampling.")
        if b.numel() == 1:
            b = b.expand(batch_size, 1)
        else:
            b = b.reshape(batch_size, 1)
        mag_min = self.mag_completeness.to(device=self.device, dtype=torch.float32).expand_as(b)
        mag_max_value = self.sampling_mag_max if mag_max is None else mag_max
        mag_max_tensor = torch.as_tensor(mag_max_value, dtype=torch.float32, device=self.device)
        if torch.any(mag_max_tensor <= mag_min):
            raise ValueError("mag_max must be greater than mag_completeness for Oracle sampling.")
        mag_dist = dist.GutenbergRichter(b=b, mag_min=mag_min, mag_max=mag_max_tensor)
        return mag_dist.sample()

    def _build_sampling_feature_cache(
        self,
        *,
        injection_times: np.ndarray,
        injection_rates: np.ndarray,
        time_unit_minutes: float,
    ) -> dict[str, np.ndarray | float]:
        dt_inj_min = np.diff(injection_times, prepend=injection_times[0]) * time_unit_minutes
        cumulative_volume = np.cumsum(injection_rates * dt_inj_min)
        elapsed_inj = self._legacy_elapsed_injection_time(
            injection_times,
            injection_rates,
            self.oracle_sampling_activity_epsilon,
        )
        return {
            "injection_times": injection_times,
            "injection_rates": injection_rates,
            "cumulative_volume": cumulative_volume,
            "elapsed_inj": elapsed_inj,
            "time_unit_minutes": float(time_unit_minutes),
        }

    def _new_sampling_feature_state(
        self,
        feature_cache: dict[str, np.ndarray | float],
    ) -> dict[str, object]:
        return {
            "cache": feature_cache,
            "ars_history": np.full(
                self.oracle_sampling_smoothing_window,
                self.oracle_sampling_log_floor,
                dtype=np.float64,
            ),
            "prev_feature_time": 0.0,
            "prev_cumulative_volume": 0.0,
        }

    @staticmethod
    def _clone_sampling_feature_state(state: dict[str, object]) -> dict[str, object]:
        return {
            "cache": state["cache"],
            "ars_history": np.asarray(state["ars_history"], dtype=np.float64).copy(),
            "prev_feature_time": float(state["prev_feature_time"]),
            "prev_cumulative_volume": float(state["prev_cumulative_volume"]),
        }

    def _advance_sampling_feature_state(
        self,
        state: dict[str, object],
        event_time: float,
    ) -> dict[str, float]:
        cache = state["cache"]
        injection_times = cache["injection_times"]
        injection_rates = cache["injection_rates"]
        cumulative_volume = cache["cumulative_volume"]
        elapsed_inj = cache["elapsed_inj"]
        time_unit_minutes = float(cache["time_unit_minutes"])

        event_time = float(event_time)
        vm_lin = float(np.interp(event_time, injection_times, injection_rates))
        vm_cum = float(np.interp(event_time, injection_times, cumulative_volume))
        dvm = vm_cum - float(state["prev_cumulative_volume"])
        state["prev_cumulative_volume"] = vm_cum

        elapsed_event = float(np.interp(event_time, injection_times, elapsed_inj))
        inter_time_min = (event_time - float(state["prev_feature_time"])) * time_unit_minutes
        state["prev_feature_time"] = event_time

        with np.errstate(divide="ignore", invalid="ignore"):
            log_rate = float(np.log10(np.reciprocal(inter_time_min)))
            vm = float(np.log10(abs(vm_lin)))
            dvc = float(np.log10(abs(dvm)))
            dts = float(np.log10(elapsed_event))

        ars_history = np.asarray(state["ars_history"], dtype=np.float64)
        ars_history[:-1] = ars_history[1:]
        ars_history[-1] = log_rate
        state["ars_history"] = ars_history

        return {
            "vm": vm,
            "dVc": dvc,
            "sv": float(np.sign(dvm)),
            "dTS": dts,
            "aRs": float(ars_history.mean()),
        }

    def _expand_sampling_feature_state(
        self,
        state: dict[str, object],
        batch_size: int,
    ) -> dict[str, object]:
        return {
            "cache": state["cache"],
            "ars_history": np.repeat(
                np.asarray(state["ars_history"], dtype=np.float64)[None, :],
                int(batch_size),
                axis=0,
            ),
            "prev_feature_time": np.full(
                int(batch_size),
                float(state["prev_feature_time"]),
                dtype=np.float64,
            ),
            "prev_cumulative_volume": np.full(
                int(batch_size),
                float(state["prev_cumulative_volume"]),
                dtype=np.float64,
            ),
        }

    def _advance_sampling_feature_state_batch(
        self,
        state: dict[str, object],
        sample_indices: torch.Tensor,
        event_times: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        cache = state["cache"]
        injection_times = cache["injection_times"]
        injection_rates = cache["injection_rates"]
        cumulative_volume = cache["cumulative_volume"]
        elapsed_inj = cache["elapsed_inj"]
        time_unit_minutes = float(cache["time_unit_minutes"])

        idx_np = sample_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        event_times_np = event_times.detach().cpu().numpy().astype(np.float64, copy=False)
        vm_lin = np.interp(event_times_np, injection_times, injection_rates)
        vm_cum = np.interp(event_times_np, injection_times, cumulative_volume)

        prev_cumulative_volume = state["prev_cumulative_volume"]
        prev_feature_time = state["prev_feature_time"]
        dvm = vm_cum - prev_cumulative_volume[idx_np]
        prev_cumulative_volume[idx_np] = vm_cum

        elapsed_event = np.interp(event_times_np, injection_times, elapsed_inj)
        inter_time_min = (event_times_np - prev_feature_time[idx_np]) * time_unit_minutes
        prev_feature_time[idx_np] = event_times_np

        with np.errstate(divide="ignore", invalid="ignore"):
            log_rate = np.log10(np.reciprocal(inter_time_min))
            vm = np.log10(np.abs(vm_lin))
            dvc = np.log10(np.abs(dvm))
            dts = np.log10(elapsed_event)

        ars_history = state["ars_history"]
        history_rows = ars_history[idx_np].copy()
        history_rows[:, :-1] = history_rows[:, 1:]
        history_rows[:, -1] = log_rate
        ars_history[idx_np] = history_rows
        ars = history_rows.mean(axis=1)

        return {
            "vm": torch.tensor(vm, dtype=torch.float32, device=self.device),
            "dVc": torch.tensor(dvc, dtype=torch.float32, device=self.device),
            "sv": torch.tensor(np.sign(dvm), dtype=torch.float32, device=self.device),
            "dTS": torch.tensor(dts, dtype=torch.float32, device=self.device),
            "aRs": torch.tensor(ars, dtype=torch.float32, device=self.device),
        }

    def _build_sampling_mark_row(
        self,
        *,
        inter_time: float,
        magnitude: float,
        features: dict[str, float],
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        inter_time_tensor = torch.tensor([inter_time], dtype=torch.float32, device=self.device)
        magnitude_tensor = torch.tensor([magnitude], dtype=torch.float32, device=self.device)
        for name in self.base_mark_order:
            if name == "time":
                values.append(self.prep_time_marks(inter_time_tensor))
            elif name == "aRs":
                mark = torch.tensor(
                    [features["aRs"]],
                    dtype=torch.float32,
                    device=self.device,
                )
                values.append(self.prep_aux_marks(mark))
            elif name == "mag":
                values.append(self.prep_mag_marks(magnitude_tensor))
            elif name == "Mc":
                values.append(self.prep_mag_marks(self.mag_completeness.reshape(1)))
            elif name in {"vm", "dVc", "sv", "dTS"}:
                mark = torch.tensor(
                    [features[name]],
                    dtype=torch.float32,
                    device=self.device,
                )
                values.append(self.prep_aux_marks(mark))
            else:
                raise ValueError(
                    f"Unsupported base mark {name!r}. Supported marks: {self.BASE_MARK_ORDER}."
                )
        if self.supplementary_mark_list:
            raise NotImplementedError(
                "Oracle fast sampling does not support supplementary_mark_list. "
                "Use base Oracle marks only for sampling."
            )
        return torch.cat(values, dim=-1).unsqueeze(0)

    def _build_sampling_mark_rows(
        self,
        *,
        inter_times: torch.Tensor,
        magnitudes: torch.Tensor,
        features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        values: list[torch.Tensor] = []
        inter_times = inter_times.to(device=self.device, dtype=torch.float32).reshape(-1)
        magnitudes = magnitudes.to(device=self.device, dtype=torch.float32).reshape(-1)
        for name in self.base_mark_order:
            if name == "time":
                values.append(self.prep_time_marks(inter_times))
            elif name == "aRs":
                values.append(self.prep_aux_marks(features["aRs"].reshape(-1)))
            elif name == "mag":
                values.append(self.prep_mag_marks(magnitudes))
            elif name == "Mc":
                values.append(self.prep_mag_marks(self.mag_completeness.expand_as(magnitudes)))
            elif name in {"vm", "dVc", "sv", "dTS"}:
                values.append(self.prep_aux_marks(features[name].reshape(-1)))
            else:
                raise ValueError(
                    f"Unsupported base mark {name!r}. Supported marks: {self.BASE_MARK_ORDER}."
                )
        if self.supplementary_mark_list:
            raise NotImplementedError(
                "Oracle fast sampling does not support supplementary_mark_list. "
                "Use base Oracle marks only for sampling."
            )
        return torch.cat(values, dim=-1).unsqueeze(1)

    def _sampling_zero_pre_params(self) -> torch.Tensor:
        return torch.zeros(
            1,
            3 * self.num_dist_components,
            dtype=torch.float32,
            device=self.device,
        )

    def _decode_last_sampling_context(self, decoder_rows: torch.Tensor) -> torch.Tensor:
        if decoder_rows.numel() == 0:
            return self._sampling_zero_pre_params()
        keep = int(getattr(self.decoder, "lookback_size", 1)) + 1
        x = decoder_rows[-keep:, :].unsqueeze(0)
        return self.decoder(x, forecasting=False)[:, -1, :]

    def _decode_last_sampling_context_batch(
        self,
        decoder_rows: torch.Tensor,
        decoder_counts: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.zeros(
            decoder_rows.shape[0],
            3 * self.num_dist_components,
            dtype=torch.float32,
            device=self.device,
        )
        unique_counts = torch.unique(decoder_counts.detach().cpu()).tolist()
        for count_value in unique_counts:
            count = int(count_value)
            if count <= 0:
                continue
            mask = decoder_counts == count
            if not torch.any(mask):
                continue
            x = decoder_rows[mask, -count:, :]
            out[mask] = self.decoder(x, forecasting=False)[:, -1, :]
        return out

    @staticmethod
    def _clone_sampling_hidden(hidden):
        if hidden is None:
            return None
        if torch.is_tensor(hidden):
            return hidden.clone()
        return tuple(item.clone() for item in hidden)

    @staticmethod
    def _expand_sampling_hidden(hidden, batch_size: int):
        if hidden is None:
            return None
        if torch.is_tensor(hidden):
            return hidden.expand(-1, int(batch_size), -1).contiguous()
        return tuple(item.expand(-1, int(batch_size), -1).contiguous() for item in hidden)

    @staticmethod
    def _select_sampling_hidden(hidden, sample_indices: torch.Tensor):
        if hidden is None:
            return None
        if torch.is_tensor(hidden):
            return hidden.index_select(1, sample_indices)
        return tuple(item.index_select(1, sample_indices) for item in hidden)

    @staticmethod
    def _merge_sampling_hidden(
        hidden,
        sample_indices: torch.Tensor,
        updated_hidden,
        batch_size: int,
    ):
        if updated_hidden is None:
            return hidden
        if torch.is_tensor(updated_hidden):
            if hidden is None:
                hidden = updated_hidden.new_zeros(
                    updated_hidden.shape[0],
                    int(batch_size),
                    updated_hidden.shape[2],
                )
            else:
                hidden = hidden.clone()
            hidden[:, sample_indices, :] = updated_hidden
            return hidden

        if hidden is None:
            hidden_items = [
                item.new_zeros(item.shape[0], int(batch_size), item.shape[2])
                for item in updated_hidden
            ]
        else:
            hidden_items = [item.clone() for item in hidden]
        for item, updated_item in zip(hidden_items, updated_hidden):
            item[:, sample_indices, :] = updated_item
        return tuple(hidden_items)

    def _expand_sampling_decoder_rows(
        self,
        decoder_rows: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keep = int(getattr(self.decoder, "lookback_size", 1)) + 1
        out = torch.zeros(
            int(batch_size),
            keep,
            self.context_size,
            dtype=torch.float32,
            device=self.device,
        )
        count = min(int(decoder_rows.shape[0]), keep)
        if count > 0:
            out[:, -count:, :] = decoder_rows[-count:, :].unsqueeze(0).expand(batch_size, -1, -1)
        counts = torch.full(
            (int(batch_size),),
            count,
            dtype=torch.long,
            device=self.device,
        )
        return out, counts

    def _append_sampling_mark_row(
        self,
        *,
        mark_row: torch.Tensor,
        hidden,
        decoder_rows: torch.Tensor,
    ) -> tuple[object, torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            decoder_row = mark_row
        else:
            history, hidden = self.encoder(mark_row, hidden)
            decoder_row = torch.cat([mark_row, history], dim=-1)

        row = decoder_row.squeeze(0).squeeze(0)
        if decoder_rows.numel() == 0:
            decoder_rows = row.unsqueeze(0)
        else:
            decoder_rows = torch.cat([decoder_rows, row.unsqueeze(0)], dim=0)

        keep = int(getattr(self.decoder, "lookback_size", 1)) + 1
        decoder_rows = decoder_rows[-keep:, :]
        pre_params = self._decode_last_sampling_context(decoder_rows)
        return hidden, decoder_rows, pre_params

    def _append_sampling_mark_rows(
        self,
        *,
        mark_rows: torch.Tensor,
        hidden,
        decoder_rows: torch.Tensor,
        decoder_counts: torch.Tensor,
        sample_indices: torch.Tensor,
        batch_size: int,
    ) -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.encoder is None:
            decoder_row = mark_rows
        else:
            selected_hidden = self._select_sampling_hidden(hidden, sample_indices)
            history, updated_hidden = self.encoder(mark_rows, selected_hidden)
            hidden = self._merge_sampling_hidden(
                hidden,
                sample_indices,
                updated_hidden,
                batch_size,
            )
            decoder_row = torch.cat([mark_rows, history], dim=-1)

        row = decoder_row[:, 0, :]
        keep = int(getattr(self.decoder, "lookback_size", 1)) + 1
        old_counts = decoder_counts.index_select(0, sample_indices)
        full_mask = old_counts >= keep
        if torch.any(full_mask):
            full_indices = sample_indices[full_mask]
            decoder_rows[full_indices, :-1, :] = decoder_rows[full_indices, 1:, :].clone()
            decoder_rows[full_indices, -1, :] = row[full_mask]

        if torch.any(~full_mask):
            partial_positions = torch.nonzero(~full_mask, as_tuple=False).flatten()
            for position in partial_positions.tolist():
                sample_idx = int(sample_indices[position].item())
                count = int(old_counts[position].item())
                start = keep - count
                if count > 0:
                    decoder_rows[sample_idx, start - 1 : -1, :] = decoder_rows[
                        sample_idx, start:, :
                    ].clone()
                decoder_rows[sample_idx, -1, :] = row[position]

        decoder_counts[sample_indices] = (old_counts + 1).clamp_max(keep)
        pre_params = self._decode_last_sampling_context_batch(
            decoder_rows.index_select(0, sample_indices),
            decoder_counts.index_select(0, sample_indices),
        )
        return hidden, decoder_rows, decoder_counts, pre_params, sample_indices

    def _build_initial_sampling_state(
        self,
        *,
        event_times: list[float],
        magnitudes: list[float],
        context_t_start: float,
        feature_cache: dict[str, np.ndarray | float],
    ) -> tuple[dict[str, object], object, torch.Tensor, torch.Tensor, float]:
        feature_state = self._new_sampling_feature_state(feature_cache)
        hidden = None
        decoder_rows = torch.empty(0, self.context_size, dtype=torch.float32, device=self.device)
        pre_params = self._sampling_zero_pre_params()
        last_event_time = float(context_t_start)

        for event_time, magnitude in zip(event_times, magnitudes):
            inter_time = max(float(event_time) - last_event_time, self._INTER_TIME_MIN)
            features = self._advance_sampling_feature_state(feature_state, float(event_time))
            mark_row = self._build_sampling_mark_row(
                inter_time=inter_time,
                magnitude=float(magnitude),
                features=features,
            )
            hidden, decoder_rows, pre_params = self._append_sampling_mark_row(
                mark_row=mark_row,
                hidden=hidden,
                decoder_rows=decoder_rows,
            )
            last_event_time = float(event_time)

        return feature_state, hidden, decoder_rows, pre_params, last_event_time

    def _build_mark_tensors(self, batch: src.data.Batch) -> list[torch.Tensor]:
        marks: list[torch.Tensor] = []
        for name in self.base_mark_order:
            if name == "time":
                marks.append(self.prep_time_marks(batch.inter_times))
            elif name == "aRs":
                marks.append(self.prep_aux_marks(batch["aRs"]))
            elif name == "mag":
                marks.append(self.prep_mag_marks(batch.mag))
            elif name == "Mc":
                marks.append(self.prep_mag_marks(self.mag_completeness.expand_as(batch.mag)))
            elif name == "vm":
                marks.append(self.prep_aux_marks(batch.vm))
            elif name == "dVc":
                marks.append(self.prep_aux_marks(batch.dVc))
            elif name == "sv":
                marks.append(self.prep_aux_marks(batch.sv))
            elif name == "dTS":
                marks.append(self.prep_aux_marks(batch.dTS))
            else:
                raise ValueError(
                    f"Unsupported base mark {name!r}. Supported marks: {self.BASE_MARK_ORDER}."
                )
        marks.extend(self.prep_aux_marks(batch[name]) for name in self.supplementary_mark_list)
        return marks

    def _require_batch_marks(self, batch: src.data.Batch) -> None:
        required_event_marks = [key for key in self.REQUIRED_EVENT_MARKS if key in self.base_mark_index]
        missing = [key for key in required_event_marks if key not in batch]
        if missing:
            raise ValueError(
                "Oracle requires event-level marks generated by the Oracle feature builder. "
                f"Missing keys: {missing}."
            )
        missing_supp = [key for key in self.supplementary_mark_list if key not in batch]
        if missing_supp:
            raise ValueError(
                f"Oracle supplementary marks are missing in batch: {missing_supp}."
            )

    def get_marks(self, batch: src.data.Batch) -> torch.Tensor:
        self._require_batch_marks(batch)
        marks = torch.cat(self._build_mark_tensors(batch), dim=-1)
        if "input_mask" in batch:
            marks = marks * batch.input_mask.unsqueeze(-1)
        return marks

    def get_pre_params(
        self,
        marks: torch.Tensor,
        *,
        forecasting: bool = False,
        idx_split: int | None = None,
    ) -> torch.Tensor:
        if forecasting:
            if idx_split is None:
                raise ValueError("idx_split must be provided when forecasting=True.")

        if self.encoder is None:
            decoder_input = marks
        else:
            history, _ = self.encoder(marks)
            decoder_input = torch.cat([marks, history], dim=-1)

        if forecasting:
            return self.decoder(decoder_input, forecasting=True, idx=idx_split)
        return self.decoder(decoder_input, forecasting=False, idx=0)

    def get_inter_time_dist(self, pre_params: torch.Tensor):
        return self.dist_decoder(pre_params)

    @staticmethod
    def _causal_shift_pre_params(pre_params: torch.Tensor) -> torch.Tensor:
        """Shift decoded parameters right by one step for causal likelihood.

        For step i, the time distribution must be conditioned on information up to i-1.
        This matches the recurrent TPP convention and avoids using same-step inter-time
        marks to predict themselves.
        """
        first = torch.zeros_like(pre_params[:, :1, :])
        return torch.cat([first, pre_params[:, :-1, :]], dim=1)

    def _causal_nll_dict(self, batch: src.data.Batch) -> dict[str, torch.Tensor]:
        marks = self.get_marks(batch)
        pre_params_raw = self.get_pre_params(marks)
        pre_params = self._causal_shift_pre_params(pre_params_raw)
        d_t_obs = batch.inter_times.clamp(self._INTER_TIME_MIN, self._INTER_TIME_MAX)
        inter_time_dist = self.get_inter_time_dist(pre_params)
        log_like = self.time_log_likelihood(
            batch=batch,
            inter_time_dist=inter_time_dist,
            state=pre_params,
            dist_from_state=self.get_inter_time_dist,
            pdf_inter_times=d_t_obs,
            survival_inter_times=d_t_obs,
        )
        nll_time = -log_like
        return {"time": nll_time, "total": nll_time}

    def _legacy_prev_log_survival(
        self,
        *,
        batch: src.data.Batch,
        pre_params: torch.Tensor,
        d_t_obs: torch.Tensor,
    ) -> torch.Tensor:
        arange = torch.arange(batch.batch_size, device=pre_params.device)
        prev_surv_dist = self.get_inter_time_dist(pre_params[arange, batch.start_idx, :])
        prev_surv_time = d_t_obs[arange, batch.start_idx] - (
            batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
        )
        return prev_surv_dist.log_survival(prev_surv_time).reshape(-1)

    def _legacy_base_log_like(
        self,
        *,
        batch: src.data.Batch,
        pre_params: torch.Tensor,
        include_prev_survival: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = batch.nll_event_mask[:, 1:-1]
        d_t_obs = batch.inter_times.clamp(self._INTER_TIME_MIN, self._INTER_TIME_MAX)
        inter_time_dist = self.get_inter_time_dist(pre_params[:, 0:-2, :])
        log_pdf = inter_time_dist.log_prob(d_t_obs[:, 1:-1])
        log_like = (log_pdf * mask).sum(-1)

        arange = torch.arange(batch.batch_size, device=pre_params.device)
        last_surv_dist = self.get_inter_time_dist(pre_params[arange, batch.end_idx, :])
        last_log_surv = last_surv_dist.log_survival(d_t_obs[arange, batch.end_idx])
        log_like = log_like + last_log_surv.reshape(-1)

        if include_prev_survival and torch.any(batch.t_nll_start != batch.t_start):
            log_like = log_like - self._legacy_prev_log_survival(
                batch=batch,
                pre_params=pre_params,
                d_t_obs=d_t_obs,
            )

        return log_like, log_pdf, d_t_obs

    def _legacy_fit_nll_dict(self, batch: src.data.Batch) -> dict[str, torch.Tensor]:
        marks = self.get_marks(batch)
        pre_params = self.get_pre_params(marks)
        log_like, _, _ = self._legacy_base_log_like(
            batch=batch,
            pre_params=pre_params,
            include_prev_survival=True,
        )
        nll_time = -log_like
        return {"time": nll_time, "total": nll_time}

    def _legacy_forecast_indices(
        self,
        *,
        seq_len: int,
        forecast_count: int,
        device: torch.device,
    ) -> list[int]:
        if seq_len <= 2:
            return []
        candidates = torch.arange(1, seq_len - 1, device=device)
        if candidates.numel() == 0:
            return []
        if forecast_count == 0 or forecast_count >= int(candidates.numel()):
            selected = candidates
        else:
            order = torch.randperm(candidates.numel(), device=device)
            selected = candidates[order[:forecast_count]]
        return [int(value.item()) for value in selected]

    def _legacy_forecast_nll_dict(
        self,
        batch: src.data.Batch,
        *,
        forecast_count: int | None = None,
    ) -> dict[str, torch.Tensor]:
        marks = self.get_marks(batch)
        pre_params = self.get_pre_params(marks)
        log_like, log_pdf, d_t_obs = self._legacy_base_log_like(
            batch=batch,
            pre_params=pre_params,
            include_prev_survival=False,
        )
        mask = batch.nll_event_mask[:, 1:-1]
        forecast_count = (
            self.oracle_forecast_count
            if forecast_count is None
            else int(forecast_count)
        )
        if forecast_count < 0:
            raise ValueError("forecast_count must be >= 0.")

        idx_list = self._legacy_forecast_indices(
            seq_len=batch.seq_len,
            forecast_count=forecast_count,
            device=marks.device,
        )
        arange = torch.arange(batch.batch_size, device=pre_params.device)
        for idx_split in idx_list:
            pre_params_f = self.get_pre_params(
                marks,
                forecasting=True,
                idx_split=idx_split,
            )
            forecast_dist = self.get_inter_time_dist(pre_params_f[:, 0:-2, :])
            forecast_log_pdf = forecast_dist.log_prob(d_t_obs[:, idx_split:-2])
            combined_log_pdf = torch.cat(
                [log_pdf[:, 0:idx_split], forecast_log_pdf],
                dim=-1,
            )
            log_like_f = (combined_log_pdf * mask).sum(-1)

            pre_params_f_full = torch.cat(
                [pre_params[:, 0:idx_split, :], pre_params_f],
                dim=1,
            )
            last_surv_dist_f = self.get_inter_time_dist(
                pre_params_f_full[arange, batch.end_idx, :]
            )
            last_log_surv_f = last_surv_dist_f.log_survival(
                d_t_obs[arange, batch.end_idx]
            )
            log_like = log_like + log_like_f + last_log_surv_f.reshape(-1)

        log_like = log_like / float(len(idx_list) + 1)
        if torch.any(batch.t_nll_start != batch.t_start):
            log_like = log_like - self._legacy_prev_log_survival(
                batch=batch,
                pre_params=pre_params,
                d_t_obs=d_t_obs,
            )

        nll_time = -log_like
        return {"time": nll_time, "total": nll_time}

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 1e-10,
        loss_mode: str | None = None,
        forecast_count: int | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch = batch.to(self.device)
        mode = (
            self.oracle_loss_mode
            if loss_mode is None
            else self._normalize_loss_mode(loss_mode)
        )
        if mode == "causal":
            out = self._causal_nll_dict(batch)
        elif mode == "legacy_fit":
            out = self._legacy_fit_nll_dict(batch)
        elif mode == "legacy_forecast":
            out = self._legacy_forecast_nll_dict(batch, forecast_count=forecast_count)
        else:
            raise ValueError(f"Unsupported oracle_loss_mode={mode!r}.")

        reduction = self.reduction if reduction is None else reduction
        out = self.reduce_nll_dict(out, batch, reduction=reduction, eps=eps)
        if return_dict:
            return out
        return out["total"]

    def _next_inter_time_dist_for_sampling(
        self,
        *,
        event_times: list[float],
        magnitudes: list[float],
        context_t_start: float,
        context_t_end: float,
        injection_times: np.ndarray,
        injection_rates: np.ndarray,
        time_unit_minutes: float,
    ):
        context_seq = self._build_sampling_sequence(
            event_times=event_times,
            magnitudes=magnitudes,
            t_start=context_t_start,
            t_end=context_t_end,
            injection_times=injection_times,
            injection_rates=injection_rates,
            time_unit_minutes=time_unit_minutes,
            include_time_series=False,
        )
        batch = src.data.Batch.from_list([context_seq]).to(self.device)
        marks = self.get_marks(batch)
        pre_params_raw = self.get_pre_params(marks)
        pre_params = self._causal_shift_pre_params(pre_params_raw)
        end_idx = int(batch.end_idx[0].item())
        next_context = pre_params[0, end_idx, :].unsqueeze(0)
        return self.get_inter_time_dist(next_context)

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: src.data.Sequence | None = None,
        return_sequences: bool = False,
        *,
        fixed_b: float | torch.Tensor | None = None,
        b_value: float | torch.Tensor | None = None,
        mag_max: float | torch.Tensor | None = None,
        injection_times=None,
        injection_rates=None,
        bg_cache_seq: src.data.Sequence | None = None,
        time_unit_minutes: float | None = None,
        max_events: int = 50_000,
        max_length: int | None = None,
        max_sample_len: int | None = None,
        predict_b: bool | None = None,
    ) -> src.data.Batch | list[src.data.Sequence]:
        """Causally sample future events with online Oracle feature construction.

        Injection covariates are treated as known exogenous inputs. Event-derived
        marks (``aRs``, ``dVc`` and ``sv``) are recomputed from the generated
        event history, while magnitudes are sampled from a fixed-b
        Gutenberg-Richter distribution.
        """
        del predict_b
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if duration < 0:
            raise ValueError("duration must be non-negative.")

        if max_length is not None:
            max_events = int(max_length)
        if max_sample_len is not None:
            max_events = int(max_sample_len)
        if max_events < 0:
            raise ValueError("max_events must be non-negative.")

        if b_value is not None and fixed_b is not None:
            raise ValueError("Pass only one of fixed_b or b_value.")
        b_for_sampling = self.sampling_mag_b if fixed_b is None and b_value is None else (
            fixed_b if fixed_b is not None else b_value
        )
        b_tensor = torch.as_tensor(b_for_sampling, dtype=torch.float32, device=self.device).reshape(-1)
        if torch.any(b_tensor <= 0):
            raise ValueError("fixed_b/b_value must be positive.")
        if b_tensor.numel() == 1:
            b_tensor = b_tensor.expand(batch_size)
        elif b_tensor.numel() != batch_size:
            raise ValueError(
                f"fixed_b/b_value must be scalar or have length batch_size={batch_size} "
                f"(got {b_tensor.numel()})."
            )

        injection_times_np, injection_rates_np = self._resolve_sampling_injection_series(
            past_seq=past_seq,
            bg_cache_seq=bg_cache_seq,
            injection_times=injection_times,
            injection_rates=injection_rates,
        )
        time_unit_minutes = (
            self.oracle_sampling_time_unit_minutes
            if time_unit_minutes is None
            else float(time_unit_minutes)
        )
        if time_unit_minutes <= 0:
            raise ValueError("time_unit_minutes must be > 0.")

        if past_seq is not None:
            forecast_start = float(past_seq.t_end)
            context_t_start = float(past_seq.t_start)
            past_event_times_tensor = past_seq.arrival_times.detach().cpu()
            time_tolerance = max(1e-9, 1e-6 * max(1.0, abs(forecast_start)))
            keep_mask = past_event_times_tensor <= forecast_start + time_tolerance
            past_event_times = [float(v) for v in past_event_times_tensor[keep_mask].tolist()]
            if "mag" in past_seq:
                past_mags_tensor = torch.as_tensor(past_seq.mag).detach().cpu()[keep_mask]
                past_magnitudes = [float(v) for v in past_mags_tensor.tolist()]
            else:
                past_magnitudes = [float(self.mag_completeness.detach().cpu().item())] * len(past_event_times)
        else:
            forecast_start = float(t_start)
            context_t_start = float(t_start)
            past_event_times = []
            past_magnitudes = []

        if past_event_times:
            past_event_times = [
                float(v)
                for v in self._strictly_increasing_event_times(
                    past_event_times,
                    t_start=context_t_start,
                    t_end=forecast_start,
                ).tolist()
            ]

        forecast_end = forecast_start + float(duration)
        sequences: list[src.data.Sequence] = []

        with torch.inference_mode():
            feature_cache = self._build_sampling_feature_cache(
                injection_times=injection_times_np,
                injection_rates=injection_rates_np,
                time_unit_minutes=time_unit_minutes,
            )
            (
                base_feature_state,
                base_hidden,
                base_decoder_rows,
                base_pre_params,
                base_last_event_time,
            ) = self._build_initial_sampling_state(
                event_times=past_event_times,
                magnitudes=past_magnitudes,
                context_t_start=context_t_start,
                feature_cache=feature_cache,
            )

            feature_state = self._expand_sampling_feature_state(base_feature_state, batch_size)
            hidden = self._expand_sampling_hidden(base_hidden, batch_size)
            decoder_rows, decoder_counts = self._expand_sampling_decoder_rows(
                base_decoder_rows,
                batch_size,
            )
            current_pre_params = base_pre_params.expand(batch_size, -1).clone()
            current_times = torch.full(
                (batch_size,),
                float(forecast_start),
                dtype=torch.float64,
                device=self.device,
            )
            last_event_times = torch.full(
                (batch_size,),
                float(base_last_event_time),
                dtype=torch.float64,
                device=self.device,
            )
            lower_bounds = (current_times - last_event_times).clamp_min(0.0)
            active = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            event_counts = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            generated_times_by_sample: list[list[float]] = [[] for _ in range(batch_size)]
            generated_magnitudes_by_sample: list[list[float]] = [[] for _ in range(batch_size)]

            while True:
                active_indices = torch.nonzero(
                    active & (event_counts < max_events),
                    as_tuple=False,
                ).flatten()
                if active_indices.numel() == 0:
                    break

                next_waits = torch.empty(
                    active_indices.numel(),
                    dtype=torch.float64,
                    device=self.device,
                )
                active_lower_bounds = lower_bounds.index_select(0, active_indices)
                conditional_mask = active_lower_bounds > 0
                if torch.any(conditional_mask):
                    conditional_indices = active_indices[conditional_mask]
                    conditional_dist = self.get_inter_time_dist(
                        current_pre_params.index_select(0, conditional_indices)
                    )
                    lower = lower_bounds.index_select(0, conditional_indices).reshape(-1, 1)
                    lower_model = lower.to(dtype=current_pre_params.dtype)
                    sampled_wait_total = conditional_dist.sample_conditional(
                        lower_bound=lower_model,
                    ).reshape(-1)
                    next_waits[conditional_mask] = (
                        sampled_wait_total.to(dtype=torch.float64)
                        - lower_model.reshape(-1).to(dtype=torch.float64)
                    )

                if torch.any(~conditional_mask):
                    unconditional_indices = active_indices[~conditional_mask]
                    unconditional_dist = self.get_inter_time_dist(
                        current_pre_params.index_select(0, unconditional_indices)
                    )
                    next_waits[~conditional_mask] = (
                        unconditional_dist.sample().reshape(-1).to(dtype=torch.float64)
                    )

                if not torch.isfinite(next_waits).all():
                    raise RuntimeError("Oracle sampled a non-finite inter-event time.")
                next_waits = next_waits.clamp_min(self._INTER_TIME_MIN)
                next_event_times = current_times.index_select(0, active_indices) + next_waits

                event_mask = next_event_times <= float(forecast_end)
                if torch.any(~event_mask):
                    active[active_indices[~event_mask]] = False
                if not torch.any(event_mask):
                    continue

                event_indices = active_indices[event_mask]
                event_times = next_event_times[event_mask]
                next_magnitudes = self._sample_magnitudes_fixed_b(
                    batch_size=int(event_indices.numel()),
                    b_value=b_tensor.index_select(0, event_indices),
                    mag_max=mag_max,
                ).reshape(-1)
                event_inter_times = (
                    event_times - last_event_times.index_select(0, event_indices)
                ).clamp_min(self._INTER_TIME_MIN)
                features = self._advance_sampling_feature_state_batch(
                    feature_state,
                    event_indices,
                    event_times,
                )
                mark_rows = self._build_sampling_mark_rows(
                    inter_times=event_inter_times,
                    magnitudes=next_magnitudes,
                    features=features,
                )
                (
                    hidden,
                    decoder_rows,
                    decoder_counts,
                    next_pre_params,
                    updated_indices,
                ) = self._append_sampling_mark_rows(
                    mark_rows=mark_rows,
                    hidden=hidden,
                    decoder_rows=decoder_rows,
                    decoder_counts=decoder_counts,
                    sample_indices=event_indices,
                    batch_size=batch_size,
                )
                current_pre_params[updated_indices] = next_pre_params

                event_times_list = event_times.detach().cpu().tolist()
                magnitudes_list = next_magnitudes.detach().cpu().tolist()
                for position, sample_idx in enumerate(event_indices.detach().cpu().tolist()):
                    generated_times_by_sample[sample_idx].append(float(event_times_list[position]))
                    generated_magnitudes_by_sample[sample_idx].append(
                        float(magnitudes_list[position])
                    )

                last_event_times[event_indices] = event_times
                current_times[event_indices] = event_times
                lower_bounds[event_indices] = 0.0
                event_counts[event_indices] += 1

            for sample_idx in range(batch_size):
                generated_times = generated_times_by_sample[sample_idx]
                generated_magnitudes = generated_magnitudes_by_sample[sample_idx]
                if (
                    int(event_counts[sample_idx].item()) >= max_events
                    and generated_times
                    and generated_times[-1] < forecast_end
                ):
                    raise RuntimeError(
                        "Oracle sampling reached max_events before the forecast window ended. "
                        "Increase max_events/max_length or inspect the checkpoint for explosive sampling."
                    )
                sample_seq = self._build_sampling_sequence_from_feature_state(
                    event_times=generated_times,
                    magnitudes=generated_magnitudes,
                    initial_feature_state=base_feature_state,
                    t_start=forecast_start,
                    t_end=forecast_end,
                    injection_times=injection_times_np,
                    injection_rates=injection_rates_np,
                )
                sequences.append(sample_seq)

        if return_sequences:
            return sequences
        return src.data.Batch.from_list(sequences).to(self.device)
