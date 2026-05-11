from __future__ import annotations

import math
from typing import Iterable

import torch

import src

from .common.oracle_blocks import OracleDistDecoder, OracleFCNDecoder, OracleRNNEncoder
from .tpp_model import TPPModel


class Oracle(TPPModel):
    """Oracle baseline ported from ORACLE-main with minimal integration changes."""

    REQUIRED_EVENT_MARKS = ("aRs", "vm", "dVc", "sv", "dTS")
    BASE_MARK_ORDER = ("time", "aRs", "mag", "Mc", "vm", "dVc", "sv", "dTS")
    DEFAULT_FUTURE_FEATURE_NAMES = ("vm", "sv", "dTS", "Mc")
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
        self.to(self.device)

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

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 1e-10,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch = batch.to(self.device)
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
        out = {"time": nll_time, "total": nll_time}
        reduction = self.reduction if reduction is None else reduction
        out = self.reduce_nll_dict(out, batch, reduction=reduction, eps=eps)
        if return_dict:
            return out
        return out["total"]
