import logging
import math
from typing import Any, Dict, List, Tuple

import src.data
import torch
from src.data.constants import PAD
from src.utils.mask_utils import get_non_pad_mask

logger = logging.getLogger(__name__)


class MixerBatchAdapter:
    def __init__(self, args):
        """
        input adapter for mixer_tpp
        """
        def _to_t(x: Any) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32)

        # ---- stats & constants
        self.tau_mean = _to_t(args.get('tau_mean'))
        self.tau_min = _to_t(args.get('tau_min'))
        self.tau_max = _to_t(args.get('tau_max'))
        self.tau_q05 = _to_t(args.get('tau_q05'))
        self.tau_q025 = _to_t(args.get('tau_q025'))
        self.log_tau_mean = self.tau_mean.log()
        self.mag_mean = _to_t(args.get('mag_mean'))
        self.time_max = _to_t(args.get('time_max'))
        self.richter_b = _to_t(args.get('richter_b_mle'))
        self.mag_completeness = _to_t(args.get('mag_completeness'))
        self.mag_mean_gr =  self.mag_completeness + 1.0 / (math.log(10.0) * self.richter_b)
        self.eps: float = 1e-10

        # ---- config
        self.extra_input_keys: List[str] = getattr(args, 'extra_input_keys', ['inter_times', 'times'])
        self.features_input_keys: List[str] = sorted(getattr(args, 'features_input_keys', ['log_inter_times', 'mag']))
        self.normalize_time: bool = getattr(args, 'normalize_time_by_token', False)
        self.shift_time_to_zero: bool = getattr(args, 'shift_time_to_zero', False)

        self.time_scale_base = torch.tensor(1.0, dtype=torch.float32)
        if self.normalize_time:
            if getattr(args, 'time_scale_base', None) is not None:
                base = args.time_scale_base
                logger.info("using time scale base %s (from args.time_scale_base)", base)
            else:
                key = getattr(args, 'time_scale_base_key', 'tau_mean')
                base = args.get(key, 1.0)
                logger.info("using time scale base %s (key: %s)", base, key)
            self.time_scale_base = _to_t(base)
            logger.info("tau mean: %s", self.tau_mean)


    # =========================
    # PIPELINE
    # =========================
    def __call__(self, batch: 'src.data.Batch') -> Dict[str, torch.Tensor]:
        arrival_times, inter_times, mag, loc = self._extract_fields(batch)
        if not hasattr(batch, 'input_mask'):
            input_mask = self._make_non_pad_mask(inter_times).squeeze(-1).float()
        else:
            input_mask = batch.input_mask.float()
        features = self._build_features(mag, loc, inter_times, input_mask.unsqueeze(-1))
        extras = self._build_extras(arrival_times, inter_times, input_mask.unsqueeze(-1))
        return {"features": features, "input_mask": input_mask, **extras}

    # ---------- step 1: parse ----------
    def _extract_fields(self, batch: 'src.data.Batch') -> Tuple[torch.Tensor, ...]:
        """Extract relevant fields from the batch object."""
        arrival_times = batch.arrival_times
        inter_times = batch.inter_times
        mag = batch.mag
        loc = getattr(batch, 'loc', None) # loc might be optional
        return arrival_times, inter_times, mag, loc

    # ---------- step 2: mask ----------
    def _make_non_pad_mask(self, inter_times: torch.Tensor) -> torch.Tensor:
        return get_non_pad_mask(inter_times)  

    # ---------- step 3: features ----------
    def _build_features(
        self,
        mag: torch.Tensor,
        loc: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        feature_map = {
            "log_inter_times": self.normalize_log_inter_times(inter_times),
            "mag": self.normalize_magnitude(mag),
            "loc": loc.unsqueeze(-1) if loc is not None and "loc" in self.features_input_keys else None
        }
        parts: List[torch.Tensor] = []
        for key in self.features_input_keys:
            if key in feature_map and feature_map[key] is not None:
                parts.append(feature_map[key])
            else:
                raise ValueError(f"Unsupported or missing feature key: {key}")

        features = torch.cat(parts, dim=-1)
        return features * non_pad_mask  # Apply mask at the end

    # ---------- step 4: extras ----------
    def _build_extras(
        self,
        arrival_times: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        
        if "times" in self.extra_input_keys:
            out["times"] = self.normalize_arrival_times(arrival_times, self.normalize_time) * non_pad_mask.squeeze(-1)
        if "inter_times" in self.extra_input_keys:
            out["inter_times"] = self.normalize_inter_times(inter_times) * non_pad_mask.squeeze(-1)
        return out

    # =========================
    # NORMALIZERS
    # =========================
    def normalize_log_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        log_tau = torch.log(torch.clamp_min(inter_times, self.eps)).unsqueeze(-1)
        return log_tau - self.log_tau_mean.to(log_tau.device)

    # def normalize_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
    #     return mag.unsqueeze(-1) - self.mag_mean.to(mag.device)

    def normalize_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        device = mag.device
        dtype = mag.dtype
        mag_mean_gr = self.mag_mean_gr.to(device=device, dtype=dtype)
        b = self.richter_b.to(device=device, dtype=dtype)
        return (mag.unsqueeze(-1) - mag_mean_gr) * b


    def normalize_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        return (inter_times - self.tau_min.to(inter_times.device)) / (self.tau_max.to(inter_times.device) - self.tau_min.to(inter_times.device) + self.eps)

    def normalize_arrival_times(self, arrival_times: torch.Tensor, normalize_time: bool = False) -> torch.Tensor:
        device = arrival_times.device
        times = arrival_times

        if self.shift_time_to_zero:
            valid_mask = times.ne(PAD)
            first_idx = torch.argmax(valid_mask.int(), dim=1)  # [B]
            offsets = times.gather(1, first_idx.unsqueeze(1)).squeeze(1)  # [B]
            times = times - offsets.unsqueeze(1)

        if normalize_time:
            return times / self.time_scale_base.to(device)
        else:
            return times

