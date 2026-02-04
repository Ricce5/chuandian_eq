import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import src.data
import math
from src.utils.mask_utils import get_non_pad_mask
from src.data.constants import PAD

logger = logging.getLogger(__name__)

class SpatialMagnitudeTimeAdapter:
    """Adapter: spatial+magnitude as mark, plus event time."""
    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),
            "event_time": bx[:, :, 1],
        }


class LocationMagnitudeTimeAdapter:
    """Adapter: separate location, magnitude and time tensors."""
    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_loc": bx[:, :, 3:5],
            "event_mag": bx[:, :, 2:3],
            "event_time": bx[:, :, 1],
        }

    
class SpatialMagnitudeTimeAdapterWithAccessor:
    """Same as `SpatialMagnitudeTimeAdapter` but exposes `get_extra_inputs` for time."""
    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": torch.cat((bx[:, :, 3:5], bx[:, :, 2:3]), dim=-1),
            "event_time": bx[:, :, 1],
        }

    def get_extra_inputs(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_time": bx[:, :, 1]}



class SpatialMagnitudeTimeBatchAdapter:
    """Adapter for batch objects exposing `loc`, `mag`, `arrival_times`."""
    def __call__(self, batch: src.data.Batch) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": torch.cat([batch.loc, batch.mag[..., None]], dim=-1),
            "event_time": batch.arrival_times,
        }

class TypeTimeBatchAdapter:
    """Adapter for categorical event types + times in batch objects."""
    def __call__(self, batch: src.data.Batch) -> Dict[str, torch.Tensor]:
        return {
            "event_type": batch.type_seq,
            "event_time": batch.arrival_times,
        }
        
class MagnitudeTimeAdapter:
    """Adapter: magnitude-only mark plus time."""
    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "event_mark": bx[:, :, 2:3],
            "event_time": bx[:, :, 1],
        }
    
class MagnitudeTimeAdapterWithAccessor:
    """Magnitude-only adapter with extra time accessor."""
    def __call__(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_mark": bx[:, :, 2:3], "event_time": bx[:, :, 1]}

    def get_extra_inputs(self, bx: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_time": bx[:, :, 1]}
    
class THPBatchAdapter:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model 

    def __call__(self, bx: src.data.Batch) -> dict:
        log_inter_times = torch.log(torch.clamp_min(bx.inter_times, 1e-10)).unsqueeze(-1)
        log_inter_times -= self.model.log_tau_mean 

        mag = bx.mag[:, :, None] - self.model.mag_mean 
        # mark = torch.cat([log_inter_times, mag], dim=-1)
        mark = mag 

        return {
            "event_mark": mark * bx.input_mask[:, :, None],
            "event_time": bx.arrival_times * bx.input_mask,
            "input_mask": bx.input_mask.float(),
        }
    
class THPLogDeltaTBatchAdapter:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model

    def __call__(self, bx: src.data.Batch) -> dict:
        log_inter_times = torch.log(torch.clamp_min(bx.inter_times, 1e-10)).unsqueeze(-1)
        log_inter_times -= self.model.log_tau_mean

        mag = bx.mag[:, :, None] - self.model.mag_mean 
        # mark = torch.cat([log_inter_times, mag], dim=-1)
        mark = mag 

        return {
            "event_mark": mark * bx.input_mask[:, :, None],
            "event_time": bx.arrival_times * bx.input_mask,
            "log_inter_time": log_inter_times * bx.input_mask[:, :, None],
            "input_mask": bx.input_mask.float(),
        }

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




class MixerAdapter:
    """
    input adapter of mixer for classification and regression tasks
    """
    def __init__(self, args, revin_layer: Optional[nn.Module] = None):
        stats = args.stats
        print("Initializing MixerAdapter with stats:", stats)
        to_t = lambda x: torch.tensor(x, dtype=torch.float32)

        # ---- stats & constants
        self.tau_mean        = to_t(stats['tau_mean'])
        self.tau_min         = to_t(stats['tau_min'])
        self.tau_max         = to_t(stats['tau_max'])
        self.tau_q025        = to_t(stats['tau_q025'])
        self.tau_q05         = to_t(stats['tau_q05'])
        self.tau_unfiltered  = to_t(stats['tau_unfiltered'])
        self.mag_mean =     to_t(stats['mag_mean']) 
        self.richter_b      = to_t(stats['b_value'])
        self.mag_completeness = to_t(stats['mag_completeness'])
        self.log_tau_mean    = self.tau_mean.log()
        self.eps: float      = 1e-10
        self.mag_mean_gr =  self.mag_completeness + 1.0 / (math.log(10.0) * self.richter_b)
        self.use_mag_cdf = getattr(args, "use_mag_cdf", False)
        if self.use_mag_cdf:
            logger.info("Using magnitude CDF normalization in MixerAdapter.")

        # ---- config
        self.extra_input_keys: List[str]    = getattr(args, 'extra_input_keys', ['inter_times', 'times'])
        self.features_input_keys: List[str] = sorted(getattr(args, 'features_input_keys', ['mag']))
        self.Twindow: float                 = getattr(args, 'Twindow', None)
        self.normalize_time: bool           = getattr(args, 'normalize_time_by_token', False)
        # Optional RevIN layer for magnitude normalization
        self.revin_layer = revin_layer

        # ---- time-scale base
        self.time_scale_base = torch.tensor(1.0, dtype=torch.float32)
        if self.normalize_time:
            if getattr(args, 'time_scale_base', None) is not None:
                base = args.time_scale_base
                logger.info("using time scale base %s (from args.time_scale_base)", base)
            else:
                key = getattr(args, 'time_scale_base_key', 'tau_unfiltered')
                base = stats.get(key, 1.0)
                logger.info("using time scale base %s (key: %s)", base, key)
            self.time_scale_base = to_t(base)
            logger.info("tau mean: %s", self.tau_mean)

    # =========================
    # PIPELINE
    # =========================
    def __call__(self, batch_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        arrival_times, arrival_times_nl, mag, loc, inter_times = self._extract_fields(batch_tensor)
        non_pad_mask = self._make_non_pad_mask(arrival_times)
        features = self._build_features(mag, loc, inter_times, non_pad_mask)
        extras = self._build_extras(arrival_times_nl, inter_times, non_pad_mask)
        return {"features": features, "non_pad_mask": non_pad_mask, **extras}

    # ---------- step 1: parse ----------
    def _extract_fields(self, batch_tensor: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Expected input format:
        [:, :, 0] = arrival_times
        [:, :, 1] = arrival_times_nl
        [:, :, 2:3] = mag
        [:, :, 3:5] = loc
        [:, :, -1]  = inter_times
        """
        arrival_times     = batch_tensor[:, :, 0]
        arrival_times_nl  = batch_tensor[:, :, 1]
        mag               = batch_tensor[:, :, 2:3]
        loc               = batch_tensor[:, :, 3:5]
        inter_times       = batch_tensor[:, :, -1]
        return arrival_times, arrival_times_nl, mag, loc, inter_times

    # ---------- step 2: mask ----------
    def _make_non_pad_mask(self, arrival_times: torch.Tensor) -> torch.Tensor:
        return get_non_pad_mask(arrival_times)  # [B, T, 1]

    # ---------- step 3: features ----------
    def _build_features(
        self,
        mag: torch.Tensor,
        loc: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> torch.Tensor:
        # Apply RevIN to magnitude if available, using row mask
        if self.revin_layer is not None:
            # Ensure RevIN module parameters are on the same device/dtype as inputs
            self.revin_layer = self.revin_layer.to(device=mag.device, dtype=mag.dtype)
            mask = non_pad_mask.squeeze(-1) if non_pad_mask.ndim == 3 else non_pad_mask
            mag_norm = self.revin_layer(mag, mode='norm', mask=mask)
        else:
            mag_norm = self.normalize_magnitude(mag)  # [B, T, 1]

        feature_map = {
            "mag": mag_norm,
            "log_inter_times": self.normalize_log_inter_times(inter_times),  # [B, T, 1]
            "loc": loc                                            # [B, T, 2]
        }
        parts = [feature_map[k] for k in self.features_input_keys]
        features = torch.cat(parts, dim=-1) * non_pad_mask
        return features  # [B, T, D]

    # ---------- step 4: extras ----------
    def _build_extras(
        self,
        arrival_times_nl: torch.Tensor,
        inter_times: torch.Tensor,
        non_pad_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}

        if "times" in self.extra_input_keys:
            times = self.normalize_arrival_times_nl(arrival_times_nl) * non_pad_mask.squeeze(-1)
            out["times"] = times  # [B, T]

        if "inter_times" in self.extra_input_keys:
            inter = self.normalize_inter_times(inter_times) * non_pad_mask.squeeze(-1)
            out["inter_times"] = inter  # [B, T]

        return out

    # =========================
    # NORMALIZERS
    # =========================
    def normalize_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        return (inter_times - self.tau_min) / (self.tau_max - self.tau_min + self.eps)

    def normalize_log_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        log_tau = torch.log(torch.clamp_min(inter_times, self.eps)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def normalize_arrival_times_nl(self, arrival_times_nl: torch.Tensor) -> torch.Tensor:
        """normalize arrival times"""
        scale = self.time_scale_base if self.normalize_time else torch.tensor(1.0, dtype=torch.float32)
        return arrival_times_nl * self.Twindow / scale


    def get_extra_inputs(self, batch_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"event_time": batch_tensor[:, :, 1]}
    

    def normalize_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        if self.use_mag_cdf:
            # use magnitude CDF normalization
            device = mag.device
            dtype = mag.dtype
            b = self.richter_b.to(device=device, dtype=dtype)
            mag_comp = self.mag_completeness.to(device=device, dtype=dtype)
            cdf = 1 - torch.exp(-b * math.log(10.0) * (mag - mag_comp + 1e-3))
            return 2*cdf-1
        else:
            device = mag.device
            dtype = mag.dtype
            mag_mean_gr = self.mag_mean_gr.to(device=device, dtype=dtype)
            b = self.richter_b.to(device=device, dtype=dtype)

            return (mag-mag_mean_gr)*b
        # return mag