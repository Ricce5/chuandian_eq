import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from src.utils.mask_utils import get_non_pad_mask

logger = logging.getLogger(__name__)


class MixerAdapter:
    """
    input adapter of mixer for classification and regression tasks
    """
    def __init__(self, args, revin_layer: Optional[nn.Module] = None):
        stats = args.stats
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

