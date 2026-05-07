import math
from typing import Dict, List

import torch
from src.utils.mask_utils import get_non_pad_mask


class RTPPTaskInputAdapter:
    """
    Input adapter for classification/regression RNN models.
    Uses RTPP-style event feature normalization with catalog statistics.
    """

    def __init__(self, args):
        stats = args.stats
        to_t = lambda x: torch.tensor(x, dtype=torch.float32)

        self.tau_min = to_t(stats["tau_min"])
        self.tau_max = to_t(stats["tau_max"])
        self.tau_mean = to_t(stats["tau_mean"])
        self.log_tau_mean = self.tau_mean.log()

        b_value = stats.get("b_value", None)
        if b_value is None:
            b_value = getattr(args, "richter_b_mle", None)
        if b_value is None:
            b_value = 1.0
        self.richter_b = to_t(b_value)
        self.mag_completeness = to_t(stats["mag_completeness"])
        self.mag_mean_gr = self.mag_completeness + 1.0 / (math.log(10.0) * self.richter_b)

        self.eps = 1e-10
        self.use_mag_cdf = bool(getattr(args, "use_mag_cdf", False))
        self.features_input_keys: List[str] = sorted(getattr(args, "features_input_keys", ["mag"]))

    def __call__(self, batch_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        arrival_times = batch_tensor[:, :, 0]
        inter_times = batch_tensor[:, :, -1]
        mag = batch_tensor[:, :, 2:3]
        loc = batch_tensor[:, :, 3:5]

        non_pad_mask = get_non_pad_mask(arrival_times)

        feature_map = {
            "mag": self._normalize_magnitude(mag),
            "log_inter_times": self._normalize_log_inter_times(inter_times),
            "inter_times": self._normalize_inter_times(inter_times).unsqueeze(-1),
            "loc": loc,
        }
        features = torch.cat([feature_map[key] for key in self.features_input_keys], dim=-1)
        features = features * non_pad_mask
        return {"features": features, "non_pad_mask": non_pad_mask}

    def _normalize_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        tau_min = self.tau_min.to(device=inter_times.device, dtype=inter_times.dtype)
        tau_max = self.tau_max.to(device=inter_times.device, dtype=inter_times.dtype)
        return (inter_times - tau_min) / (tau_max - tau_min + self.eps)

    def _normalize_log_inter_times(self, inter_times: torch.Tensor) -> torch.Tensor:
        log_tau = torch.log(torch.clamp_min(inter_times, self.eps)).unsqueeze(-1)
        log_tau_mean = self.log_tau_mean.to(device=inter_times.device, dtype=inter_times.dtype)
        return log_tau - log_tau_mean

    def _normalize_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        if self.use_mag_cdf:
            b = self.richter_b.to(device=mag.device, dtype=mag.dtype)
            mc = self.mag_completeness.to(device=mag.device, dtype=mag.dtype)
            cdf = 1 - torch.exp(-b * math.log(10.0) * (mag - mc + 1e-3))
            return 2 * cdf - 1

        b = self.richter_b.to(device=mag.device, dtype=mag.dtype)
        mag_mean_gr = self.mag_mean_gr.to(device=mag.device, dtype=mag.dtype)
        return (mag - mag_mean_gr) * b

