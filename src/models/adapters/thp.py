from typing import Optional

import src.data
import torch
import torch.nn as nn


class THPBatchAdapter:
    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model

    def __call__(self, bx: src.data.Batch) -> dict:
        log_inter_times = torch.log(torch.clamp_min(bx.inter_times, 1e-10)).unsqueeze(-1)
        log_inter_times -= self.model.log_tau_mean

        mag = bx.mag[:, :, None] - self.model.mag_mean
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
        mark = mag

        return {
            "event_mark": mark * bx.input_mask[:, :, None],
            "event_time": bx.arrival_times * bx.input_mask,
            "log_inter_time": log_inter_times * bx.input_mask[:, :, None],
            "input_mask": bx.input_mask.float(),
        }
