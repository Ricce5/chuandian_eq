import abc
import torch
from src.utils.interp import interp_uniform_time_series, integrate_uniform_time_series
from src.data.dot_dict import DotDict
from .base import BGModel

@BGModel.register("proportional")
class ProportionalBGModel(BGModel):
    def __init__(self,d_feature, scale_init,device=None):
        super().__init__(device=device,scale_init=scale_init)
        self.ts_batch_cache = None
        self.device = device
        self.fc = torch.nn.Linear(d_feature, 1, bias=False)
        if device is not None:
            self.to(device)
    
    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        out = self.fc(time_series)  # (B, T, 1)
        scaled_intensity = torch.nn.functional.softplus(out)  # (B, T, 1)   
        return scaled_intensity

