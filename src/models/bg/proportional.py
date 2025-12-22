import abc
import torch
from src.utils.interp import interp_uniform_time_series, integrate_uniform_time_series
from src.data.dot_dict import DotDict
from .base import BGModel

@BGModel.register("proportional")
class ProportionalBGModel(BGModel):
    def __init__(self,d_feature, scale_init,device=None,no_weight_decay=False):
        super().__init__(device=device,scale_init=scale_init,no_weight_decay=no_weight_decay)
        self.ts_batch_cache = None
        self.device = device
        self.fc = torch.nn.Linear(d_feature, 1, bias=False)
        if no_weight_decay:
            self.fc.weight._no_weight_decay = True  
        torch.nn.init.constant_(self.fc.weight, 1.0)
        if device is not None:
            self.to(device)
    
    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        scaled_intensity = self.fc(time_series)  # (B, T, 1)
        return scaled_intensity

