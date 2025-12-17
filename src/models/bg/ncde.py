import torch
from src.models.ncde.slcde import LinearCDEBlock
from src.data.dot_dict import DotDict
from .base import BGModel


@BGModel.register("ncde")
class NCDEBGModel(BGModel):

    def __init__(self, d_feature: int,  scale_init: float,
                d_model: int, device: torch.device | None = None,
                  **kwargs):
        super().__init__(device=device,scale_init=scale_init)
        self.d_feature = d_feature
        self.fc_in = torch.nn.Linear(d_feature, d_model)

        self.ncde = LinearCDEBlock(
                        input_dim=d_model,
                        hidden_dim=d_model,
            **kwargs    
        )
        self.fc_out = torch.nn.Linear(d_model, 1)
        self.device = device
        if device is not None:
            self.to(device)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        model_in = self.fc_in(time_series)  # (B, T, d_model)
        model_out = self.ncde(model_in.contiguous())  # (B, T, d_model)
        scaled_intensity = torch.nn.functional.softplus(self.fc_out(model_out))  # (B, T, 1)   
        return scaled_intensity



    

  