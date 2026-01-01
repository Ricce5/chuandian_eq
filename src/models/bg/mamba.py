import torch
from mamba_ssm import Mamba,Mamba2
from src.data.dot_dict import DotDict
from .base import BGModel


@BGModel.register("mamba")
class MambaBGModel(BGModel):
    """SSM-based background intensity model using Mamba.

    Maps input features over time through a Mamba SSM, then projects
    the hidden states to a scalar non-negative intensity with a
    positive weight vector from :class:`BGModel`.
    """

    def __init__(self, d_feature: int,  scale_init: float,
                 model_type: str, d_model: int, d_state: int, 
                 device: torch.device | None = None,
                 fc_in_bias: bool = True, fc_out_bias: bool = True,
                 take_abs: bool = False):
        super().__init__(device=device,scale_init=scale_init)
        self.d_feature = d_feature
        self.d_state = d_state
        self.take_abs = take_abs

        cls = {"mamba": Mamba, "mamba2": Mamba2}.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.mamba = cls(d_model=d_model, d_state=d_state, d_conv=4)
        self.fc_in = torch.nn.Linear(d_feature, d_model, bias=fc_in_bias)
        self.fc_out = torch.nn.Linear(d_model, 1, bias=fc_out_bias)
        self.device = device
        if device is not None:
            self.to(device)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        """Project input features to hidden states and apply Mamba SSM.

        Args:
            time_series: input features of shape (B, T, F)

        Returns:
            ``(B, T, 1)`` tensor of (non-negative) intensities.
        """
        ssm_in = self.fc_in(time_series)  # (B, T, d_model)
        ssm_out = self.mamba(ssm_in.contiguous())  # (B, T, d_model)
        
        if self.take_abs:
            scaled_intensity = torch.abs(self.fc_out(ssm_out))
        else:
            scaled_intensity = torch.nn.functional.softplus(self.fc_out(ssm_out))
        return scaled_intensity



    

  