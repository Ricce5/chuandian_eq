import torch
from mamba_ssm import Mamba, Mamba2
from .base import BGModel
from .kernel import _causal_depthwise_conv1d


@BGModel.register("mamba")
class MambaBGModel(BGModel):
    """SSM-based background intensity model using Mamba.

    Maps input features over time through a Mamba SSM, then projects
    the hidden states to a scalar non-negative intensity with a
    positive weight vector from :class:`BGModel`.
    """

    def __init__(
        self,
        d_feature: int,
        scale_init: float,
        model_type: str,
        d_model: int,
        d_state: int,
        device: torch.device | None = None,
        smooth_kernel_size: int | None = None,
    ):
        super().__init__(device=device, scale_init=scale_init)
        self.d_feature = d_feature
        self.d_state = d_state

        cls = {"mamba": Mamba, "mamba2": Mamba2}.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.mamba = cls(d_model=d_model, d_state=d_state, d_conv=4)
        self.fc_in = torch.nn.Linear(d_feature, d_model, bias=False)
        self.fc_out = torch.nn.Linear(d_model, 1, bias=False)

        # Optional causal depthwise smoothing to suppress fast spikes while staying online/causal.
        if smooth_kernel_size is not None and smooth_kernel_size > 1:
            h = torch.ones(smooth_kernel_size, dtype=torch.float32) / float(smooth_kernel_size)
            self.register_buffer("smoothing_kernel", h, persistent=False)
        else:
            self.smoothing_kernel = None
        
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
        if self.smoothing_kernel is not None:
            ssm_out = _causal_depthwise_conv1d(ssm_out, self.smoothing_kernel)
        
        intensity = self.fc_out(ssm_out)
        return intensity  # (B, T, 1)
