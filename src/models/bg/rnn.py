import torch
from torch.nn import LSTM, GRU
from src.data.dot_dict import DotDict
from .base import BGModel

@BGModel.register("rnn")
class RNNBGModel(BGModel):
    """RNN-based background intensity model using LSTM or GRU.

    Maps input features (F) to d_model, processes them over time 
    through a selected RNN (LSTM/GRU), then projects the hidden states 
    (d_model) to a scalar non-negative intensity.
    """

    def __init__(self, d_feature: int, scale_init: float,
                 d_model: int, model_type: str, num_layers: int = 1,
                 device: torch.device | None = None, ):
        
        super().__init__(device=device, scale_init=scale_init)
        self.d_feature = d_feature
        self.d_model = d_model
        rnn_cls = {"lstm": LSTM, "gru": GRU}.get(model_type.lower())
        
        if rnn_cls is None:
            raise ValueError(f"Unknown RNN model_type: {model_type}. Must be 'lstm' or 'gru'.")
        # (B, T, d_feature) -> (B, T, d_model)
        self.fc_in = torch.nn.Linear(d_feature, d_model)

        self.rnn = rnn_cls(
            input_size=d_model,           # 输入维度现在是 d_model
            hidden_size=d_model,          # 隐藏状态维度是 d_model
            num_layers=num_layers,
            batch_first=True,             # 保证 (B, T, H) 形状
            bidirectional=False
        )

        self.fc_out = torch.nn.Linear(d_model, 1)
        self.device = device
        
        if device is not None:
            self.to(device)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        """Project input features to hidden states and apply the selected RNN.

        Args:
            time_series: input features of shape (B, T, F)

        Returns:
            ``(B, T, 1)`` tensor of non-negative scaled intensity.
        """
        rnn_in = self.fc_in(time_series)  
        rnn_out, _ = self.rnn(rnn_in.contiguous())  # (B, T, d_model)
        out = self.fc_out(rnn_out)  # (B, T, 1)
        return out
