import torch

from src.models.lstm import LSTM


def test_lstm_output_is_bounded_in_0_1():
    model = LSTM(
        feature_size=8,
        hidden_size=16,
        output_size=1,
        num_layers=2,
        lstm_dropout=0.1,
        device=torch.device("cpu"),
    )
    x = torch.randn(32, 5, 8)
    y = model(x)
    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)
