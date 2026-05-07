import torch.nn as nn


def _resolve_output_activation(name):
    if name is None:
        return None

    key = str(name).strip().lower()
    if key == "null":
        return None
    if key == "sigmoid":
        return nn.Sigmoid()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()

    raise ValueError(
        f"Unsupported LSTM output_activation: {name}. "
        "Use null or one of: sigmoid, tanh, relu."
    )


class LSTM(nn.Module):
    def __init__(
        self,
        feature_size,
        hidden_size,
        output_size,
        num_layers,
        lstm_dropout,
        device,
        output_activation="sigmoid",
    ):
        super().__init__()
        self.device = device
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, self.num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.dropout = nn.Dropout(p=lstm_dropout)
        self.output_activation = _resolve_output_activation(output_activation)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        output = self.dropout(output)
        output = self.fc(output)
        if self.output_activation is not None:
            output = self.output_activation(output)
        return output[:, -1, :].squeeze(1)
