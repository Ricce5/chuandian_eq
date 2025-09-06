import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, feature_size, hidden_size, output_size, num_layers, lstm_dropout, device):
        super().__init__()
        self.device = device
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, self.num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_size, self.output_size).to(self.device)
        self.dropout = nn.Dropout(p=lstm_dropout)

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
        return output[:, -1, :].squeeze(1)