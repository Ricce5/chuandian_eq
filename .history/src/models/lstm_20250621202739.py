import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.hidden_size = args.lstm_hidden_size
        self.num_layers = args.lstm_num_layers
        self.feature_size =  len(args.feature_cols)
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, self.num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_size,1).to(self.device)
        self.dropout = nn.Dropout(p=args.lstm_dropout)

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