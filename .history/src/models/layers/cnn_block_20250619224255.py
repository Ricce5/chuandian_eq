import torch.nn as nn



class CNNBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super(CNNBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_out, 
                      kernel_size=3, padding= 1), # 输入 (B, l, C)
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride= 4, padding=0) 
        )
    
    def forward(self, x):
        x =  self.cnn(x.permute(0, 2, 1)).transpose(1,2) 
        return x