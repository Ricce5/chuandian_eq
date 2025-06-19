import torch
import torch.nn as nn
import math



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
    
class ScaledSoftplus(nn.Module):
    '''
    Use different beta for mark-specific intensities
    '''
    def __init__(self, num_marks, threshold=20.):
        super(ScaledSoftplus, self).__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_marks), requires_grad=True)  # [num_marks]

    def forward(self, x):
        '''
        :param x: [..., num_marks]
        '''
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(
            beta_x <= self.threshold,
            torch.log1p(beta_x.clamp(max=math.log(1e5)).exp()) / beta,
            x,  # if above threshold, then the transform is effectively linear
        )