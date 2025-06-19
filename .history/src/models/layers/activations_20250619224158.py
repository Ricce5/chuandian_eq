import torch
import torch.nn as nn
import math

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