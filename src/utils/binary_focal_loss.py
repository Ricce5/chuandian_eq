import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import sigmoid_focal_loss

class FocalLossWrapper(nn.Module):
    """ 
    Wrapper for sigmoid focal loss from torchvision
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        return sigmoid_focal_loss(
            inputs=inputs,
            targets=targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction
        )

