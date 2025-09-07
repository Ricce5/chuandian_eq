import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.ops import sigmoid_focal_loss

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, reduction="mean"):
        """
        alpha: 平衡因子 (0~1之间)，控制正负样本重要性
        gamma: 调节因子，控制难样本关注度
        reduction: "mean", "sum", "none"
        """
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits, shape [N, *]  (没有经过 sigmoid)
        targets: 标签, shape [N, *]   (取值 0 或 1, float)
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # 预测概率
        pt = torch.exp(-bce_loss)
        # focal loss 公式
        focal_loss =   (1 - pt) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss

class FocalLossWrapper(nn.Module):
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

