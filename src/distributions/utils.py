import torch


def clamp_preserve_gradients(x: torch.Tensor, min: float, max: float) -> torch.Tensor:
    """Clamp the tensor while preserving gradients in the clamped region.""" # 在数值上阶段，但在反向传播时保留梯度
    return x + (x.clamp(min, max) - x).detach()
