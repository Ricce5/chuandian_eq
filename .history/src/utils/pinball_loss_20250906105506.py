import torch
import torch.nn as nn
import torch.nn.functional as F

class PinballLoss(nn.Module):
    """
    Pinball (Quantile) loss.
    - 单分位数: 传入 tau (float)
    - 多分位数: 传入 taus (list/tuple of floats)，input 需对应形状 [N, Tq]
    - 平滑版（Quantile Huber）: 传入 huber_k>0
    - 分位数不交叉: non_crossing=True 时对输出做 cummax
    """
    def __init__(self, tau: float = 0.5, taus=None, reduction: str = "mean",
                 huber_k: float = None, non_crossing: bool = False):
        super().__init__()
        if taus is not None:
            # 统一到升序
            taus = sorted(list(taus))
            self.register_buffer("taus", torch.tensor(taus, dtype=torch.float32).view(1, -1))
            self.single = False
        else:
            assert 0.0 < tau < 1.0, "tau must be in (0,1)"
            self.register_buffer("taus", torch.tensor([[tau]], dtype=torch.float32))  # [1,1]
            self.single = True
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction
        self.huber_k = float(huber_k) if huber_k is not None else None
        self.non_crossing = bool(non_crossing)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        input:  单分位数 -> [N] 或 [N,1]
                多分位数 -> [N, Tq]
        target: [N] 或 [N,1]
        """
        taus = self.taus.to(input.device)           # [1,Tq]
        if target.ndim == 1:
            target = target.unsqueeze(1)            # [N,1]
        if input.ndim == 1:
            input = input.unsqueeze(1)              # [N,1]

        # 可选：强约束避免分位数交叉
        if (not self.single) and self.non_crossing:
            input = torch.cummax(input, dim=1).values

        # 广播到 [N, Tq]
        if target.size(1) == 1 and input.size(1) > 1:
            target = target.expand(-1, input.size(1))

        diff = target - input                       # [N, Tq]

        if self.huber_k is None:  # 标准 pinball
            loss = torch.maximum(taus * diff, (taus - 1.0) * diff)
        else:  # Quantile Huber（更平滑，梯度稳定）
            k = self.huber_k
            abs_diff = diff.abs()
            quad = torch.minimum(abs_diff, torch.tensor(k, device=diff.device))
            lin = abs_diff - quad
            huber = 0.5 * quad * quad / k + lin
            loss = torch.where(diff >= 0, taus * huber, (taus - 1.0) * huber)

        # 归约
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # [N, Tq]
