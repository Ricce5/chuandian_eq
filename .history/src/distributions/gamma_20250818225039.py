import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from .distribution import Distribution  
from src.data.constants import PAD

class Gamma(Distribution):
    """
    Gamma(α, β)分布，其中α > 0（浓度参数），β > 0（速率参数）
    PDF: f(x) = β^α / Γ(α) * x^(α-1) * exp(-β * x),  x > 0
    均值 = α / β
    方差 = α / β^2
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive

    def __init__(self, concentration: torch.Tensor, rate: torch.Tensor, validate_args=None):
        concentration_mask = concentration == PAD
        rate_mask = rate == PAD
        self.mask = ~rate_mask
        assert not torch.any(concentration_mask & ~rate_mask), "concentration 和 rate 的 PAD 位置必须一致"
        concentration.clamp_min(1e-10)
        rate.clamp_min(1e-10)
        self.concentration, self.rate = broadcast_all(concentration, rate)
        batch_shape = self.concentration.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value, mask=None):
        if self._validate_args:
            self._validate_sample(value)
        alpha = self.concentration
        beta = self.rate
        out = (
            alpha * beta.log()
            - torch.lgamma(alpha)
            + (alpha - 1) * value.log()
            - beta * value
        )

        assert torch.all(mask.bool()[~self.mask]) == False,  "All PAD positions must be masked out."
        if mask is not None:
            out = torch.where(mask, out, torch.zeros_like(out))
        return out

    def rsample(self, sample_shape=torch.Size()):
        """
        使用重参数化技巧进行采样；要求α > 0（这里满足条件）。
        """
        shape = torch.Size(sample_shape) + self.batch_shape
        alpha = self.concentration.expand(shape)
        beta = self.rate.expand(shape)
        return torch._standard_gamma(alpha) / beta

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate ** 2)

    @property
    def mode(self):
        """
        Gamma分布的众数。对于α > 1，众数 = (α - 1) / β。
        对于α ≤ 1，众数为0（即分布会偏向0）。
        """
        mode = torch.where(self.concentration > 1, (self.concentration - 1) / self.rate, torch.tensor(0.0))
        return mode

    def sample(self, sample_shape=torch.Size()):
        """
        直接从Gamma分布中生成样本（非重参数化方式）。
        """
        shape = torch.Size(sample_shape) + self.batch_shape
        alpha = self.concentration.expand(shape)
        beta = self.rate.expand(shape)
        return torch._standard_gamma(alpha) / beta

    def entropy(self):
        """
        计算Gamma分布的熵：
        H(X) = α - log(β) + log(Γ(α)) + (1 - α) * ψ(α)
        其中ψ(α)是Gamma函数的Digamma函数。
        """
        alpha = self.concentration
        beta = self.rate
        return alpha - torch.log(beta) + torch.lgamma(alpha) + (1 - alpha) * torch.digamma(alpha)

    def cdf(self, value):
        """
        计算Gamma分布的累积分布函数（CDF）。
        """
        alpha = self.concentration
        beta = self.rate
        return torch.special.gammainc(alpha, beta * value)
