import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from .distribution import Distribution  
from src.data.constants import PAD
class Gamma(Distribution):
    """
    Gamma(α, β) with α>0 (concentration), β>0 (rate)
    PDF: f(x) = β^α / Γ(α) * x^(α-1) * exp(-β x),  x > 0
    mean = α / β
    var  = α / β^2
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive

    def __init__(self, concentration: torch.Tensor, rate: torch.Tensor, validate_args=None):
        concentration_mask = concentration == PAD
        rate_mask = rate == PAD
        assert not torch.any(concentration_mask & ~rate_mask), "concentration 和 rate 的 PAD 位置必须一致"
        self.concentration, self.rate = broadcast_all(concentration, rate)
        batch_shape = self.concentration.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        alpha = self.concentration
        beta = self.rate

        return (
            alpha * beta.log()
            - torch.lgamma(alpha)
            + (alpha - 1) * value.log()
            - beta * value
        )

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling; requires α>0 (true here).
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
