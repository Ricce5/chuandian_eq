import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all

from .distribution import Distribution

class Gamma(Distribution):
    """
    Gamma distribution with shape α > 0 and rate β > 0
    PDF: f(x) = β^α / Γ(α) * x^(α-1) * exp(-β x),  x > 0
    mean = α / β
    var = α / β^2
    """
    arg_constraints = {
        "concentration": constraints.positive,  # α
        "rate": constraints.positive            # β
    }
    support = constraints.positive

    def __init__(self, concentration: torch.Tensor, rate: torch.Tensor, eps: float = 1e-10, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)
        self.eps = eps
        batch_shape = self.concentration.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, x):
        x = torch.clamp_min(x, self.eps)
        alpha = self.concentration
        beta = self.rate
        return (
            alpha * beta.log()
            - torch.lgamma(alpha)
            + (alpha - 1) * x.log()
            - beta * x
        )

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling using PyTorch's internal gamma sampler
        """
        shape = torch.Size(sample_shape) + self.batch_shape
        return torch._standard_gamma(self.concentration.expand(shape)) / self.rate.expand(shape)

    @property
    def mean(self):
        return self.concentration / self.rate

    @property
    def variance(self):
        return self.concentration / (self.rate ** 2)
