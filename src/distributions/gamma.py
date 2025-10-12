import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.distributions import Distribution
import torch.nn.functional as F

class Gamma(Distribution):
    """
    Gamma(α, β) distribution, where α > 0 (concentration parameter) and β > 0 (rate parameter).
    PDF: f(x) = β^α / Γ(α) * x^(α-1) * exp(-β * x),  x > 0
    Mean = α / β
    Variance = α / β^2
    """
    arg_constraints = {
        "concentration": constraints.nonnegative,
        "rate": constraints.nonnegative,
    }
    support = constraints.nonnegative

    def __init__(self, concentration: torch.Tensor, rate: torch.Tensor, validate_args=None):
        #  to avoid modifying the input tensors
        concentration = concentration.clone()
        rate = rate.clone()

        # If PAD positions are 0, create masks: mask indicates valid positions
        concentration_mask = (concentration == 0)  # 0 is PAD
        rate_mask = (rate == 0)
        self.mask = (~rate_mask).bool()  # Keep only valid positions

        # Check if PAD positions of concentration and rate are consistent
        if torch.any(concentration_mask != rate_mask):
            raise ValueError("PAD positions of concentration and rate must be consistent.")


        concentration.clamp_min_(1e-3)
        rate.clamp_min_(1e-3)

        self.concentration, self.rate = broadcast_all(concentration, rate)

        assert torch.all(self.concentration > 0), "concentration contains non-positive values"
        assert torch.all(self.rate > 0), "rate contains non-positive values"

        batch_shape = self.concentration.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def log_prob(self, value, mask=None):
        if self._validate_args:
            self._validate_sample(value)

        alpha = self.concentration
        beta = self.rate
        out = alpha * beta.log() - torch.lgamma(alpha) + (alpha - 1) * value.log() - beta * value

        if mask is not None:
            mask = mask.bool()
            assert not torch.any(mask[~self.mask]), "All PAD positions must be masked out."
            out = torch.where(mask, out, torch.zeros_like(out))
        return out
        
    def log_likelihood(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        log_probs = self.log_prob(x, mask)
        return log_probs.sum(-1)  # Sum across the second dimension (event dimension)


    def rsample(self, sample_shape=torch.Size()):
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
        # mode = (α - 1) / β, for α >= 1; mode = 0 for α < 1
        mode = torch.where(self.concentration > 1, (self.concentration - 1) / self.rate, torch.tensor(0.0))
        return mode

    def entropy(self):
        # calculate the entropy of the Gamma distribution
        alpha = self.concentration
        beta = self.rate
        return alpha - torch.log(beta) + torch.lgamma(alpha) + (1 - alpha) * torch.digamma(alpha)

    def cdf(self, value):
        alpha = self.concentration
        beta = self.rate
        return torch.special.gammainc(alpha, beta * value)
