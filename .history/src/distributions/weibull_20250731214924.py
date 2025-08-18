import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
import torch.nn.functional as F

from .distribution import Distribution


class Weibull(Distribution):
    arg_constraints = {"scale": constraints.positive, "shape": constraints.positive}   # scale and shape are parameters of the Weibull distribution

    def __init__(
        self, scale: torch.Tensor, shape: torch.Tensor, eps=1e-10, validate_args=None  # eps is a small value to avoid numerical issues
    ):
        self.scale, self.shape = broadcast_all(scale, shape) # broadcast to ensure they have the same shape
        self.eps = eps
        batch_shape = self.scale.shape
        super().__init__(batch_shape, validate_args=validate_args) # automatically check constraints

    def log_hazard(self, x):
        x = torch.clamp_min(x, self.eps)  # ensure x > 0 for numerical stability
        return self.scale.log() + self.shape.log() + (self.shape - 1) * x.log()

    def log_survival(self, x):
        x = torch.clamp_min(x, self.eps)  # ensure x > 0 for numerical stability
        self.shape = F.softplus(self.shape).clamp(min=1e-3)
        return self.scale.neg() * torch.pow(x, self.shape)

    def log_prob(self, x):
        return self.log_hazard(x) + self.log_survival(x)

    @property
    def mean(self):
        log_lmbd = self.shape.reciprocal().neg() * self.scale.log()
        return torch.exp(log_lmbd + torch.lgamma(1 + self.shape.reciprocal()))

    def rsample(self, sample_shape=torch.Size()): # Reparameterized sampling method
        shape = torch.Size(sample_shape) + self.batch_shape
        z = torch.empty(
            shape, device=self.scale.device, dtype=self.scale.dtype
        ).exponential_(1.0)                       # sample from an exponential distribution  
        samples = (z * self.scale.reciprocal() + self.eps).pow(self.shape.reciprocal()) # .reciprocal() means 1/scale
        return samples

    def sample_conditional(self, lower_bound, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        u = torch.empty(
            shape, device=self.scale.device, dtype=self.scale.dtype
        ).uniform_()
        survival = self.log_survival(lower_bound).exp()   
        u = u * survival                                  # u 小于 survival 
        return (-u.log() * self.scale.reciprocal() + self.eps).pow(
            self.shape.reciprocal()                       # x = S^(-1)(u)  S单调，需要采样大于 lower_bound 的 x  S^(-1)(u) = F^{-1}(1-u)
        )
