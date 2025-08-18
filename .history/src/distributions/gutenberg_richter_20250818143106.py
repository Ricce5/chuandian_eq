import torch
from torch.distributions import constraints
import math
from .distribution import Distribution

LOG10 = math.log(10.0)

class GutenbergRichter(Distribution):
    arg_constraints = {"b": constraints.positive}

    def __init__(self, b, mag_min=2.0, mag_max=10):
        self.b = b
        self.mag_min = mag_min
        self.mag_max = mag_max
        batch_shape = b.shape
        super().__init__(batch_shape, validate_args=False)

    def log_hazard(self, x):
        return torch.zeros_like(x)

    def log_survival(self, x):
        return torch.zeros_like(x)

    def log_prob(self, x):
        valid = (x >= self.mag_min) & (x <= self.mag_max)
        denom = 10 ** (-self.b * self.mag_min) - 10 ** (-self.b * self.mag_max)
        log_norm_const = torch.log(self.b * LOG10) - torch.log(denom)
        log_pdf = log_norm_const - self.b * LOG10 * x
        return torch.where(valid, log_pdf, torch.full_like(x, float("-inf")))

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.b.dtype, device=self.b.device)
        z = torch.exp(-self.b * LOG10 * (x - self.mag_min))          # 10^{-b (x - Mmin)} = exp( -b ln10 * ... )
        F = (1.0 - z) / self._norm_denom
        F = torch.clamp(F, 0.0, 1.0)
        F = torch.where(x < self.mag_min, torch.zeros_like(F), F)
        F = torch.where(x > self.mag_max, torch.ones_like(F), F)


        
        return F
    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        u = torch.empty(shape, device=self.b.device, dtype=self.b.dtype).uniform_() # Generate uniform random numbers
        return self.b.reciprocal().neg() * torch.log10(
            -u * (10 ** (-self.b * self.mag_min) - 10 ** (-self.b * self.mag_max)) 
            + 10 ** (-self.b * self.mag_min)
        )
