import torch
from torch.distributions import constraints

from .distribution import Distribution
from torch.distributions.utils import broadcast_all



class Lomax(Distribution):
    """
    Lomax (Pareto Type II) distribution on x >= 0
        PDF: f(x) = (α/λ) * (1 + x/λ)^(-(α+1))
        CDF: F(x) = 1 - (1 + x/λ)^(-α)
        SF : S(x) = (1 + x/λ)^(-α)
        Hazard: h(x) = α / (λ + x)

    Args:
        scale (Tensor): λ > 0 (scale)
        shape (Tensor): α > 0 (shape)
        eps (float): small constant for numerical stability
    """
    arg_constraints = {"scale": constraints.positive, "shape": constraints.positive}
    support = constraints.greater_than_eq(0.0)

    def __init__(self, scale: torch.Tensor, shape: torch.Tensor, eps: float = 1e-10, validate_args=None):
        self.scale, self.shape = broadcast_all(scale, shape)  # λ, α
        self.eps = eps
        batch_shape = self.scale.shape
        super().__init__(batch_shape, validate_args=validate_args)

    def _one_plus_x_over_scale(self, x):
        # t = 1 + x/λ，确保数值稳定且 x >= 0
        x = torch.clamp_min(x, 0.0)
        return 1.0 + x / torch.clamp_min(self.scale, self.eps)

    def log_survival(self, x):
        # log S(x) = -α * log(1 + x/λ)
        t = self._one_plus_x_over_scale(x)
        return -self.shape * torch.log(torch.clamp_min(t, self.eps))

    def log_hazard(self, x):
        # log h(x) = log α - log(λ + x)
        x = torch.clamp_min(x, 0.0)
        return self.shape.log() - torch.log(torch.clamp_min(self.scale + x, self.eps))

    def log_prob(self, x):
        # log f(x) = log α - log λ - (α+1)*log(1 + x/λ)
        t = self._one_plus_x_over_scale(x)
        return self.shape.log() - self.scale.log() - (self.shape + 1.0) * torch.log(torch.clamp_min(t, self.eps))

    @property
    def mean(self):
        # E[X] = λ / (α - 1)  (α > 1)，否则为 +inf
        out = self.scale / torch.clamp(self.shape - 1.0, min=self.eps)
        # 对 α <= 1 的位置设为 +inf
        mask = self.shape <= 1.0
        if mask.any():
            out = out.clone()
            out[mask] = torch.tensor(float("inf"), dtype=out.dtype, device=out.device)
        return out

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampling via exponential variable:
          If E ~ Exp(1), then X = λ*(exp(E/α) - 1) ~ Lomax(λ, α)
        """
        shape = torch.Size(sample_shape) + self.batch_shape
        # E ~ Exp(1)
        E = torch.empty(shape, device=self.scale.device, dtype=self.scale.dtype).exponential_(1.0)
        return self.scale * (torch.exp(E / self.shape) - 1.0)

    def sample_conditional(self, lower_bound, sample_shape=torch.Size()):
        """
        Sample X | X >= lower_bound.
        For Lomax, the excess life is still Lomax with updated scale:
           X - L | X >= L  ~  Lomax(λ + L, α)
        So we can sample Y ~ Lomax(λ+L, α) and return X = L + Y.
        """
        L = torch.clamp_min(lower_bound, 0.0)
        # 生成与 batch 对齐的形状
        shape = torch.Size(sample_shape) + torch.broadcast_shapes(L.shape, self.batch_shape)

        # 采样 Y ~ Lomax(λ+L, α) 使用同样的 reparam 技巧
        lam = (self.scale + L).expand(shape)
        alpha = self.shape.expand(shape)
        E = torch.empty(shape, device=lam.device, dtype=lam.dtype).exponential_(1.0)
        Y = lam * (torch.exp(E / alpha) - 1.0)
        return L + Y