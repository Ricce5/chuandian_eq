import torch
import math
from torch.distributions import constraints
from .distribution import Distribution

LOG10 = math.log(10.0)

class GutenbergRichter(Distribution):
    arg_constraints = {"b": constraints.positive}

    def __init__(self, b, mag_min=2.0, mag_max=10.0):
        self.b = b  # b now can be a tensor of size (B, L)
        self.mag_min = mag_min
        self.mag_max = mag_max
        batch_shape = b.shape
        super().__init__(batch_shape, validate_args=False)

        # 计算规范化常数，用于 CDF 和 PDF 的规范化
        self._dM = torch.as_tensor(self.mag_max - self.mag_min, dtype=self.b.dtype, device=self.b.device)
        self._ten_pow_neg_b_dM = torch.exp(-self.b * LOG10 * self._dM)  # 10^{-b*(Mmax-Mmin)}
        self._norm_denom = 1.0 - self._ten_pow_neg_b_dM  # 1 - 10^{-bΔ}

        # 防止极端 b 值下可能出现 0
        self._norm_denom = torch.clamp(self._norm_denom, min=torch.finfo(self.b.dtype).eps)

    @property
    def support(self):
        return constraints.interval(self.mag_min, self.mag_max)

    # -------- 核心：log pdf / cdf / survival / hazard --------
    def log_prob(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Expand b to match the shape of x
        b = self.b.expand_as(x)  # Shape (B, L)

        valid = (x >= self.mag_min) & (x <= self.mag_max)
        denom = 10 ** (-b * self.mag_min) - 10 ** (-b * self.mag_max)
        log_norm_const = torch.log(b * LOG10) - torch.log(denom)
        log_pdf = log_norm_const - b * LOG10 * x
        
        if mask is not None:
            log_pdf = torch.where(mask, log_pdf, torch.zeros_like(log_pdf))  # Masking invalid values
        return torch.where(valid, log_pdf, torch.zeros_like(x))  # Ensure values outside [mag_min, mag_max] are 0

    def log_survival(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        b = self.b.expand_as(x)  # Shape (B, L)
        # log(1 - F(x))，用稳定形式
        F = self.cdf(x)
        survival = torch.log1p(-F)
        
        if mask is not None:
            survival = torch.where(mask, survival, torch.zeros_like(survival))
        return survival

    def cdf(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.b.dtype, device=self.b.device)
        b = self.b.expand_as(x)  # Shape (B, L)
        z = torch.exp(-b * LOG10 * (x - self.mag_min))  # 10^{-b (x - Mmin)} = exp( -b ln10 * ... )
        F = (1.0 - z) / self._norm_denom  # Use precomputed normalization constant
        F = torch.clamp(F, 0.0, 1.0)
        F = torch.where(x < self.mag_min, torch.zeros_like(F), F)
        F = torch.where(x > self.mag_max, torch.ones_like(F), F)
        return F
    
    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size(sample_shape) + self.batch_shape
        u = torch.empty(shape, device=self.b.device, dtype=self.b.dtype).uniform_()  # Generate uniform random numbers
        # Generate samples based on each event's b value
        return self.b.reciprocal().neg() * torch.log10(
            -u * (10 ** (-self.b * self.mag_min) - 10 ** (-self.b * self.mag_max)) 
            + 10 ** (-self.b * self.mag_min)
        )

    def log_likelihood(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        log_probs = self.log_prob(x, mask)
        return log_probs.sum(-1)  # Sum across the second dimension (event dimension)

    def aic(self, x: torch.Tensor, mask: torch.Tensor = None, k: int = 1) -> torch.Tensor:
        ll = self.log_likelihood(x, mask)
        return -2.0 * ll + 2.0 * k

    def bic(self, x: torch.Tensor, mask: torch.Tensor = None, k: int = 1) -> torch.Tensor:
        n = torch.as_tensor(x.numel(), dtype=self.b.dtype, device=self.b.device)
        ll = self.log_likelihood(x, mask)
        return -2.0 * ll + k * torch.log(n)

    def ks_stat_pvalue(self, x: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            x = x[mask]  # Apply mask to filter out invalid values
        x = x[(x >= self.mag_min) & (x <= self.mag_max)]
        
        if x.numel() == 0:
            raise ValueError("No samples within [mag_min, mag_max].")
        
        x_sorted, _ = torch.sort(x)
        n = x_sorted.numel()
        F = self.cdf(x_sorted)
        ecdf = torch.arange(1, n + 1, device=x.device, dtype=x.dtype) / n
        D_plus = torch.max(ecdf - F)
        D_minus = torch.max(F - (ecdf - 1.0 / n))
        D = torch.max(D_plus, D_minus)
        # 近似 p 值（Kolmogorov 渐近式，保守）
        p = 2.0 * torch.exp(-2.0 * n * D * D)
        p = torch.clamp(p, 0.0, 1.0)
        return D, p
