# ref: https://zenodo.org/records/8161777 Using Deep Learning for Flexible and Scalable Earthquake Forecasting
import torch
import math
from torch.distributions import constraints
from .distribution import Distribution
from typing import Optional

LOG10 = math.log(10.0)

class GutenbergRichter(Distribution):
    arg_constraints = {"b": constraints.positive}

    def __init__(self, b, mag_min=2.0, mag_max=10.0):
        self.b = torch.tensor(b, dtype=torch.float32) if not isinstance(b, torch.Tensor) else b
        self.mag_min = mag_min
        self.mag_max = mag_max
        batch_shape = self.b.shape
        super().__init__(batch_shape, validate_args=False)


    def norm_denom(self, b:Optional[torch.Tensor]=None):
        if b is None:
            b = self.b
        dM = torch.as_tensor(self.mag_max - self.mag_min, dtype=b.dtype, device=b.device)
        ten_pow_neg_b_dM = torch.exp(-b * LOG10 * dM)  # 10^{-b*(Mmax-Mmin)}
        norm_denom = 1.0 - ten_pow_neg_b_dM  # 1 - 10^{-bΔ}
        norm_denom = torch.clamp(norm_denom, min=torch.finfo(b.dtype).eps)
        return norm_denom

    @property
    def support(self):
        return constraints.interval(self.mag_min, self.mag_max)

    # -------- log pdf / cdf / survival / hazard --------
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
        # log(1 - F(x))
        F = self.cdf(x)
        survival = torch.log1p(-F)
        
        if mask is not None:
            survival = torch.where(mask, survival, torch.zeros_like(survival))
        return survival

    def cdf(self, x: torch.Tensor ,b:Optional[torch.Tensor]=None) -> torch.Tensor:
        x = torch.as_tensor(x, dtype=self.b.dtype, device=self.b.device)
        if b is None:
            b = self.b.expand_as(x)  # Shape (B, L)
        z = torch.exp(-b * LOG10 * (x - self.mag_min))  # 10^{-b (x - Mmin)} = exp( -b ln10 * ... )
        F = (1.0 - z) / self.norm_denom(b)
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
        """
        Compute the Kolmogorov-Smirnov (KS) statistic and its approximate p-value 
        for a given set of samples and their corresponding theoretical cumulative 
        distribution function (CDF).
        Args:
            x (torch.Tensor): A tensor of sample values.
            mask (torch.Tensor, optional): A boolean mask tensor to filter valid 
            samples and corresponding `b` values. If None, no filtering is applied.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - D (torch.Tensor): The KS statistic, representing the maximum 
              difference between the empirical CDF and the theoretical CDF.
            - p (torch.Tensor): The approximate p-value for the KS statistic, 
              computed using the Kolmogorov asymptotic formula.
        Raises:
            ValueError: If no samples fall within the range `[mag_min, mag_max]`.
        Notes:
            - The method filters out samples that are not within the range 
              `[mag_min, mag_max]` before computing the KS statistic.
            - The theoretical CDF is computed using the `self.cdf` method, which 
              takes the sorted samples and their corresponding `b` values.
            - The p-value is clamped to the range [0.0, 1.0] for numerical stability.
        """
        if mask is not None:
            x = x[mask]  # Apply mask to filter valid samples
            b = self.b[mask]  # Similarly, filter valid `b` values
        else:
            b = self.b  # Use original `b` if no mask is provided

        # Filter out samples outside the range [mag_min, mag_max]
        valid_mask = (x >= self.mag_min) & (x <= self.mag_max)
        x = x[valid_mask]  # Select valid samples within the magnitude range
        b = b[valid_mask]  # Synchronize `b` with valid samples

        if x.numel() == 0:
            raise ValueError("No samples within [mag_min, mag_max].")
        
        # Sort valid samples
        x_sorted, indices = torch.sort(x)
        b_sorted = b[indices]  # Sort `b` based on the indices of sorted `x`

        n = x_sorted.numel()

        # Compute the theoretical CDF for each sorted sample
        F = self.cdf(x_sorted, b_sorted)

        # Compute the empirical CDF: ecdf = [1/n, 2/n, ..., n/n]
        ecdf = torch.arange(1, n + 1, device=x.device, dtype=x.dtype) / n

        # Compute the Kolmogorov-Smirnov statistic
        D_plus = torch.max(ecdf - F)
        D_minus = torch.max(F - (ecdf - 1.0 / n))
        D = torch.max(D_plus, D_minus)

        # Approximate p-value (Kolmogorov asymptotic formula, conservative)
        p = 2.0 * torch.exp(-2.0 * n * D * D)
        p = torch.clamp(p, 0.0, 1.0)

        return D, p
       
