import math
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Gamma as TorchGamma, constraints
from torch.distributions.utils import broadcast_all

from .distribution import Distribution
from .mixture import MixtureSameFamily  # 你扩展过的版本
from .lomax import Lomax                            # 上次我给你的 Lomax 类（scale=lambda, shape=alpha）


# ---------- 适配：给 Gamma 增加 log_survival 与条件采样 ----------
class _SurvivalMixin:
    def log_survival(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        p = self.cdf(x).clamp(0.0, 1.0 - eps)
        return torch.log1p(-p)

    def sample_conditional(self, lower_bound, sample_shape=torch.Size()):
        # 右截：X|X>=L  =>  用分位数变换： q = F(L)+U*(1-F(L));  X = F^{-1}(q)
        L = torch.as_tensor(lower_bound, dtype=self.rate.dtype, device=self.rate.device)
        L = L.expand(self.batch_shape)
        F_L = self.cdf(L).clamp(0.0, 1.0 - 1e-12)
        shape = torch.Size(sample_shape) + self.batch_shape
        u = torch.empty(shape, device=L.device, dtype=L.dtype).uniform_()
        q = F_L.expand(shape) + u * (1.0 - F_L).expand(shape)
        return self.icdf(q)

class GammaWithSurvival(_SurvivalMixin, TorchGamma):
    """
    与 torch.distributions.Gamma 参数一致：
      concentration=alpha>0, rate=beta>0
    额外提供：log_survival, sample_conditional
    """
    pass


# ---------- 适配：把 Lomax 右移到起点 M0 ----------
class ShiftedLomax(Distribution):
    """
    若 X ~ Lomax(lambda, alpha)（定义域 x>=0），则 M = M0 + X 的 PDF：
      f_M(m) = (alpha/lambda) * (1 + (m-M0)/lambda)^(-(alpha+1)),  m>=M0
    """
    arg_constraints = {"scale": constraints.positive, "shape": constraints.positive}

    def __init__(self, scale: torch.Tensor, shape: torch.Tensor, M0: torch.Tensor, eps: float = 1e-10, validate_args=None):
        self.scale, self.shape, self.M0 = broadcast_all(scale, shape, M0)
        self.eps = eps
        batch_shape = torch.broadcast_shapes(self.scale.shape, self.shape.shape, self.M0.shape)
        super().__init__(batch_shape, validate_args=validate_args)
        self._base = Lomax(scale=self.scale, shape=self.shape, eps=eps, validate_args=False)

    def _to_x(self, m):
        return torch.clamp_min(m - self.M0, 0.0)

    def log_prob(self, m):
        return self._base.log_prob(self._to_x(m))

    def log_survival(self, m):
        return self._base.log_survival(self._to_x(m))

    def log_hazard(self, m):
        return self._base.log_hazard(self._to_x(m))

    def rsample(self, sample_shape=torch.Size()):
        x = self._base.rsample(sample_shape)
        return x + self.M0.expand(x.shape)

    def sample_conditional(self, lower_bound, sample_shape=torch.Size()):
        # 条件 M >= L 等价于 X >= L - M0（负值时当作 0）
        L = torch.as_tensor(lower_bound, dtype=self.scale.dtype, device=self.scale.device)
        Lx = torch.clamp_min(L - self.M0, 0.0)
        x = self._base.sample_conditional(Lx, sample_shape)
        return x + self.M0.expand(x.shape)


# ---------- 核心：工厂类，返回 (b 混合, 震级混合) ----------
class SeismicMixtureFactory:
    """
    用法：
      factory = SeismicMixtureFactory(
          weight_logits=...,              # [..., K]
          b_concentration=...,            # [..., K]  (alpha_k)
          b_rate=...,                     # [..., K]  (beta_k)
          M0=...,                         # [...] or scalar
      )
      b_mixture, mag_mixture = factory.build()

    结果：
      b_mixture   = MixtureSameFamily( Categorical(logits), GammaWithSurvival(...) )
      mag_mixture = MixtureSameFamily( Categorical(logits), ShiftedLomax(scale=1/(beta ln10), shape=alpha, M0) )
    """
    def __init__(self, weight_logits: torch.Tensor,
                       b_concentration: torch.Tensor,
                       b_rate: torch.Tensor,
                       M0: torch.Tensor | float,
                       validate_args: bool = False):
        # 归一化 logits（可选，你也可以在外部先做 log_softmax）
        self.weight_logits = weight_logits
        self.alpha = b_concentration
        self.beta = b_rate
        self.M0 = torch.as_tensor(M0, dtype=self.alpha.dtype, device=self.alpha.device)
        self.validate_args = validate_args

        # 形状校验：最后一维为 K（分量数）
        if self.weight_logits.shape[-1] != self.alpha.shape[-1] or self.alpha.shape[-1] != self.beta.shape[-1]:
            raise ValueError("The last dimension (K) of weight_logits, b_concentration, and b_rate must match.")

    def build(self):
        # 1) b 的混合（Gamma 分量）
        mix = Categorical(logits=self.weight_logits)                 # [..., K]
        b_components = GammaWithSurvival(self.alpha, self.beta)      # batch_shape=[..., K]
        b_mixture = MixtureSameFamily(mixture_distribution=mix, component_distribution=b_components)

        # 2) 震级的混合（Shifted Lomax 分量）
        ln10 = torch.tensor(math.log(10.0), dtype=self.beta.dtype, device=self.beta.device)
        lam = 1.0 / (self.beta * ln10)                               # λ_k = 1/(β_k ln 10)
        shape = self.alpha                                           # α_k
        mag_components = ShiftedLomax(scale=lam, shape=shape, M0=self.M0)
        mag_mixture = MixtureSameFamily(mixture_distribution=mix, component_distribution=mag_components)

        return b_mixture, mag_mixture
