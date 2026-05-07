# Pure Zhuang-normalized ETAS implementation.
# Productivity kernel:
#   g(t, M) = K * exp(alpha_e * (M - M_c)) * (p - 1) * c^(p - 1) * (t + c)^(-p)
import logging
import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from scipy.stats import poisson
from torch.utils.checkpoint import checkpoint
from tqdm.auto import trange

from src.data.batch import Batch, get_mask, pad_sequence
from src.data.sequence import Sequence

from .tpp_model import TPPModel

logger = logging.getLogger(__name__)


def _to_tensor(x, ref: torch.Tensor):
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


def _compute_branching_ratio_ogata(
    K: float = 0.1,
    alpha_e: float = math.log(10.0),
    b: float = 1.0,
    M_min: float = 0.0,
    M_max: float = 10.0,
    M_ref: float = 0.0,
) -> float:
    """Compute infinite-horizon ETAS branching ratio under normalized ETAS parameterization.

    Magnitudes follow a truncated Gutenberg-Richter distribution on [M_min, M_max].
    """
    delta_m = M_max - M_min
    if delta_m <= 0:
        raise ValueError("M_max must be greater than M_min.")
    if b <= 0:
        raise ValueError("richter b must be strictly positive.")
    if K < 0:
        raise ValueError("K must be non-negative.")

    beta = b * math.log(10.0)
    denom = 1.0 - math.exp(-beta * delta_m)
    if np.isclose(denom, 0.0):
        raise ValueError("Invalid magnitude range or b value for branching ratio.")

    slope = alpha_e - beta
    if np.isclose(slope, 0.0):
        integral = delta_m
    else:
        integral = (math.exp(slope * delta_m) - 1.0) / slope

    expectation = (
        math.exp(alpha_e * (M_min - M_ref))
        * beta
        / denom
        * integral
    )
    branching_ratio = K * expectation

    if branching_ratio > 1:
        logger.warning("Branching ratio (normalized ETAS): %s", branching_ratio)
    return branching_ratio


def gen_mag(shape=1, b=1, M_min=0, M_max=10):
    """Draw samples from the truncated Gutenberg-Richter distribution."""
    u = np.random.random(shape)
    mag = (
        -1.0
        / b
        * np.log10(-u * (10 ** (-b * M_min) - 10 ** (-b * M_max)) + 10 ** (-b * M_min))
    )
    return mag


def _omori_int_np(T1, T2, c, p):
    """Integral of (t + c)^(-p) from T1 to T2 (NumPy)."""
    if np.isclose(p, 1.0):
        return np.log(T2 + c) - np.log(T1 + c)
    one_minus_p = 1.0 - p
    return ((T2 + c) ** one_minus_p - (T1 + c) ** one_minus_p) / one_minus_p


def _omori_inv_np(T1, T2, c, p, size=1, t_max=1e10):
    """Draw samples from Omori law on [T1, T2] using inverse transform."""
    if t_max <= 0:
        raise ValueError("t_max must be positive.")
    if c <= 0:
        raise ValueError("c must be positive.")

    T1 = np.clip(np.asarray(T1, dtype=np.float64), 0.0, t_max)
    T2 = np.clip(np.asarray(T2, dtype=np.float64), 0.0, t_max)
    if np.any(T2 < T1):
        raise ValueError("T2 must be >= T1 after clipping to [0, t_max].")

    u = np.random.random(size=size)
    total_mass = _omori_int_np(0.0, t_max, c, p)
    cdf_T1 = _omori_int_np(0.0, T1, c, p) / total_mass
    cdf_T2 = _omori_int_np(0.0, T2, c, p) / total_mass
    u_prime = u * (cdf_T2 - cdf_T1) + cdf_T1

    if np.isclose(p, 1.0):
        return c * np.exp(u_prime * (np.log(t_max + c) - np.log(c))) - c

    one_minus_p = 1.0 - p
    base = u_prime * total_mass * one_minus_p + c ** one_minus_p
    base = np.maximum(base, np.finfo(np.float64).tiny)
    return base ** (1.0 / one_minus_p) - c


class ETASZhuang(TPPModel):
    """ETAS model with Zhuang-style normalized temporal kernel parameterization.

    Uses:
        g = K * exp(alpha_e*(M-Mc)) * (p-1)*c^(p-1) * (t+c)^(-p)

    Notes:
        - Uses natural exponential base for magnitude productivity.
        - Keeps a single parameterization (K, alpha_e) to avoid mixing.
    """

    def __init__(
        self,
        omori_p_init: float = 1.08,
        omori_c_init: float = 0.1,
        base_rate_init: float = 0.02,
        productivity_K_init: float = 0.11,
        productivity_alpha_e_init: float = math.log(10.0),
        richter_b: float = 1.0,
        mag_completeness: float = 2.0,
        mag_max: float = 10.0,
        report_params: bool = True,
        device: Optional[torch.device] = None,
        bg_model=None,
        fix_mu: bool = False,
        fixed_mu_value: Optional[float] = None,
        loss_reduction: str = "per_time",
        query_chunk_size: int = 0,
        history_chunk_size: int = 0,
        use_grad_checkpoint: bool = False,
        enforce_subcritical: bool = False,
        max_branching_ratio: float = 0.95,
        enforce_p_gt_one: bool = False,
        min_omori_p: float = 1.001,
        constraint_softness: float = 1e-3,
    ):
        super().__init__()
        if omori_p_init <= 1.0:
            raise ValueError("Normalized ETAS parameterization requires p > 1.")
        if max_branching_ratio <= 0.0 or max_branching_ratio >= 1.0:
            raise ValueError("max_branching_ratio must be in (0, 1).")
        if min_omori_p <= 1.0:
            raise ValueError("min_omori_p must be strictly larger than 1.")
        if constraint_softness <= 0.0:
            raise ValueError("constraint_softness must be positive.")

        self.enforce_subcritical = bool(enforce_subcritical)
        self.max_branching_ratio = float(max_branching_ratio)
        self.enforce_p_gt_one = bool(enforce_p_gt_one)
        self.min_omori_p = float(min_omori_p)
        self.constraint_softness = float(constraint_softness)

        self.fix_mu = fix_mu
        base_rate_init_t = torch.as_tensor(
            base_rate_init, device=device, dtype=torch.get_default_dtype()
        )

        self.log_p_minus_one = nn.Parameter(torch.tensor(math.log(omori_p_init - 1.0)))
        self.log_c = nn.Parameter(torch.tensor(math.log(omori_c_init)))
        self.log_mu = nn.Parameter(base_rate_init_t.log())
        if self.fix_mu:
            self.log_mu.requires_grad = False
            mu_value = (
                torch.as_tensor(
                    fixed_mu_value, device=device, dtype=base_rate_init_t.dtype
                )
                if fixed_mu_value is not None
                else torch.zeros(1, device=device, dtype=base_rate_init_t.dtype)
            )
            self.register_buffer("mu_fixed", mu_value)

        self.log_K = nn.Parameter(torch.tensor(math.log(productivity_K_init)))
        self.alpha_e_param = nn.Parameter(torch.tensor(float(productivity_alpha_e_init)))

        self.register_buffer("M_c", torch.tensor(mag_completeness))
        self.register_buffer("M_m", torch.tensor(mag_max))
        self.register_buffer("b", torch.tensor(richter_b))

        self.report_params = report_params
        self.device = device
        self.bg_model = bg_model
        self.reduction = loss_reduction
        self.query_chunk_size = int(query_chunk_size)
        self.history_chunk_size = int(history_chunk_size)
        self.use_grad_checkpoint = bool(use_grad_checkpoint)
        self.to(device)

    @property
    def p(self):
        raw_p = 1.0 + torch.exp(self.log_p_minus_one)
        if not self.enforce_p_gt_one:
            return raw_p
        min_p = torch.as_tensor(
            self.min_omori_p, device=raw_p.device, dtype=raw_p.dtype
        )
        softness = torch.as_tensor(
            self.constraint_softness, device=raw_p.device, dtype=raw_p.dtype
        )
        return self._soft_lower_bound(raw_p, min_p, softness)

    @property
    def c(self):
        return torch.exp(self.log_c)

    @property
    def mu(self):
        if getattr(self, "fix_mu", False):
            return self.mu_fixed
        return torch.exp(self.log_mu)

    @property
    def K(self):
        raw_K = torch.exp(self.log_K)
        if not self.enforce_subcritical:
            return raw_K

        prefactor = self._branching_ratio_prefactor(self.alpha_e)
        eps = torch.finfo(raw_K.dtype).eps
        max_ratio = torch.as_tensor(
            self.max_branching_ratio, device=raw_K.device, dtype=raw_K.dtype
        ).clamp_min(eps)
        K_cap = (max_ratio / prefactor.clamp_min(eps)).clamp_min(eps)
        softness = torch.as_tensor(
            self.constraint_softness, device=raw_K.device, dtype=raw_K.dtype
        ) * K_cap
        softness = softness.clamp_min(eps)
        return self._soft_upper_bound(raw_K, K_cap, softness)

    @property
    def alpha_e(self):
        return self.alpha_e_param

    @staticmethod
    def _soft_upper_bound(
        x: torch.Tensor, upper: torch.Tensor, softness: torch.Tensor
    ) -> torch.Tensor:
        return upper - softness * F.softplus((upper - x) / softness)

    @staticmethod
    def _soft_lower_bound(
        x: torch.Tensor, lower: torch.Tensor, softness: torch.Tensor
    ) -> torch.Tensor:
        return lower + softness * F.softplus((x - lower) / softness)

    def _branching_ratio_prefactor(self, alpha_e_t: torch.Tensor) -> torch.Tensor:
        b = self.b.to(device=alpha_e_t.device, dtype=alpha_e_t.dtype)
        M_c = self.M_c.to(device=alpha_e_t.device, dtype=alpha_e_t.dtype)
        M_m = self.M_m.to(device=alpha_e_t.device, dtype=alpha_e_t.dtype)
        eps = torch.finfo(alpha_e_t.dtype).eps

        delta_m = (M_m - M_c).clamp_min(eps)
        beta = b * math.log(10.0)
        denom = (1.0 - torch.exp(-beta * delta_m)).clamp_min(eps)

        slope = alpha_e_t - beta
        zero = torch.zeros_like(slope)
        near_zero = torch.isclose(slope, zero, rtol=1e-6, atol=1e-8)
        safe_slope = torch.where(near_zero, torch.ones_like(slope), slope)
        integral = torch.where(
            near_zero,
            delta_m,
            (torch.exp(safe_slope * delta_m) - 1.0) / safe_slope,
        )
        pref = (beta / denom) * integral
        return pref.clamp_min(eps)

    @property
    def alpha(self):
        """Equivalent alpha under 10-base form (read-only)."""
        return self.alpha_e / math.log(10.0)

    @property
    def k(self):
        """Equivalent k under 10-base unnormalized kernel (read-only)."""
        return self.K * self.omori_norm_factor

    @property
    def omori_norm_factor(self):
        return (self.p - 1.0) * self.c.pow(self.p - 1.0)

    @property
    def branching_ratio(self) -> float:
        """Infinite-horizon branching ratio under normalized ETAS parameterization."""
        return float(
            _compute_branching_ratio_ogata(
                K=float(self.K.detach().cpu()),
                alpha_e=float(self.alpha_e.detach().cpu()),
                b=float(self.b.detach().cpu()),
                M_min=float(self.M_c.detach().cpu()),
                M_max=float(self.M_m.detach().cpu()),
                M_ref=float(self.M_c.detach().cpu()),
            )
        )

    def effective_branching_ratio(self, t_max: float = 1e4) -> float:
        """Finite-horizon branching ratio on [0, t_max]."""
        p = float(self.p.detach().cpu())
        c = float(self.c.detach().cpu())
        norm = float(self.omori_norm_factor.detach().cpu())
        temporal_mass = norm * float(_omori_int_np(0.0, float(t_max), c, p))
        return self.branching_ratio * temporal_mass

    @staticmethod
    def _torch_omori_int(T1, T2, c, p):
        c_t = torch.as_tensor(c, device=T1.device)
        p_t = torch.as_tensor(p, device=T1.device)
        dtype = torch.promote_types(torch.promote_types(T1.dtype, c_t.dtype), p_t.dtype)

        T1_t = T1.to(dtype=dtype)
        T2_t = T2.to(dtype=dtype)
        c_t = c_t.to(dtype=dtype)
        p_t = p_t.to(dtype=dtype)

        one = torch.ones((), device=T1.device, dtype=dtype)
        if torch.isclose(p_t, one):
            return torch.log(T2_t + c_t) - torch.log(T1_t + c_t)
        one_minus_p = one - p_t
        return ((T2_t + c_t).pow(one_minus_p) - (T1_t + c_t).pow(one_minus_p)) / one_minus_p

    @staticmethod
    def _torch_gen_mag(shape, b, M_min, M_max, device, dtype):
        u = torch.rand(shape, device=device, dtype=dtype)
        return (-1.0 / b) * torch.log10(
            -u * (10 ** (-b * M_min) - 10 ** (-b * M_max)) + 10 ** (-b * M_min)
        )

    @staticmethod
    def _torch_omori_inv(T1, T2, c, p, size, t_max, device, dtype):
        c_t = torch.as_tensor(c, device=device, dtype=dtype)
        p_t = torch.as_tensor(p, device=device, dtype=dtype)
        u = torch.rand(size, device=device, dtype=dtype)

        zero = torch.zeros((), device=device, dtype=dtype)
        tmax_t = torch.tensor(t_max, device=device, dtype=dtype)
        T1 = torch.clamp(T1, min=0.0, max=float(t_max))
        T2 = torch.clamp(T2, min=0.0, max=float(t_max))
        if torch.any(T2 < T1):
            raise ValueError("T2 must be >= T1 after clipping to [0, t_max].")
        F0 = ETASZhuang._torch_omori_int(zero, tmax_t, c_t, p_t)

        F1 = ETASZhuang._torch_omori_int(zero, T1, c_t, p_t) / F0
        F2 = ETASZhuang._torch_omori_int(zero, T2, c_t, p_t) / F0
        u_prime = u * (F2 - F1) + F1

        one = torch.tensor(1.0, device=device, dtype=dtype)
        if torch.isclose(p_t, one):
            log_term = torch.log(tmax_t + c_t) - torch.log(c_t)
            return c_t * torch.exp(u_prime * log_term) - c_t

        one_minus_p = one - p_t
        base = u_prime * F0 * one_minus_p + c_t.pow(one_minus_p)
        return base.pow(1.0 / one_minus_p) - c_t

    @staticmethod
    def _resolve_chunk_size(total: int, configured: int) -> int:
        if configured and configured > 0:
            return max(1, min(int(configured), int(total)))
        return max(1, int(total))

    @staticmethod
    def _iter_chunks(total: int, chunk_size: int):
        for start in range(0, int(total), int(chunk_size)):
            end = min(start + int(chunk_size), int(total))
            yield start, end

    def _zhuang_history_contrib(
        self,
        t_query_chunk: torch.Tensor,
        t_hist_chunk: torch.Tensor,
        productivity_hist_chunk: torch.Tensor,
        survival_hist_chunk: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = t_query_chunk.unsqueeze(-1) - t_hist_chunk.unsqueeze(-2)
        prev_mask = (delta_t > 0) & survival_hist_chunk.unsqueeze(-2)
        omori = (delta_t.clamp_min(0.0) + self.c).pow(-self.p)
        return (omori * productivity_hist_chunk.unsqueeze(-2) * prev_mask).sum(-1)

    def _maybe_checkpoint_history_contrib(
        self,
        *,
        t_query_chunk: torch.Tensor,
        t_hist_chunk: torch.Tensor,
        productivity_hist_chunk: torch.Tensor,
        survival_hist_chunk: torch.Tensor,
    ) -> torch.Tensor:
        if self.use_grad_checkpoint and self.training:
            return checkpoint(
                self._zhuang_history_contrib,
                t_query_chunk,
                t_hist_chunk,
                productivity_hist_chunk,
                survival_hist_chunk,
                use_reentrant=False,
            )
        return self._zhuang_history_contrib(
            t_query_chunk,
            t_hist_chunk,
            productivity_hist_chunk,
            survival_hist_chunk,
        )

    def _intensity_from_history(
        self,
        *,
        t_query: torch.Tensor,
        t_history: torch.Tensor,
        productivity: torch.Tensor,
        survival_mask_bool: torch.Tensor,
        query_chunk_size: int,
        history_chunk_size: int,
    ) -> torch.Tensor:
        s_total = t_query.size(1)
        l_total = t_history.size(1)
        out = torch.zeros_like(t_query) + self.mu
        query_chunk_size = self._resolve_chunk_size(s_total, query_chunk_size)
        history_chunk_size = self._resolve_chunk_size(l_total, history_chunk_size)

        for q_start, q_end in self._iter_chunks(s_total, query_chunk_size):
            t_query_chunk = t_query[:, q_start:q_end]
            intensity_chunk = torch.zeros_like(t_query_chunk) + self.mu

            for h_start, h_end in self._iter_chunks(l_total, history_chunk_size):
                t_hist = t_history[:, h_start:h_end]
                productivity_hist = productivity[:, h_start:h_end]
                survival_hist = survival_mask_bool[:, h_start:h_end]
                intensity_chunk = intensity_chunk + self._maybe_checkpoint_history_contrib(
                    t_query_chunk=t_query_chunk,
                    t_hist_chunk=t_hist,
                    productivity_hist_chunk=productivity_hist,
                    survival_hist_chunk=survival_hist,
                )

            out[:, q_start:q_end] = intensity_chunk
        return out

    def nll_loss(
        self,
        batch: Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        reduction = self.reduction if reduction is None else reduction

        t = batch.arrival_times
        survival_mask = get_mask(
            batch.inter_times,
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx,
        )

        t_select, intensity_mask = masked_select_per_row(t, batch.nll_event_mask)
        amp = self.K * torch.exp(self.alpha_e * (batch.mag - self.M_c))
        productivity = self.omori_norm_factor * amp
        intensity = self._intensity_from_history(
            t_query=t_select,
            t_history=t,
            productivity=productivity,
            survival_mask_bool=survival_mask.bool(),
            query_chunk_size=self.query_chunk_size,
            history_chunk_size=self.history_chunk_size,
        )

        if self.bg_model is not None:
            f_intensity = self.bg_model.intensity(batch, t_query=t_select)
            intensity += f_intensity

        intensity_safe = intensity.clamp_min(eps)
        log_intensity = (torch.log(intensity_safe) * intensity_mask).sum(-1)

        t_end = batch.t_end.unsqueeze(-1)
        t_nll_start = batch.t_nll_start.unsqueeze(-1)
        dt_end = (t_end - t).clamp_min(0.0)
        dt_start = (t_nll_start - t).clamp_min(0.0)
        omori_int = self._torch_omori_int(dt_start, dt_end, self.c, self.p)
        omori_int = omori_int * survival_mask
        kernel_mass = self.omori_norm_factor * omori_int
        integral = (kernel_mass * amp).sum(-1)
        integral += (batch.t_end - batch.t_nll_start) * self.mu

        if self.bg_model is not None:
            f_integral = self.bg_model.intensity_integral(batch)
            integral += f_integral

        nll_time = -log_intensity + integral

        bg_kl = None
        if self.bg_model is not None and hasattr(self.bg_model, "kl_term"):
            bg_kl = self.bg_model.kl_term(batch, eps=eps)
        nll_total = nll_time if bg_kl is None else (nll_time + bg_kl)

        out_dict = {
            "time": nll_time,
            "total": nll_total,
        }
        if bg_kl is not None:
            out_dict["bg_kl"] = bg_kl

        out = self.reduce_nll_dict(
            out_dict,
            batch,
            reduction=reduction,
            eps=eps,
        )
        if return_dict:
            return out
        return out["total"]

    def h_intensity(self, batch: Batch, t_query: torch.Tensor = None) -> torch.Tensor:
        t = batch.arrival_times[:, :-1]
        mag = batch.mag[:, :-1]
        survival_mask = get_mask(
            batch.inter_times[:, :-1],
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx - 1,
        )
        if t_query is None:
            t_query = t
        amp = self.K * torch.exp(self.alpha_e * (mag - self.M_c))
        productivity = self.omori_norm_factor * amp
        return self._intensity_from_history(
            t_query=t_query,
            t_history=t,
            productivity=productivity,
            survival_mask_bool=survival_mask.bool(),
            query_chunk_size=self.query_chunk_size,
            history_chunk_size=self.history_chunk_size,
        )

    def prefix_h_integral(
        self,
        *,
        t_all: torch.Tensor,
        mag_all: torch.Tensor,
        t0: torch.Tensor,
        t_query: torch.Tensor,
        query_block_size: int = 256,
    ) -> torch.Tensor:
        """Compute integral of ``h(s)`` from ``t0`` to each value in ``t_query``."""
        one_minus_p = 1.0 - self.p
        productivity = self.k * torch.exp(self.alpha_e * (mag_all - self.M_c))

        dt_start = (t0 - t_all).clamp_min(0.0)
        dt_start_term = (dt_start + self.c).pow(one_minus_p)

        out = torch.empty_like(t_query)
        block_size = int(max(1, query_block_size))
        for st in range(0, int(t_query.numel()), block_size):
            t_chunk = t_query[st : st + block_size]
            dt_end = (t_chunk.unsqueeze(1) - t_all.unsqueeze(0)).clamp_min(0.0)
            omori_int = ((dt_end + self.c).pow(one_minus_p) - dt_start_term.unsqueeze(0)) / one_minus_p
            int_h = (omori_int * productivity.unsqueeze(0)).sum(dim=1)
            int_h = int_h + (t_chunk - t0) * self.mu
            out[st : st + block_size] = int_h
        return out

    def training_step(self, batch, batch_idx):
        loss = self.nll_loss(batch).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        if self.report_params:
            for param_name in ["p", "c", "mu", "K", "alpha_e"]:
                self.log(
                    f"params/{param_name}",
                    getattr(self, param_name).item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch.batch_size,
                )
        return loss

    def sample_thinning(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[Sequence] = None,
        random_state: int = 123,
        max_length: int = 50_000,
        n_jobs: int = -1,
        return_sequences: bool = False,
        verbose: bool = False,
    ) -> Union[Batch, List[Sequence]]:
        p = float(self.p.detach().cpu())
        c = float(self.c.detach().cpu())
        mu = float(self.mu.detach().cpu())
        K = float(self.K.detach().cpu())
        alpha_e = float(self.alpha_e.detach().cpu())
        M_c = float(self.M_c.detach().cpu())
        M_m = float(self.M_m.detach().cpu())
        norm = float(self.omori_norm_factor.detach().cpu())

        branch = self.branching_ratio
        if branch > 1:
            raise ValueError(
                f"The process is explosive: branching ratio {branch:.2f} is > 1."
            )

        def get_intensity(t: float, t_past: np.ndarray, mag_past: np.ndarray):
            omori = (t - t_past + c) ** (-p)
            productivity = K * np.exp(alpha_e * (mag_past - M_c)) * norm
            return (omori * productivity).sum() + mu

        def bernoulli(success_proba: float):
            if success_proba < 0 or success_proba > 1:
                raise ValueError("Success probability must be in [0, 1] range")
            return np.random.uniform() < success_proba

        def sample_single_seq(t_start_single, seed):
            np.random.seed(seed)
            if past_seq is not None:
                past_tau = past_seq.inter_times.cpu().numpy().copy()
                arrival_times = np.cumsum(past_tau[:-1]) + past_seq.t_start
                magnitudes = past_seq.mag.cpu().numpy().copy()
                t_start_local = float(past_seq.t_end)
            else:
                arrival_times = np.array([], dtype=np.float64)
                magnitudes = np.array([], dtype=np.float64)
                t_start_local = t_start_single

            t_current = t_start_local
            t_end = t_start_local + duration
            upper_bound = get_intensity(t_current, arrival_times, magnitudes)
            if upper_bound <= 0:
                return dict(
                    inter_times=np.array([duration], dtype=np.float64),
                    t_start=t_start_local,
                    mag=np.array([], dtype=np.float64),
                )
            tau_current = 0.0
            inter_times = []

            while True:
                tau = np.random.exponential(1.0 / upper_bound)
                tau_current += tau
                t_current = t_current + tau
                if t_current > t_end:
                    break

                lambda_current = get_intensity(t_current, arrival_times, magnitudes)
                p_accept = lambda_current / upper_bound
                if verbose:
                    logger.info(
                        "Candidate event at %.3f, acceptance prob = %.2f",
                        t_current,
                        p_accept,
                    )
                if bernoulli(p_accept):
                    arrival_times = np.append(arrival_times, t_current)
                    magnitudes = np.append(
                        magnitudes,
                        gen_mag(
                            b=float(self.b.detach().cpu()),
                            M_min=M_c,
                            M_max=M_m,
                        ),
                    )
                    inter_times.append(tau_current)
                    tau_current = 0.0

                upper_bound = get_intensity(t_current, arrival_times, magnitudes)
                upper_bound = max(upper_bound, np.finfo(np.float64).tiny)
                if len(inter_times) > max_length:
                    logger.warning(
                        "Stopping generation since max_length exceeded (likely explosive process)."
                    )
                    return None

            inter_times = np.append(inter_times, max(duration - np.sum(inter_times), 0.0))
            valid_idx = (arrival_times > t_start_local) & (arrival_times <= t_end)
            return dict(
                inter_times=inter_times,
                t_start=t_start_local,
                mag=magnitudes[valid_idx],
            )

        sequences = []
        while len(sequences) < batch_size:
            num_seq_to_generate = batch_size - len(sequences)
            new_sequences = Parallel(n_jobs=n_jobs)(
                delayed(sample_single_seq)(t_start, seed)
                for seed in trange(random_state, num_seq_to_generate + random_state)
            )
            filtered = [Sequence(**seq) for seq in new_sequences if seq is not None]
            sequences.extend(filtered)

        return sequences if return_sequences else Batch.from_list(sequences)

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[Sequence] = None,
        random_state: int = 123,
        max_length: Optional[int] = 50_000,
        t_max: float = 1e10,
        n_jobs: int = -1,
        return_sequences: bool = False,
    ) -> Union[Batch, List[Sequence]]:
        p = float(self.p.detach().cpu())
        c = float(self.c.detach().cpu())
        mu = float(self.mu.detach().cpu())
        K = float(self.K.detach().cpu())
        alpha_e = float(self.alpha_e.detach().cpu())
        b = float(self.b.detach().cpu())
        M_c = float(self.M_c.detach().cpu())
        M_m = float(self.M_m.detach().cpu())
        norm = float(self.omori_norm_factor.detach().cpu())

        if past_seq is not None:
            t_start = float(past_seq.t_end)

        # branch = self.effective_branching_ratio(t_max=t_max)
        # if branch > 1:
        #     raise ValueError(
        #         f"The process is explosive: branching ratio {branch:.2f} is > 1."
        #     )

        def sample_single_seq(seed, bg_times: Optional[np.ndarray] = None):
            np.random.seed(seed)
            if past_seq is not None:
                past_tau = past_seq.inter_times.cpu().numpy().copy()
                arrival_times = np.cumsum(past_tau[:-1]) + past_seq.t_start
                magnitudes = past_seq.mag.cpu().numpy().copy()
                parent_catalog = np.column_stack((arrival_times, magnitudes))
            else:
                parent_catalog = []

            t_end = t_start + duration

            Nback = poisson.rvs(mu * duration) if self.bg_model is None else 0
            if self.bg_model is None:
                background_events = [np.random.uniform(t_start, t_end, Nback).T]
                background_events.append(gen_mag(shape=Nback, b=b, M_min=M_c, M_max=M_m))
                background_catalog = np.column_stack(background_events)
            else:
                if bg_times is None:
                    times_list = self.bg_model.sample_nhpp_inverse(
                        B=1,
                        t0=torch.tensor([t_start], device=self.device),
                        dt=torch.tensor([duration], device=self.device),
                        sample_sequence=True,
                        mu=float(self.mu.item()),
                    )
                    t_back = np.array(times_list[0], dtype=np.float64)
                else:
                    t_back = bg_times
                Nback = t_back.size
                if Nback > 0:
                    m_back = gen_mag(shape=Nback, b=b, M_min=M_c, M_max=M_m)
                    background_catalog = np.column_stack([t_back, m_back]).astype(np.float64)
                else:
                    background_catalog = np.empty((0, 2), dtype=np.float64)

            parent_catalog = (
                np.vstack((parent_catalog, background_catalog))
                if len(parent_catalog) > 0
                else background_catalog
            )

            offspring_catalog = []
            whole_catalog = []
            num_generated = Nback

            while True:
                whole_catalog.append(parent_catalog)

                amp = K * np.exp(alpha_e * (parent_catalog[:, -1] - M_c))
                TAU1 = np.clip(t_start - parent_catalog[:, 0], a_min=0.0, a_max=t_max)
                TAU2 = np.clip(t_end - parent_catalog[:, 0], a_min=0.0, a_max=t_max)
                interval_mass = norm * _omori_int_np(TAU1, TAU2, c, p)
                prod_in_interval = amp * interval_mass
                prod_in_interval = np.clip(prod_in_interval, a_min=0.0, a_max=None)

                N_aftershock = np.random.poisson(prod_in_interval)
                I = N_aftershock != 0
                short_parent_catalog = parent_catalog[I, :]
                short_N_aftershock = N_aftershock[I]

                for ieq, iNaft, itau1, itau2 in zip(
                    short_parent_catalog, short_N_aftershock, TAU1[I], TAU2[I]
                ):
                    t_parent = ieq[0]
                    dti = _omori_inv_np(itau1, itau2, c, p, size=iNaft, t_max=t_max)
                    t_aftershock = t_parent + dti

                    m_aftershock = gen_mag(iNaft, b=b, M_min=M_c, M_max=M_m)
                    aftershock_catalog = np.column_stack([t_aftershock, m_aftershock])
                    offspring_catalog.append(aftershock_catalog)

                if not offspring_catalog:
                    break

                offspring_catalog = np.vstack(offspring_catalog)
                parent_catalog = offspring_catalog
                offspring_catalog = []
                num_generated += len(parent_catalog)
                if max_length is not None and num_generated > max_length:
                    logger.warning("Exceeded %s events, discarding sequence", max_length)
                    return None

            whole_catalog = np.vstack(whole_catalog)
            whole_catalog = whole_catalog[whole_catalog[:, 0].argsort()]

            arrival_times = whole_catalog[:, 0]
            magnitudes = whole_catalog[:, -1]
            valid_idx = (arrival_times > t_start) & (arrival_times <= t_end)
            fc_arrival_times = arrival_times[valid_idx]
            fc_magnitudes = magnitudes[valid_idx]

            inter_times = np.diff(fc_arrival_times, prepend=t_start, append=t_end)
            return Sequence(
                inter_times=inter_times,
                t_start=t_start,
                mag=fc_magnitudes,
            )

        sequences = []
        np.random.seed(random_state)
        starting_seed = np.random.randint(0, 100000)
        while len(sequences) < batch_size:
            num_seq_to_generate = batch_size - len(sequences)

            bg_times_bulk: Optional[List[np.ndarray]] = None
            if self.bg_model is not None:
                times_list = self.bg_model.sample_nhpp_inverse(
                    B=num_seq_to_generate,
                    t0=torch.full((num_seq_to_generate,), t_start, device=self.device),
                    dt=torch.full((num_seq_to_generate,), duration, device=self.device),
                    sample_sequence=True,
                    mu=float(self.mu.item()),
                )

                def _to_numpy(x):
                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy().astype(np.float64)
                    return np.asarray(x, dtype=np.float64)

                bg_times_bulk = [_to_numpy(t_) for t_ in times_list]

            new_sequences = Parallel(n_jobs=n_jobs)(
                delayed(sample_single_seq)(seed, None if bg_times_bulk is None else bg_times_bulk[i])
                for i, seed in enumerate(trange(starting_seed, starting_seed + num_seq_to_generate))
            )
            filtered = [seq for seq in new_sequences if seq is not None]
            sequences.extend(filtered)
            starting_seed += num_seq_to_generate

        return sequences if return_sequences else Batch.from_list(sequences)

    def sample_gpu_parallel(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional["Sequence"] = None,
        max_length: Optional[int] = 50_000,
        t_max: float = 1e10,
        return_sequences: bool = False,
        dtype: torch.dtype = torch.float64,
    ) -> Union["Batch", List["Sequence"]]:
        device = self.device
        if device is None:
            device = next(self.parameters()).device

        p = self.p.to(device=device, dtype=dtype)
        c = self.c.to(device=device, dtype=dtype)
        mu = self.mu.to(device=device, dtype=dtype)
        K = self.K.to(device=device, dtype=dtype)
        alpha_e = self.alpha_e.to(device=device, dtype=dtype)
        b = self.b.to(device=device, dtype=dtype)

        M_c = self.M_c.to(device=device, dtype=dtype)
        M_m = self.M_m.to(device=device, dtype=dtype)

        branch = self.effective_branching_ratio(t_max=t_max)
        if branch > 1:
            raise ValueError(f"The process is explosive: branching ratio {branch:.2f} is > 1.")

        if past_seq is not None:
            t0 = float(past_seq.t_end)
        else:
            t0 = float(t_start)

        t_start_vec = torch.full((batch_size,), t0, device=device, dtype=dtype)
        t_end_vec = t_start_vec + float(duration)
        norm = (p - 1.0) * c.pow(p - 1.0)

        catalogs = []
        if past_seq is not None:
            past_tau = past_seq.inter_times.cpu().numpy().copy()
            tau_tensor = torch.tensor(past_tau[:-1], dtype=dtype, device=device)
            arrival_single = torch.cumsum(tau_tensor, dim=0) + float(past_seq.t_start)
            mags_single = torch.tensor(past_seq.mag, dtype=dtype, device=device)

            arrival = arrival_single.expand(batch_size, -1).reshape(-1)
            mags = mags_single.expand(batch_size, -1).reshape(-1)
            bid = torch.arange(batch_size, device=device, dtype=torch.int64).repeat_interleave(
                arrival_single.numel()
            )
            catalogs.append((arrival, mags, bid))

        lam = mu * duration
        N_back = torch.poisson(lam.expand(batch_size)).to(torch.int64)
        max_back = int(N_back.max().item()) if N_back.numel() > 0 else 0
        if max_back > 0:
            u_time = torch.rand((batch_size, max_back), device=device, dtype=dtype)
            t_back = t_start_vec[:, None] + u_time * float(duration)
            m_back = self._torch_gen_mag(
                (batch_size, max_back),
                b=float(b.item()),
                M_min=float(M_c.item()),
                M_max=float(M_m.item()),
                device=device,
                dtype=dtype,
            )
            mask_back = (torch.arange(max_back, device=device)[None, :] < N_back[:, None])
            t_back_flat = t_back[mask_back]
            m_back_flat = m_back[mask_back]
            bid_flat = torch.repeat_interleave(
                torch.arange(batch_size, device=device, dtype=torch.int64),
                N_back,
            )
            catalogs.append((t_back_flat, m_back_flat, bid_flat))

        if len(catalogs) == 0:
            empty = [
                Sequence(
                    inter_times=np.array([duration], dtype=np.float64),
                    t_start=float(t_start_vec[i].item()),
                    mag=np.array([], dtype=np.float64),
                )
                for i in range(batch_size)
            ]
            return empty if return_sequences else Batch.from_list(empty)

        t_parent = torch.cat([c_[0] for c_ in catalogs], dim=0)
        m_parent = torch.cat([c_[1] for c_ in catalogs], dim=0)
        b_parent = torch.cat([c_[2] for c_ in catalogs], dim=0)

        parent_catalog = (t_parent, m_parent, b_parent)
        events_all = [parent_catalog]
        total_events = t_parent.numel()

        while True:
            t_parent, m_parent, b_parent = parent_catalog
            TAU1 = (t_start_vec[b_parent] - t_parent).clamp(min=0.0, max=t_max)
            TAU2 = (t_end_vec[b_parent] - t_parent).clamp(min=0.0, max=t_max)

            amp = K * torch.exp(alpha_e * (m_parent - M_c))
            interval_mass = norm * self._torch_omori_int(TAU1, TAU2, c, p)
            prod_in_interval = (amp * interval_mass).clamp_min(0.0)

            N_child = torch.poisson(prod_in_interval).to(torch.int64)
            has_child = N_child > 0
            if not has_child.any():
                break

            parent_idx = torch.nonzero(has_child, as_tuple=False).squeeze(-1)
            repeat_idx = torch.repeat_interleave(parent_idx, N_child[has_child])
            n_child_total = repeat_idx.numel()

            t_sel = t_parent[repeat_idx]
            bid_sel = b_parent[repeat_idx]
            tau1_sel = TAU1[repeat_idx]
            tau2_sel = TAU2[repeat_idx]

            dti = self._torch_omori_inv(
                tau1_sel,
                tau2_sel,
                c,
                p,
                size=(n_child_total,),
                t_max=t_max,
                device=device,
                dtype=dtype,
            )
            t_child = t_sel + dti

            m_child = self._torch_gen_mag(
                (n_child_total,),
                b=float(b.item()),
                M_min=float(M_c.item()),
                M_max=float(M_m.item()),
                device=device,
                dtype=dtype,
            )

            if (max_length is not None) and (total_events + n_child_total > max_length):
                break

            child_catalog = (t_child, m_child, bid_sel)
            events_all.append(child_catalog)
            parent_catalog = child_catalog
            total_events += n_child_total

        all_t = torch.cat([ev[0] for ev in events_all], dim=0)
        all_m = torch.cat([ev[1] for ev in events_all], dim=0)
        all_b = torch.cat([ev[2] for ev in events_all], dim=0)

        seq_list: List[Sequence] = []
        for bidx in range(batch_size):
            mask = all_b == bidx
            if not mask.any():
                seq_list.append(
                    Sequence(
                        inter_times=np.array([duration], dtype=np.float64),
                        t_start=float(t_start_vec[bidx].item()),
                        mag=np.array([], dtype=np.float64),
                    )
                )
                continue

            times = all_t[mask]
            mags = all_m[mask]

            in_win = (times > t_start_vec[bidx]) & (times <= t_end_vec[bidx])
            times = times[in_win]
            mags = mags[in_win]

            if times.numel() == 0:
                seq_list.append(
                    Sequence(
                        inter_times=np.array([duration], dtype=np.float64),
                        t_start=float(t_start_vec[bidx].item()),
                        mag=np.array([], dtype=np.float64),
                    )
                )
                continue

            sort_idx = torch.argsort(times)
            times = times[sort_idx]
            mags = mags[sort_idx]

            inter_times = torch.diff(
                torch.cat(
                    [
                        t_start_vec[bidx : bidx + 1],
                        times,
                        t_end_vec[bidx : bidx + 1],
                    ]
                )
            )

            seq_list.append(
                Sequence(
                    inter_times=inter_times.detach().cpu().numpy().astype(np.float64),
                    t_start=float(t_start_vec[bidx].item()),
                    mag=mags.detach().cpu().numpy().astype(np.float64),
                )
            )

        return seq_list if return_sequences else Batch.from_list(seq_list)

    def print_params(self):
        params = {
            "p": self.p.detach().cpu().item(),
            "c": self.c.detach().cpu().item(),
            "mu": self.mu.detach().cpu().item(),
            "K": self.K.detach().cpu().item(),
            "alpha_e": self.alpha_e.detach().cpu().item(),
            "k_eq": self.k.detach().cpu().item(),
            "alpha_eq": self.alpha.detach().cpu().item(),
            "b": float(self.b.detach().cpu().item()),
            "M_c": float(self.M_c.detach().cpu().item()),
            "M_m": float(self.M_m.detach().cpu().item()),
        }
        print("ETAS (normalized/Zhuang) model parameters:")
        for name, value in params.items():
            print(f"  {name} = {value}")

    def set_params(
        self,
        *,
        p: Optional[float] = None,
        c: Optional[float] = None,
        mu: Optional[float] = None,
        K: Optional[float] = None,
        alpha_e: Optional[float] = None,
        b: Optional[float] = None,
    ) -> None:
        """Set ETAS parameters under normalized ETAS parameterization."""
        with torch.no_grad():
            if p is not None:
                if p <= 1.0:
                    raise ValueError("Normalized ETAS parameterization requires p > 1.")
                p_t = _to_tensor(p - 1.0, self.log_p_minus_one)
                self.log_p_minus_one.copy_(torch.log(p_t))

            if c is not None:
                if c <= 0:
                    raise ValueError("c must be strictly positive.")
                c_t = _to_tensor(c, self.log_c)
                self.log_c.copy_(torch.log(c_t))

            if mu is not None:
                if mu < 0:
                    raise ValueError("mu must be non-negative.")
                mu_t = _to_tensor(mu, self.log_mu)
                if not self.fix_mu:
                    self.log_mu.copy_(torch.log(mu_t.clamp_min(torch.finfo(mu_t.dtype).tiny)))
                else:
                    self.mu_fixed.copy_(mu_t)

            if K is not None:
                if K < 0:
                    raise ValueError("K must be non-negative.")
                K_t = _to_tensor(K, self.log_K)
                self.log_K.copy_(torch.log(K_t.clamp_min(torch.finfo(K_t.dtype).tiny)))

            if alpha_e is not None:
                a_t = _to_tensor(alpha_e, self.alpha_e_param)
                self.alpha_e_param.copy_(a_t)

            if b is not None:
                if b <= 0:
                    raise ValueError("richter b must be strictly positive.")
                b_t = _to_tensor(b, self.b)
                self.b.copy_(b_t)


def masked_select_per_row(matrix, mask):
    """Perform masked select on each row and return a padded tensor."""
    assert matrix.shape == mask.shape and matrix.ndim == 2
    selected_rows = []
    for matrix_row, mask_row in zip(matrix, mask.bool()):
        selected_rows.append(matrix_row.masked_select(mask_row))

    new_matrix = pad_sequence(selected_rows)
    new_mask = pad_sequence([torch.ones_like(s) for s in selected_rows])
    return new_matrix, new_mask.float()
