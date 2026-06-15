"""Shared utilities for ETAS-family temporal point process models."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def to_tensor_like(value, reference: torch.Tensor) -> torch.Tensor:
    """Convert ``value`` to a tensor on the same device/dtype as ``reference``."""
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def gen_magnitude(shape=1, b=1, m_min=0, m_max=10):
    """Draw samples from the truncated Gutenberg-Richter magnitude law."""
    u = np.random.random(shape)
    mag = (
        -1.0
        / b
        * np.log10(
            -u * (10.0 ** (-b * m_min) - 10.0 ** (-b * m_max))
            + 10.0 ** (-b * m_min)
        )
    )
    return mag


def torch_gen_magnitude(shape, b, m_min, m_max, device, dtype):
    """Torch version of truncated Gutenberg-Richter inverse sampling."""
    u = torch.rand(shape, device=device, dtype=dtype)
    return (-1.0 / b) * torch.log10(
        -u * (10.0 ** (-b * m_min) - 10.0 ** (-b * m_max))
        + 10.0 ** (-b * m_min)
    )


def omori_integral_np(t1, t2, c, p):
    """Integral of ``(t + c)^(-p)`` from ``t1`` to ``t2`` in NumPy space."""
    if np.isclose(p, 1.0):
        return np.log(t2 + c) - np.log(t1 + c)
    return ((t2 + c) ** (1.0 - p) - (t1 + c) ** (1.0 - p)) / (1.0 - p)


def sample_omori_truncated_np(t1, t2, c, p, size=1, t_max=1e10):
    """Sample Omori lags on ``[t1, t2]`` by inverse transform."""
    if t_max <= 0:
        raise ValueError("t_max must be positive.")
    if c <= 0:
        raise ValueError("c must be positive.")

    t1 = np.clip(np.asarray(t1, dtype=np.float64), 0.0, t_max)
    t2 = np.clip(np.asarray(t2, dtype=np.float64), 0.0, t_max)
    if np.any(t2 < t1):
        raise ValueError("T2 must be >= T1 after clipping to [0, t_max].")

    u = np.random.random(size=size)
    total_mass = omori_integral_np(0.0, t_max, c, p)
    cdf_t1 = omori_integral_np(0.0, t1, c, p) / total_mass
    cdf_t2 = omori_integral_np(0.0, t2, c, p) / total_mass
    u_prime = u * (cdf_t2 - cdf_t1) + cdf_t1

    if np.isclose(p, 1.0):
        return c * np.exp(u_prime * (np.log(t_max + c) - np.log(c))) - c

    one_minus_p = 1.0 - p
    base = u_prime * total_mass * one_minus_p + c ** one_minus_p
    base = np.maximum(base, np.finfo(np.float64).tiny)
    return base ** (1.0 / one_minus_p) - c


def torch_omori_integral(t1, t2, c, p):
    """Integral of ``(t + c)^(-p)`` from ``t1`` to ``t2`` in torch space."""
    c_t = torch.as_tensor(c, device=t1.device)
    p_t = torch.as_tensor(p, device=t1.device)
    dtype = torch.promote_types(torch.promote_types(t1.dtype, c_t.dtype), p_t.dtype)

    t1 = t1.to(dtype=dtype)
    t2 = t2.to(dtype=dtype)
    c_t = c_t.to(dtype=dtype)
    p_t = p_t.to(dtype=dtype)

    one = torch.ones((), device=t1.device, dtype=dtype)
    if torch.isclose(p_t, one):
        return torch.log(t2 + c_t) - torch.log(t1 + c_t)
    one_minus_p = one - p_t
    return ((t2 + c_t).pow(one_minus_p) - (t1 + c_t).pow(one_minus_p)) / one_minus_p


def sample_omori_truncated_torch(t1, t2, c, p, size, t_max, device, dtype):
    """Torch inverse-transform sampling from Omori lags on ``[t1, t2]``."""
    c_t = torch.as_tensor(c, device=device, dtype=dtype)
    p_t = torch.as_tensor(p, device=device, dtype=dtype)
    u = torch.rand(size, device=device, dtype=dtype)

    zero = torch.zeros((), device=device, dtype=dtype)
    tmax_t = torch.tensor(t_max, device=device, dtype=dtype)
    t1 = torch.clamp(t1, min=0.0, max=float(t_max))
    t2 = torch.clamp(t2, min=0.0, max=float(t_max))
    if torch.any(t2 < t1):
        raise ValueError("T2 must be >= T1 after clipping to [0, t_max].")

    total_mass = torch_omori_integral(zero, tmax_t, c_t, p_t)
    cdf_t1 = torch_omori_integral(zero, t1, c_t, p_t) / total_mass
    cdf_t2 = torch_omori_integral(zero, t2, c_t, p_t) / total_mass
    u_prime = u * (cdf_t2 - cdf_t1) + cdf_t1

    one = torch.tensor(1.0, device=device, dtype=dtype)
    if torch.isclose(p_t, one):
        log_term = torch.log(tmax_t + c_t) - torch.log(c_t)
        return c_t * torch.exp(u_prime * log_term) - c_t

    one_minus_p = one - p_t
    base = u_prime * total_mass * one_minus_p + c_t.pow(one_minus_p)
    return base.pow(1.0 / one_minus_p) - c_t


def omori_history_contribution(
    *,
    t_query_chunk: torch.Tensor,
    t_hist_chunk: torch.Tensor,
    productivity_hist_chunk: torch.Tensor,
    survival_hist_chunk: torch.Tensor,
    c: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    """Compute chunked ETAS triggering intensity from historical parents."""
    delta_t = t_query_chunk.unsqueeze(-1) - t_hist_chunk.unsqueeze(-2)
    previous_event_mask = (delta_t > 0) & survival_hist_chunk.unsqueeze(-2)
    omori = (delta_t.clamp_min(0.0) + c).pow(-p)
    return (
        omori * productivity_hist_chunk.unsqueeze(-2) * previous_event_mask
    ).sum(-1)


def soft_upper_bound(x: torch.Tensor, upper: torch.Tensor, softness: torch.Tensor) -> torch.Tensor:
    """Smooth approximation of ``min(x, upper)``."""
    return upper - softness * F.softplus((upper - x) / softness)


def soft_lower_bound(x: torch.Tensor, lower: torch.Tensor, softness: torch.Tensor) -> torch.Tensor:
    """Smooth approximation of ``max(x, lower)``."""
    return lower + softness * F.softplus((x - lower) / softness)


def resolve_chunk_size(total: int, configured: int) -> int:
    """Resolve ``0`` as full length and positive values as capped chunks."""
    if configured and configured > 0:
        return max(1, min(int(configured), int(total)))
    return max(1, int(total))


def iter_chunks(total: int, chunk_size: int):
    """Yield ``[start, end)`` chunks covering ``total`` elements."""
    for start in range(0, int(total), int(chunk_size)):
        end = min(start + int(chunk_size), int(total))
        yield start, end


__all__ = [
    "gen_magnitude",
    "iter_chunks",
    "omori_history_contribution",
    "omori_integral_np",
    "resolve_chunk_size",
    "sample_omori_truncated_np",
    "sample_omori_truncated_torch",
    "soft_lower_bound",
    "soft_upper_bound",
    "to_tensor_like",
    "torch_gen_magnitude",
    "torch_omori_integral",
]
