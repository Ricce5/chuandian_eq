# Rotary Embedding for continuous time as noted in our paper
# ref: https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/layers/rotary.py

import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Tuple, Union


def rotate_half(x, interleaved=False):
    """
    Rotate the last dimension by half for rotary embedding.

    Args:
        x: Input tensor whose last dimension is rotary channels.
        interleaved: Whether channels are arranged as interleaved pairs
            (..., d0_even, d0_odd, d1_even, d1_odd, ...).

    Returns:
        Tensor with the same shape as ``x`` after half-rotation.
    """
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return rearrange(torch.stack([-x2, x1], dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    Apply rotary embedding to the rotary sub-dimension of attention states.

    Args:
        x: Tensor of shape ``(batch_size, seqlen, nheads, headdim)``.
        cos: Rotary cosine cache with shape
            ``(seqlen, rotary_dim/2)`` or ``(batch_size, seqlen, rotary_dim/2)``.
        sin: Rotary sine cache with the same shape contract as ``cos``.
        interleaved: Whether rotary channels use interleaved layout.

    Returns:
        Tensor with same shape as ``x`` where only the first ``rotary_dim``
        channels are rotated and the remaining channels are kept unchanged.
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    if not interleaved:
        cos = repeat(cos, "... d -> ... 1 (2 d)")
        sin = repeat(sin, "... d -> ... 1 (2 d)")
    else:
        cos = repeat(cos, "... d -> ... 1 (d 2)")
        sin = repeat(sin, "... d -> ... 1 (d 2)")

    x_rot = x[..., :ro_dim]
    x_pass = x[..., ro_dim:]
    return torch.cat([x_rot * cos + rotate_half(x_rot, interleaved) * sin, x_pass], dim=-1)


def compute_nonzero_center_per_sample(times: torch.Tensor, mode: str = "midpoint") -> torch.Tensor:
    """
    Compute center from non-zero elements in each sample.
    
    Args:
        times: (batch, seqlen) tensor
        mode: 'mean' | 'midpoint'
        
    Returns:
        (batch,) tensor of center values
    """
    assert mode in ["mean", "midpoint"], f"Unsupported mode: {mode}"

    non_zero_mask = times != 0  # (batch, seqlen)
    times_float = times.float()

    if mode == "mean":
        sums = (times_float * non_zero_mask).sum(dim=1)
        counts = non_zero_mask.sum(dim=1).clamp(min=1) 
        center = sums / counts
    elif mode == "midpoint":
        times_with_inf = times_float.clone()
        times_with_inf[~non_zero_mask] = float('inf')
        min_vals, _ = torch.min(times_with_inf, dim=1)
        times_with_ninf = times_float.clone()
        times_with_ninf[~non_zero_mask] = float('-inf')
        max_vals, _ = torch.max(times_with_ninf, dim=1)
        center = (min_vals + max_vals) / 2

        # deal with all-zero case
        all_zero_mask = non_zero_mask.sum(dim=1) == 0
        center[all_zero_mask] = 0 
    return center




class RotaryEmbeddingTime(nn.Module):
    """
    Rotary positional embedding module for continuous/event time.

    This module supports both standard RoPE and scaled RoPE (separate scale
    for query/key paths) using per-sample time centers computed from non-zero
    timestamps.
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        time_center: Optional[float] = None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved
        self.scale_base = scale_base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq.to(device), persistent=False)

        if scale_base is not None:
            scale = (torch.arange(0, dim, 2, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
        else:
            scale = None
        self.register_buffer("scale", scale, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self._center_cached = time_center if time_center is not None else None


    def _update_cos_sin_cache(self, times: torch.Tensor, dtype: torch.dtype, device: torch.device):
        """
        Build or refresh cosine/sine caches from input times.

        Args:
            times: Time tensor with shape ``(seqlen,)`` or ``(batch, seqlen)``.
            dtype: Target dtype for cached trigonometric tensors.
            device: Target device where caches should live.

        Notes:
            When ``scale_base`` is enabled, query/key caches are scaled
            inversely to preserve compatibility with xPos-style scaling.
        """
        # times: (seqlen,) or (batch, seqlen)
        inv_freq = self.inv_freq.to(device)
        if times.ndim == 1:
            # (seqlen,) @ (rotary_dim/2,) -> (seqlen, rotary_dim/2)
            freqs = torch.outer(times, inv_freq)
        elif times.ndim == 2:
            # (batch, seqlen) @ (rotary_dim/2,) -> (batch, seqlen, rotary_dim/2)
            freqs = torch.einsum('bs,d->bsd', times, inv_freq)
        else:
            raise ValueError("`times` should be of shape (seqlen,) or (batch, seqlen)")

        if self.scale_base is None:
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
            self._cos_k_cached = torch.cos(freqs).to(dtype)
            self._sin_k_cached = torch.sin(freqs).to(dtype)
        else:
            center = self._center_cached
            power = (times - center) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "... -> ... 1")
            self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)


    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        times: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply time-aware rotary embedding to query/key tensors.

        Args:
            qkv: One of the following:
                - ``(B, S, 3, H, D)`` fused QKV layout,
                - ``(B, S, H_total, D)`` grouped-head layout,
                - ``(B, S, H, D)`` query-only when ``kv`` is provided.
            kv: Optional KV tensor, shape ``(B, S, 2, H, D)`` or ``(B, S, H, D)``.
            times: Time positions, shape ``(S,)`` or ``(B, S)``.
            seqlen_offset: Offset used in streaming/incremental decoding.
            max_seqlen: Optional max sequence length for center scaling.
            num_heads_q: Required when ``qkv`` uses grouped-head layout.

        Returns:
            - If ``kv is None``: rotated tensor with same layout as ``qkv``.
            - If ``kv is not None``: tuple ``(q_rot, kv_rot)``.

        Raises:
            AssertionError: If ``times`` is not provided.
        """
        # print(f"seqlen_offset {seqlen_offset} max_seqlen {max_seqlen}")
        if times.ndim == 1:
            times = times[None, :] 
        device = qkv.device
        dtype = qkv.dtype
        seq_len = times.shape[1]
        # only update center when seqlen_offset == 0
        # scale the center if max_seqlen is provided
        if seqlen_offset == 0:
            self._center_cached = compute_nonzero_center_per_sample(times, 'midpoint')[:, None]
            if max_seqlen is not None:
                t_min = times[:, [0]]
                scale = max_seqlen  / seq_len
                diff = self._center_cached - t_min 
                diff = torch.where(diff > 0, diff, torch.ones_like(diff))
                self._center_cached = diff * scale + t_min
        assert times is not None, "times must be provided for rotary embedding"
        self._update_cos_sin_cache(times, dtype, device)

        if kv is None:
            # fused case: apply to qkv
            if qkv.dim() == 5:  # (B, S, 3, H, D)
                q = qkv[:, :, 0]
                k = qkv[:, :, 1]
                v = qkv[:, :, 2]
                q = apply_rotary_emb_torch(q, self._cos_cached, self._sin_cached, self.interleaved)
                k = apply_rotary_emb_torch(k, self._cos_k_cached, self._sin_k_cached, self.interleaved)
                return torch.stack([q, k, v], dim=2)  # (B, S, 3, H, D)
            else:  # (B, S, H_total, D), e.g. GQA
                assert num_heads_q is not None
                n_heads_total = qkv.shape[2]
                n_heads_k = (n_heads_total - num_heads_q) // 2
                q = qkv[:, :, :num_heads_q]
                k = qkv[:, :, num_heads_q : num_heads_q + n_heads_k]
                q = apply_rotary_emb_torch(q, self._cos_cached, self._sin_cached, self.interleaved)
                k = apply_rotary_emb_torch(k, self._cos_k_cached, self._sin_k_cached, self.interleaved)
                qkv_rot = torch.cat([q, k, qkv[:, :, num_heads_q + n_heads_k :]], dim=2)
                return qkv_rot
        else:
            q = apply_rotary_emb_torch(qkv, self._cos_cached, self._sin_cached, self.interleaved)
            if kv.dim() == 5:
                kv_rot = kv.clone()
                kv_rot[:, :, 0] = apply_rotary_emb_torch(kv[:, :, 0], self._cos_k_cached, self._sin_k_cached, self.interleaved)
            if kv.dim() == 4:
                kv_rot = kv.clone()
                kv_rot = apply_rotary_emb_torch(kv, self._cos_k_cached, self._sin_k_cached, self.interleaved)
            return q, kv_rot
    