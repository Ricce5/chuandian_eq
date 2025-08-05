import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional, Tuple, Union


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        return rearrange(torch.stack([-x2, x1], dim=-1), "... d two -> ... (d two)", two=2)


def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
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


class RotaryEmbeddingTime(nn.Module):
    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        device=None,
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


    def _update_cos_sin_cache(self, times: torch.Tensor, dtype: torch.dtype, device: torch.device):
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
            center = times.float().mean(dim=-1, keepdim=True)
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
        num_heads_q: Optional[int] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        qkv: (batch, seqlen, 3, nheads, headdim) or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
             or just (batch, seqlen, nheads, headdim) if kv is provided (i.e., this is Q)
        kv: optional, (batch, seqlen, 2, nheads, headdim)
        """
        device = qkv.device
        dtype = qkv.dtype

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
            kv[:, :, 0] = apply_rotary_emb_torch(kv[:, :, 0], self._cos_k_cached, self._sin_k_cached, self.interleaved)
            return q, kv
