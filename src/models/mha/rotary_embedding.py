import torch
import torch.nn as nn
import math


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, interleaved=False, device=None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.interleaved = interleaved

        # 用于频率计算的倒数频率向量
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)  # shape: (dim // 2,)

    def get_cos_sin(self, positions):
        """
        positions: shape (batch, seqlen) or (seqlen,)
        returns: cos, sin of shape (batch, seqlen, dim // 2) or (seqlen, dim // 2)
        """
        freqs = torch.einsum("...i,d->...id", positions, self.inv_freq)  # [batch, seqlen, dim//2] or [seqlen, dim//2]
        return torch.cos(freqs), torch.sin(freqs)

    def apply_rotary_emb(self, x, cos, sin):
        """
        x: (batch, seqlen, nheads, dim)
        cos/sin: (batch or 1, seqlen, dim//2) or (seqlen, dim//2)
        """
        x1, x2 = x[..., 0::2], x[..., 1::2]  # 拆成偶数和奇数位置
        # 广播 cos/sin 到正确形状
        cos = cos.unsqueeze(-2)  # [..., 1, dim//2]
        sin = sin.unsqueeze(-2)
        x_rot = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # [..., dim]
        return x_rot

    def forward(self, q, kv, positions=None):
        """
        q: (batch, seqlen, nheads, dim)
        kv: (batch, seqlen, 2, nheads_kv, dim)
        positions: (batch, seqlen) or (seqlen,) or None
        """
        if positions is None:
            positions = torch.arange(q.shape[1], device=q.device, dtype=torch.float)

        if positions.dim() == 1:
            # (seqlen,) → (1, seqlen)
            positions = positions.unsqueeze(0)

        cos, sin = self.get_cos_sin(positions)  # shape: (batch, seqlen, dim//2)

        # 扩展 cos/sin 为 broadcastable 到 q.shape
        cos = cos.to(dtype=q.dtype, device=q.device)
        sin = sin.to(dtype=q.dtype, device=q.device)

        # 只对前 rotary_dim 部分做旋转
        q_rot = self.apply_rotary_emb(q[..., :self.dim], cos, sin)
        k_rot = self.apply_rotary_emb(kv[..., 0, :, :self.dim], cos, sin)

        # 拼回原向量
        q = torch.cat([q_rot, q[..., self.dim:]], dim=-1)
        kv[..., 0, :, :self.dim] = k_rot  # 注意是对 kv 中的 key 做 RoPE
        return q, kv


