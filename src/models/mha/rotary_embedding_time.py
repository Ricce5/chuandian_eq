import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
from einops import rearrange
from flash_attn.layers.rotary import  apply_rotary_emb_qkv_, apply_rotary_emb_kv_, apply_rotary_emb_func ,apply_rotary_emb_torch


class RotaryEmbeddingTime(torch.nn.Module):
    """
    Rotary Position Embedding / XPos with optional real-time support
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        pos_idx_in_fp32: bool = True,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32

        # Generate and save the inverse frequency buffer
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base

        # XPos scale
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim) / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        # Cache for cos/sin
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    # ----------------- 新增方法: 支持真实时间 -----------------
    def _update_cos_sin_cache_with_times(self, times, device=None, dtype=None):
        """
        times: shape [seqlen] or [batch, seqlen], 单位自定义（秒/毫秒等）
        """
        if times.dim() == 2:
            assert times.shape[0] == 1, "Batch dimension should be 1 for time input"
            times = times[0]

        times = times.to(torch.float32)  
        if self.inv_freq.dtype != torch.float32:
            inv_freq = self._compute_inv_freq(device=device)
        else:
            inv_freq = self.inv_freq

        # [seqlen, dim/2]
        freqs = torch.outer(times, inv_freq)

        if self.scale is None:
            self._cos_cached = torch.cos(freqs).to(dtype)
            self._sin_cached = torch.sin(freqs).to(dtype)
        else:
            seqlen = times.shape[0]
            power = (
                torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                - seqlen // 2
            ) / self.scale_base
            scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")

            self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
            self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
            self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
            self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    # ----------------- 原始离散位置缓存逻辑 -----------------
    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen

            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                inv_freq = self.inv_freq.to(torch.float32)
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(seqlen, dtype=self.scale.dtype, device=self.scale.device)
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(power, "s -> s 1")
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    # ----------------- forward -----------------
    def forward(
        self,
        qkv: torch.Tensor,
        kv: Optional[torch.Tensor] = None,
        seqlen_offset: Union[int, torch.Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
        times: Optional[torch.Tensor] = None,  # <--- 新增
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        如果传入 times，将用真实时间计算 RoPE；否则使用离散位置索引。
        """
        seqlen = qkv.shape[1]

        if times is not None:
            self._update_cos_sin_cache_with_times(times, device=qkv.device, dtype=qkv.dtype)
        elif max_seqlen is not None:
                self._update_cos_sin_cache(max_seqlen, device=qkv.device, dtype=qkv.dtype)
        elif isinstance(seqlen_offset, int):
                self._update_cos_sin_cache(seqlen + seqlen_offset, device=qkv.device, dtype=qkv.dtype)

        # 后续与原实现一致
        if kv is None:
            if self.scale is None:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
            else:
                return apply_rotary_emb_qkv_(
                    qkv,
                    self._cos_cached,
                    self._sin_cached,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                    num_heads_q=num_heads_q,
                )
        else:
            q = qkv
            q = apply_rotary_emb_func(
                q,
                self._cos_cached,
                self._sin_cached,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            if self.scale is None:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_cached,
                    self._sin_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            else:
                kv = apply_rotary_emb_kv_(
                    kv,
                    self._cos_k_cached,
                    self._sin_k_cached,
                    interleaved=self.interleaved,
                    seqlen_offsets=seqlen_offset,
                )
            return q, kv