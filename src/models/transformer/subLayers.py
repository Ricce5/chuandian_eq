# References:
# 1. Spatio-temporal Diffusion Point Processes: 
#    https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes
# 2. EasyTemporalPointProcess: 
#    https://github.com/ant-research/EasyTemporalPointProcess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attentions import StandardAttention, FlashAttentionWrapper, ProbAttention,FullAttention,BaseAttention
from math import sqrt
from typing import Optional, Dict
from src.utils.mask_utils import TriangularCausalMask, get_self_attn_mask_from_non_pad_mask,get_attn_mask_with_cache


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module supporting scaled_dot, full, prob, and flash attention with KV cache support """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True, attn_types=("full", "flash")):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.attn_modules = nn.ModuleDict()
        for name in attn_types:
            AttnCls = BaseAttention.by_name(name)
            kwargs = {
                "attn_dropout": dropout,
                "output_attention": True,
                "scale": 1.0 / sqrt(d_k) 
            }
            if name == "flash":
                kwargs["precision"] = "fp16"  
            self.attn_modules[name] = AttnCls(**kwargs)

        self.active_attn_type = None 
        
    def set_attn_type(self, attn_type: str):
        if attn_type not in self.attn_modules:
            raise ValueError(f"Unsupported attention type: {attn_type}")
        self.active_attn_type = attn_type
    
    def set_dropout(self, p: float):
        for attn_type, module in self.attn_modules.items():
            if hasattr(module, "set_dropout"):
                module.set_dropout(p)
            else:
                print(f"[Warning] Attention type '{attn_type}' does not support dropout setting.")


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        non_pad_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        causal: bool = True,
    ):
        B, L_q, _ = q.shape
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v
        residual = q

        if self.normalize_before:
            q = self.layer_norm(q)


        # === Project ===
        q_proj = self.w_qs(q).view(B, L_q, n_head, d_k)
        k_proj = self.w_ks(k).view(B, -1, n_head, d_k)
        v_proj = self.w_vs(v).view(B, -1, n_head, d_v)

        
        # === Extend cache AFTER attention ===
        if cache is not None and cache.get("k", None) is not None:
            k_proj = torch.cat([cache["k"], k_proj], dim=1)
            v_proj = torch.cat([cache["v"], v_proj], dim=1)

        L_k = k_proj.size(1) 

        # print(f"q_proj:{torch.sum(q_proj)}, k_proj:{torch.sum(k_proj)}, v_proj:{torch.sum(v_proj)}")
        # print(f"q_proj.shape:{q_proj.shape}, k_proj.shape:{k_proj.shape}, v_proj.shape:{v_proj.shape}")
        # print(f"q_proj[0, :,0,0]:{q_proj[0, :,0,0]}")
        # === Attention ===
        attn_module = self.attn_modules[self.active_attn_type]
        out, attn = attn_module(
            q_proj, k_proj, v_proj,
            non_pad_mask=non_pad_mask,
            attn_mask=attn_mask,
            causal=causal,
        )

        # print(f"out:{out[0, :,0,0]}")

        # === Post attention projection ===
        out = out.contiguous().view(B, L_q, -1)
        out = self.dropout(self.fc(out))
        out = out + residual
        if not self.normalize_before:
            out = self.layer_norm(out)

        # === Update cache: only append current step ===
        if cache is not None:
            self.append_valid_kv(cache, k_proj, v_proj, non_pad_mask)
        return out, attn

    @staticmethod
    def append_valid_kv(
        cache: Dict[str, torch.Tensor],
        k: torch.Tensor,
        v: torch.Tensor,
        valid_kv_mask: torch.Tensor
    ):
        """
        Append only valid (non-padding) positions in k/v to the cache.

        Assumes all samples in batch have the same valid_kv_mask.

        Args:
            cache: Dict with 'k' and 'v' tensors of shape (B, S_prev, H, D)
            k, v: Newly projected keys/values of shape (B, L_new, H, D)
            valid_kv_mask: (B, L_new) boolean mask where True means valid position
        """
        B, L, H, D = k.shape
        cache_len = cache["k"].size(1) if "k" in cache and cache["k"] is not None else 0

        if not torch.all(valid_kv_mask == valid_kv_mask[0]):
            raise ValueError("All samples in batch must have same valid_kv_mask for caching.")


        extended_mask = torch.cat([
            torch.zeros(cache_len, dtype=torch.bool, device=valid_kv_mask.device),
            valid_kv_mask[0].bool()
        ], dim=0)  # shape: (L_total,)

        k_valid = k[:, extended_mask]  # shape: (B, L_valid, H, D)
        v_valid = v[:, extended_mask]

        if "k" in cache and cache["k"] is not None:
            cache["k"] = torch.cat([cache["k"], k_valid], dim=1)
            cache["v"] = torch.cat([cache["v"], v_valid], dim=1)
        else:
            cache["k"], cache["v"] = k_valid, v_valid


        



class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


