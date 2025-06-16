import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Modules import StandardAttention, FlashAttentionWrapper, ProbAttention,FullAttention,BaseAttention
import src.models.dstpp.Constants as Constants
from math import sqrt

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module supporting scaled_dot, full, and prob attention """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True, attn_type='full'):
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

        if attn_type == 'full':
            # from .Modules import FullAttention  
            Attention = BaseAttention.by_name('Full')
            self.attention = Attention(
                scale=1.0 / sqrt(d_k),
                attn_dropout=dropout,
                output_attention=True
            )
        elif attn_type == 'scaled_dot':
            # from .Modules import StandardAttention 
            Attention = BaseAttention.by_name('Standard')
            self.attention = Attention(
                scale=1.0 / sqrt(d_k),
                attn_dropout=dropout,
                output_attention=True
            )
        elif attn_type == 'prob':
            # from .Modules import ProbAttention
            ProbAttention = BaseAttention.by_name('Prob')
            self.attention = ProbAttention(
                scale=1.0 / sqrt(d_k),
                attn_dropout=dropout,
                output_attention=True
            )
        elif attn_type == 'flash':
            # from .Modules import FlashAttentionWrapper
            FlashAttentionWrapper = BaseAttention.by_name('Flash')
            self.attention = FlashAttentionWrapper(
                attn_dropout=dropout,
                output_attention=True
            )

        else:
            raise ValueError(f"Unsupported attn_type: {attn_type}")

    def forward(self, q, k, v, mask=None):
        B, L_q, _ = q.size()
        L_k, L_v = k.size(1), v.size(1)
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Linear projection + reshape: [B, L, n_head, D]
        q = self.w_qs(q).view(B, L_q, n_head, d_k)
        k = self.w_ks(k).view(B, L_k, n_head, d_k)
        v = self.w_vs(v).view(B, L_v, n_head, d_v)
        padding_mask = (residual.abs().sum(dim=-1) != 0)  # [B, L]
        
        # Prepare attn_mask: [B, H, L, S]
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B, 1, L, L]
      
        output, attn = self.attention(q, k, v, attn_mask=mask, padding_mask=padding_mask)

        # Combine heads: [B, L, n_head * D]
        output = output.contiguous().view(B, L_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)

        return output, attn


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


