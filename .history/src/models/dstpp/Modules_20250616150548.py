import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .masking import TriangularCausalMask, ProbMask
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from  math import sqrt
from abc import ABC, abstractmethod
from src.utils.registrable import Registrable


class BaseAttention(nn.Module, ABC,Registrable):
    """
    Abstract base class for attention mechanisms.

    Subclasses must implement the `forward` method.

    Expected input shapes:
        query, key, value: [B, L, H, D]
        attn_mask: Optional[Tensor] with shape broadcastable to [B, H, L, S]
        padding_mask: Optional[Tensor] with shape [B, L]

    Returns:
        output: [B, L, H, D]
        attention_weights (optional): [B, H, L, S] or None
    """

    def __init__(self,output_attention=False):
        super().__init__()
        self.output_attention = output_attention

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor = None,
        padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute attention.

        Args:
            query: [B, L, H, D]
            key: [B, L, H, D]
            value: [B, L, H, D]
            attn_mask: [B, H, L, S] or [B, L, S] or None
            padding_mask: [B, L] or None

        Returns:
            output: [B, L, H, D]
            attention_weights: [B, H, L, S] or None
        """
        pass


@BaseAttention.register(name="Standard")
class StandardAttention(BaseAttention):
    """
    Multi-Head Scaled Dot-Product Attention with mask support.

    Args:
        scale (float): Use 1 / sqrt(d_k) as scale.
        attn_dropout (float): Dropout rate after softmax.
        output_attention (bool): Whether to return attention weights.
    """

    def __init__(self, scale, attn_dropout=0.1, output_attention=True):
        super().__init__(output_attention=output_attention)
        self.scale = scale 
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, attn_mask=None, padding_mask=None):
        """
        Args:
            q, k, v: [B, L, H, D]
            attn_mask: [B, H, L, S] or [B, L, S] or None
            padding_mask: Optional, not used (reserved for flash-attn compatibility)

        Returns:
            output: [B, L, H, D]
            attn_weights (optional): [B, H, L, S]
        """
        # [B, L, H, D] → [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q * self.scale, k.transpose(2, 3))  # [B, H, L, S]

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L, S]
            scores = scores.masked_fill(attn_mask, -1e9)
        

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)  # [B, H, L, D]
        output = output.transpose(1, 2)  # → [B, L, H, D]

        if self.output_attention:
            return output, attn_weights
        else:
            return output, None

@BaseAttention.register(name="Full")
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attn_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__(output_attention=output_attention)
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self, q, k, v, attn_mask,padding_mask=None):
        B, L, H, E = q.shape
        _, S, _, D = v.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=q.device).mask

            scores.masked_fill_(attn_mask, -1e-9)

        attn_weights = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attn_weights, v)

        if self.output_attention:
            return (output.contiguous(), attn_weights)
        else:
            return (output.contiguous(), None)
        
@BaseAttention.register(name="Prob")
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attn_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__(output_attention=output_attention)
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attn_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1)) # factor*ln(L_q)*L_k

        return Q_K, M_top # M_top: [B, H, n_top] (n_top = c*ln(L_q))

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # use mask
            assert(L_Q == L_V) # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            # 构造 ProbAttention 的因果 mask
            prob_mask = ProbMask(B, H, L_Q, index, scores, device=V.device).mask  # [B, H, top_k, L]

            if attn_mask is not None:
                # attn_mask: [B, 1, L_Q, L_V]
                # 取出 top-k 行
                external_mask = attn_mask.expand(B, H, L_Q, L_V)[
                    torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :
                ]  # shape: [B, H, top_k, L_V]

                # 合并两种掩码：True 表示需要 mask
                attn_mask = prob_mask | external_mask
            else:
                attn_mask = prob_mask

            scores.masked_fill_(attn_mask, -1e-9)

        attn = torch.softmax(scores, dim=-1)

        context_in[
            torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :
        ] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[
                torch.arange(B)[:, None, None],
                torch.arange(H)[None, :, None],
                index, :
            ] = attn
            return context_in, attns
        else:
            return context_in, None


    def forward(self, q, k, v, attn_mask,padding_mask=None):
        B, L_Q, H, D = q.shape
        _, L_K, _, _ = k.shape

        q = q.transpose(2,1)
        k = k.transpose(2,1)
        v = v.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(q, k, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context 
        context = self._get_initial_context(v, L_Q)
        # update the context with selected top_k q
        context, attn = self._update_context(context, v, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn
    


class FlashAttentionWrapper(nn.Module):
    def __init__(self, attn_dropout=0.1, causal=True, output_attention=False):
        super().__init__()
        self.dropout = attn_dropout
        self.causal = causal
        self.output_attention = output_attention

    def forward(self, q, k, v, padding_mask=None, attn_mask=None):
        # 没有使用 attn_mask，因为 FlashAttention 不支持
        """
        Args:
            q, k, v: [B, L, H, D]
            padding_mask: [B, L] where 1=True means valid, 0=False means padding
        Returns:
            out: [B, L, H, D]
        """
        B, L, H, D = q.shape
        device = q.device

        # Ensure float16 for FlashAttention
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        # Use custom unpad_input: returns 5 values
        q_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(q, padding_mask)
        k_unpad, _, _, _, _ = unpad_input(k, padding_mask)
        v_unpad, _, _, _, _ = unpad_input(v, padding_mask)

        # Stack QKV into shape [total, 3, H, D]
        qkv = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)  # [total, 3, H, D]

        # Call varlen FlashAttention
        out_unpad = flash_attn_varlen_qkvpacked_func(
            qkv,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=self.dropout,
            softmax_scale=None,
            causal=self.causal,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=False
        )

        # Recover [B, L, H, D]
        out = pad_input(out_unpad, indices, B, L)
        out = out.to(torch.float32)

        return (out, None) 
