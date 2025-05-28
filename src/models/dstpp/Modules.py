import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .masking import TriangularCausalMask, ProbMask
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
from xformers.components.attention import build_attention
from xformers.components.attention.utils import maybe_merge_masks
from  math import sqrt


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn



class ScaledDotProductMultiHeadAttention(nn.Module):
    def __init__(self, scale, attn_dropout=0.1, output_attention=True):
        super().__init__()
        self.attention = ScaledDotProductAttention(temperature=scale, attn_dropout=attn_dropout)
        self.output_attention = output_attention

    def forward(self, q, k, v, attn_mask=None):
        """
        q, k, v: [B, L, H, D]  → internally transposed to [B, H, L, D]
        attn_mask: [B, H, L, S] or broadcastable
        """
        # Transpose to [B, H, L, D]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Ensure mask shape is broadcastable to [B, H, L_q, L_k]
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)

        output, attn = self.attention(q, k, v, mask=attn_mask)  # [B, H, L, D], [B, H, L, L]

        # Transpose back to [B, L, H, D]
        output = output.transpose(1, 2)

        if self.output_attention:
            return output, attn
        else:
            return output, None

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attn_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attn_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device).mask

            scores.masked_fill_(attn_mask, -1e-9)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        

        if torch.isnan(scores).any():
            print("NaN in scores!")

        if torch.isnan(A).any():
            print("NaN in attention weights!")

        if torch.isnan(V).any():
            print("NaN in output!")

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attn_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
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


    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item() # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item() # c*ln(L_q) 

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        # add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context 
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
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





def recover_padding_mask(attn_mask: torch.Tensor) -> torch.Tensor:
    """
    从 [B, 1, L, S] 的 combined attention mask 中恢复原始 padding mask（[B, S]）。

    参数:
        attn_mask (torch.Tensor): 结合了 padding 和 causal 的 mask，形状为 [B, 1, L, S]。
                                值为 True 表示该位置被屏蔽。

    返回:
        padding_mask (torch.Tensor): [B, S]，bool 类型，True 表示该 key 是 padding。
    """
    B, _, L, S = attn_mask.shape

    # 构造 causal mask: [1, 1, L, S]
    causal_mask = torch.triu(
        torch.ones((L, S), dtype=torch.bool, device=attn_mask.device),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)  # → [1, 1, L, S]

    # padding mask = (combined mask) & (非 causal 部分)
    padding_only = attn_mask & ~causal_mask  # → [B, 1, L, S]

    # 对 query 维度取最大，判断哪些 key 总是被屏蔽
    padding_mask = padding_only.any(dim=2).squeeze(1)  # → [B, S]

    return padding_mask
