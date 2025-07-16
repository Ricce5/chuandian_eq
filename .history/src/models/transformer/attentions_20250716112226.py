import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.mask_utils import TriangularCausalMask, ProbMask, get_self_attn_mask_from_non_pad_mask, get_attn_mask_with_cache
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func, flash_attn_kvpacked_func,flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input
from  math import sqrt
from abc import ABC, abstractmethod
from src.utils.registrable import Registrable
from typing import Optional


class BaseAttention(nn.Module, ABC,Registrable):
    """
    Abstract base class for attention mechanisms.

    Subclasses must implement the `forward` method.

    Expected input shapes:
        query, key, value: [B, L, H, D]
        attn_mask: Optional[Tensor] with shape broadcastable to [B, H, L, S]
        non_pad_mask: Optional[Tensor] with shape [B, L]

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
        attn_mask: Optional[torch.Tensor] = None,
        non_pad_mask: torch.Tensor = None,
        causal: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute attention.

        Args:
            query: [B, L, H, D]
            key: [B, L, H, D]
            value: [B, L, H, D]
            attn_mask: [B, H, L, S] or [B, L, S] or None
            non_pad_mask: [B, L] or None

        Returns:
            output: [B, L, H, D]
            attention_weights: [B, H, L, S] or None
        """
        pass

    @staticmethod
    def build_attn_mask(
        non_pad_mask: torch.Tensor,
        L_k: int,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Generate an attention mask compatible with cached key/value tensors,
        combining both key padding mask and optional causal (future-masking).

        Args:
            non_pad_mask (Tensor): Boolean tensor of shape [B, L_q],
                where True indicates a valid (non-padding) token.
            L_k (int): Total key sequence length, including any cached keys.
            causal (bool): If True, applies a causal (upper triangular) mask to prevent
                attending to future positions.

        Returns:
            Tensor: Boolean attention mask of shape [B, 1, L_q, L_k],
                where True indicates masked positions.
        """
        assert L_k is not None, "L_k must be provided"
        non_pad_mask = non_pad_mask.to(dtype=torch.bool) if non_pad_mask is not None else None
        self_attn_mask = get_self_attn_mask_from_non_pad_mask(non_pad_mask, causal=causal)
        attn_mask = get_attn_mask_with_cache(self_attn_mask, L_k)
        attn_mask = attn_mask.unsqueeze(1) if attn_mask is not None else None
        return attn_mask
  
    @staticmethod
    def mask_out(out, non_pad_mask):
        """
        Apply non-padding mask to the output.
        Args:
            out: [B, L, H, D]
            non_pad_mask: [B, L] or None
        Returns:
            out: [B, L, H, D] with padding positions set to 0
        """
        if non_pad_mask is not None:
            mask = non_pad_mask.bool().unsqueeze(2).unsqueeze(3)  
            out = torch.where(mask, out, torch.zeros_like(out))  
        return out
    
    def set_dropout(self, p: float):
        """
        Optional: Override in subclasses if they use dropout.
        """
        pass



@BaseAttention.register(name="standard")
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

    def forward(self, q, k, v, non_pad_mask=None, attn_mask=None, causal=True):
        """
        Args:
            q, k, v: [B, L, H, D]
            attn_mask: [B, H, L, S] or [B, L, S] or None
            non_pad_mask: Optional, not used (reserved for flash-attn compatibility)

        Returns:
            output: [B, L, H, D]
            attn_weights (optional): [B, H, L, S]
        """
        # [B, L, H, D] → [B, H, L, D]
        L_k = k.shape[1] 
        if attn_mask is None:
            attn_mask = self.build_attn_mask(non_pad_mask, L_k, causal=causal)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q * self.scale, k.transpose(2, 3))  # [B, H, L, S]

        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L, S]
            scores = scores.masked_fill(attn_mask, -np.inf)
        

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)  # [B, H, L, D]``
        output = output.transpose(1, 2)  # → [B, L, H, D]
        output = self.mask_out(output, non_pad_mask)
        if self.output_attention:
            return output, attn_weights
        else:
            return output, None
    def set_dropout(self, p: float):
        self.dropout = nn.Dropout(p)

   

@BaseAttention.register(name="full")
class FullAttention(BaseAttention):
    def __init__(self, scale=None, attn_dropout=0.1, output_attention=False):
        super().__init__(output_attention=output_attention)
        self.scale = scale
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, non_pad_mask=None, attn_mask=None, causal=True):
        B, L_q, H, E = q.shape
        _, L_k, _, D = v.shape
        scale = self.scale or 1. / sqrt(E)
        if attn_mask is None:
            attn_mask = self.build_attn_mask(non_pad_mask, L_k, causal=causal)

        scores = torch.einsum("blhe,bshe->bhls", q, k)
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, L, S]
            scores = scores.masked_fill(attn_mask, -np.inf)
        attn_weights = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attn_weights, v)
        output = self.mask_out(output, non_pad_mask)
        if self.output_attention:
            return (output.contiguous(), attn_weights)
        else:
            return (output.contiguous(), None)
    def set_dropout(self, p: float):
        self.dropout = nn.Dropout(p)


@BaseAttention.register(name="prob")
class ProbAttention(BaseAttention):
    def __init__(self, mask_flag=True, factor=10, scale=None, attn_dropout=0.1, output_attention=False):
        super().__init__(output_attention=output_attention)
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

            scores.masked_fill_(attn_mask, -np.inf)

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


    def forward(self, q, k, v, non_pad_mask=None, attn_mask=None, causal=True):
        B, L_Q, H, D = q.shape
        _, L_K, _, _ = k.shape
        if attn_mask is None:
            attn_mask = self.build_attn_mask(non_pad_mask, L_K, causal=causal)

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
        out = context.transpose(2,1).contiguous()  # [B, L_Q, H, D]
        out = self.mask_out(out, non_pad_mask)
        return out, attn
    
    def set_dropout(self, p: float):
        device = next(self.parameters()).device  # 获取当前模块的 device
        self.dropout = nn.Dropout(p).to(device)


    

    
@BaseAttention.register(name="flash")
class FlashAttentionWrapper(BaseAttention):
    def __init__(self, attn_dropout=0.1, output_attention=False, scale=None, precision="fp16"):
        super().__init__(output_attention=output_attention)
        self.dropout = attn_dropout
        self.scale = scale
        self.precision = precision.lower()

        if self.precision not in {"fp16", "bf16"}:
            raise ValueError(f"Unsupported precision '{self.precision}'. Must be 'fp16' or 'bf16'.")

    def forward(self, q, k, v, non_pad_mask=None, attn_mask=None, causal=True):
        torch.autograd.set_detect_anomaly(True)
        if torch.isnan(q).any():
            print("警告：输入张量 'q' 包含 NaN 值。")
            # 可以选择在这里抛出错误，或根据需求记录日志/处理
            # raise ValueError("Input tensor 'q' contains NaN values.")
        if torch.isnan(k).any():
            print("警告：输入张量 'k' 包含 NaN 值。")
        if torch.isnan(v).any():
            print("警告：输入张量 'v' 包含 NaN 值。")
        B, L_q, H, D = q.shape
        L_kv = k.shape[1]

        assert q.device.type == "cuda", "FlashAttention requires CUDA device"

        # Select dtype based on precision
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map[self.precision]

        # Convert q/k/v to target dtype
        q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)

        # Handle padding mask
        print(non_pad_mask.shape)
        if not non_pad_mask.all():
            print("attn non_pad_mask 中存在 False 值。")
        else:
            print("attn non_pad_mask 中所有值都是 True。")
        if non_pad_mask is None:
            non_pad_mask_q = torch.ones(B, L_q, dtype=torch.bool, device=q.device)
        else:
            non_pad_mask_q = non_pad_mask.bool()
        
        

        # Handle prefix cache: if kv is longer than q, pad kv mask
        if L_q != L_kv:
            assert L_q < L_kv, "L_q must be <= L_kv for prefix masking"
            non_pad_mask_kv = self.build_kv_non_pad_mask_with_cache(non_pad_mask_q, total_kv_len=L_kv)
        else:
            non_pad_mask_kv = non_pad_mask_q

        # Unpad q/k/v for FlashAttention
        q_unpad, q_indices, cu_q, q_max_len, _ = unpad_input(q, non_pad_mask_q)
        k_unpad, _, cu_k, k_max_len, _ = unpad_input(k, non_pad_mask_kv)
        v_unpad, _, _, _, _ = unpad_input(v, non_pad_mask_kv)

        # Determine scaling factor
        softmax_scale = self.scale if self.scale is not None else 1.0 / sqrt(D)
        out_unpad = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=q_max_len,
            max_seqlen_k=k_max_len,
            dropout_p=self.dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=(-1, -1),
            return_attn_probs=False,
        )

        # Pad back to [B, L_q, H, D]
        out = pad_input(out_unpad, q_indices, B, L_q)
        if torch.isnan(out).any():
            print("警告：输出张量 'out' 包含 NaN 值。")
            # raise ValueError("Output tensor 'out' contains NaN values.")
        return out.to(torch.float32), None

    @staticmethod
    def build_kv_non_pad_mask_with_cache(
        non_pad_mask_q: torch.Tensor,  # [B, L_q]
        total_kv_len: int,         # L_kv
    ) -> torch.Tensor:
        B, L_q = non_pad_mask_q.shape
        L_kv = total_kv_len
        L_cache = L_kv - L_q

        if L_cache == 0:
            return non_pad_mask_q

        prefix_valid = torch.ones((B, L_cache), dtype=torch.bool, device=non_pad_mask_q.device)
        return torch.cat([prefix_valid, non_pad_mask_q], dim=1)  # [B, L_kv]
    
    def set_dropout(self, p: float):
        self.dropout = p




# @BaseAttention.register(name="flash")
# class FlashAttentionWrapper(BaseAttention):
#     def __init__(self, attn_dropout=0.1, causal=True, output_attention=False,precision="fp16",scale=None):
#         super().__init__( output_attention= output_attention)
#         self.dropout = attn_dropout
    

#     def forward(self, q, k, v, non_pad_mask=None, attn_mask=None, causal=True):
#         # 没有使用 attn_mask，因为 FlashAttention 不支持
#         """
#         Args:
#             q, k, v: [B, L, H, D]
#             non_pad_mask: [B, L] where 1=True means valid, 0=False means padding
#         Returns:
#             out: [B, L, H, D]
#         """
#         B, L, H, D = q.shape
#         device = q.device

#         # Ensure float16 for FlashAttention
#         q = q.to(torch.float16)
#         k = k.to(torch.float16)
#         v = v.to(torch.float16)

#         # Use custom unpad_input: returns 5 values
#         q_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(q, non_pad_mask)
#         k_unpad, _, _, _, _ = unpad_input(k, non_pad_mask)
#         v_unpad, _, _, _, _ = unpad_input(v, non_pad_mask)

#         # Stack QKV into shape [total, 3, H, D]
#         qkv = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)  # [total, 3, H, D]

#         # Call varlen FlashAttention？
#         out_unpad = flash_attn_varlen_qkvpacked_func(
#             qkv,
#             cu_seqlens=cu_seqlens,
#             max_seqlen=max_seqlen,
#             dropout_p=self.dropout,
#             softmax_scale=None,
#             causal=causal,
#             window_size=(-1, -1),
#             softcap=0.0,
#             alibi_slopes=None,
#             deterministic=False,
#             return_attn_probs=False
#         )

#         # Recover [B, L, H, D]
#         out = pad_input(out_unpad, indices, B, L)
#         out = out.to(torch.float32)

#         return (out, None) 

# @BaseAttention.register(name="flash_kv")
# class FlashKVAttentionWrapper(BaseAttention):
#     def __init__(self, attn_dropout=0, output_attention=False):
#         super().__init__(output_attention=output_attention)
#         self.dropout = attn_dropout
#     def forward(
#         self,
#         q: torch.Tensor,      # [B, L_q, H, D]
#         k: torch.Tensor,      # [B, L_k, H, D]
#         v: torch.Tensor,      # [B, L_k, H, D]
#         non_pad_mask: torch.Tensor = None,
#         attn_mask: torch.Tensor = None,
#         causal: bool = True,
#     ):
#         B, L, H, D = q.shape


#         # Ensure float16 for FlashAttention
#         q = q.to(torch.float16)
#         k = k.to(torch.float16)
#         v = v.to(torch.float16)

#         kv = torch.stack([k, v], dim=2)  # [B, L_k, 2, H, D]

#         out = flash_attn_kvpacked_func(
#             dropout_p=self.dropout,
#             q=q,
#             kv=kv,
#             causal=causal,
#             softmax_scale=None,
#             return_attn_probs=False
#         )  # [B, L_q, H, D]

#         out = out.to(torch.float32)
#         return out, None