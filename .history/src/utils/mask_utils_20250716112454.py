import torch
import src.data.constants as Constants
from src.data.constants import PAD
from typing import Optional

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)   # [L_Q, L_K] ，use upper triangular matrix
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)            #  use index to select the top-k queries
        self._mask = indicator.view(scores.shape).to(device) #  [B, H, top_k, L_K]
    
    @property
    def mask(self):
        return self._mask
    

def get_non_pad_mask(seq,pad=PAD):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    non_pad_mask = seq.ne(pad).type(torch.float).unsqueeze(-1)
    non_pad_mask[:,0,:] = 1.0  # ensure the first dimension is 1.0
    return non_pad_mask


def get_attn_key_pad_mask(seq_k, seq_q,pad=PAD):            # 兼容qk不等的情况
    """ For masking out the padding part of key sequence. """
    assert seq_k.dim() == 2 and seq_q.dim() == 2
    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    non_pad_mask = seq_k.eq(pad)
    non_pad_mask[:,0] = 0 
    non_pad_mask = non_pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return non_pad_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """
    assert seq.dim() == 2
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask

def get_self_attn_mask(seq,pad=PAD):
    self_attn_mask_subseq = get_subsequent_mask(seq)
    self_attn_mask_keypad = get_attn_key_pad_mask(seq,seq,pad)
    self_attn_mask_keypad = self_attn_mask_keypad.type_as(self_attn_mask_subseq)
    self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)
    return self_attn_mask

def get_time_window_mask(timestamps: torch.Tensor, window_size: float) -> torch.Tensor:
    """
    Sliding-window causal mask  (True=被掩蔽, False=可见)
    """
    diff = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)   
    mask = (diff < 0) | (diff > window_size)            
    return mask.bool()

def remove_all_zero_rows(b_x, non_pad_mask, lengths):
    original_shape = b_x.shape
    rows_mask = torch.nonzero(lengths).squeeze(-1)
    if rows_mask.numel() == 0:
        raise ValueError("All rows are zero-length, nothing to process.")  # 检查是否有有效行
    valid_lengths = lengths[rows_mask]
    valid_rows = b_x[rows_mask]  # 仅保留有效的行
    valid_non_pad_mask = non_pad_mask[rows_mask]  # 仅保留有效行的 non_pad_mask
    
    return valid_rows, valid_non_pad_mask, valid_lengths, rows_mask, original_shape

def get_last_valid_step(enc_out, non_pad_mask):
    length = non_pad_mask.sum(dim=1)
    length = torch.clamp(length, min=1)
    last_step_index = (length.squeeze() - 1).long()
    batch_idx = torch.arange(non_pad_mask.size(0), device=non_pad_mask.device)
    enc_last = enc_out[batch_idx, last_step_index]
    last_mask = non_pad_mask[batch_idx, last_step_index]
    return enc_last, last_mask


def get_self_attn_mask_from_non_pad_mask(
    non_pad_mask: Optional[torch.Tensor],
    causal: bool = True,
    use_query_mask: bool = False,
) -> Optional[torch.Tensor]:
    """
    构造自注意力掩码（支持 padding 和 causal）。
    Args:
        non_pad_mask: (B, L), bool tensor，True 表示有效位置，False 为 PAD
        causal: 是否添加下三角 causal mask（防止信息泄露）
    Returns:
        attn_mask: (B, L, L), bool tensor，True 表示该位置被 mask
    """
    if non_pad_mask is None:
        if not causal:
            return None
        else:
            raise ValueError("non_pad_mask is required ...")

    B, L = non_pad_mask.shape
    device = non_pad_mask.device

    # key padding mask: mask key 的 PAD 位置
    key_pad_mask = ~non_pad_mask.bool().unsqueeze(1).expand(B, L, L)  # (B, L, L)
    query_pad_mask = ~non_pad_mask.bool().unsqueeze(2).expand(B, L, L) if use_query_mask else torch.zeros((B, L, L), dtype=torch.bool, device=device)
    pad_mask = key_pad_mask | query_pad_mask  
    if causal:
        # causal mask：屏蔽未来位置
        causal_mask = torch.triu(torch.ones((L, L), device=device), diagonal=1).bool()
        causal_mask = causal_mask.unsqueeze(0).expand(B, L, L)
        return pad_mask | causal_mask
    else:
        return pad_mask




def get_attn_mask_with_cache(
    self_attn_mask: Optional[torch.Tensor],
    L_k: int,
) -> torch.Tensor:
    """
    扩展self attention mask，兼容缓存KV时的L_k扩展。
    Args:
        self_attn_mask: (B, L_q, L_q) 或 None
        L_k: 当前总的key长度（包含缓存）
    Returns:
        attn_mask: (B, L_q, L_k), bool tensor
    """
    if self_attn_mask is None:
        return None

    B, L_q, _ = self_attn_mask.size()
    assert L_k >= L_q, "L_k must be greater than or equal to L_q"
    device = self_attn_mask.device
    dtype = torch.bool

    attn_mask = torch.zeros(B, L_q, L_k, dtype=dtype, device=device)

    # 将self_attn_mask填充到最后L_q部分（当前query对应的key）
    attn_mask[:, :, -L_q:] = self_attn_mask

    return attn_mask


    

