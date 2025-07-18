import torch.nn as nn
import torch
import math
from .subLayers import MultiHeadAttention, PositionwiseFeedForward
from src.utils.mask_utils import remove_all_zero_rows
from typing import Optional, Tuple, Dict

class EncoderLayer(nn.Module):
    """ Compose with two layers """
    # non_pad_mask 是一个掩码张量，用于确保填充位置不会影响计算
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True,attn_types=("full",  "flash")):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, attn_types=attn_types)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)
    
    def set_attn_type(self, attn_type: str):
        self.slf_attn.set_attn_type(attn_type)

    def set_attn_dropout(self, p: float):
        self.slf_attn.set_dropout(p)

    def forward(self, enc_input, non_pad_mask=None, attn_mask=None, cache: Optional[Dict[str, torch.Tensor]] = None):
        enc_output, enc_slf_attn = self.slf_attn(                         # 根据点积注意力，为0的填充部分计算后为0
            enc_input, enc_input, enc_input, non_pad_mask=non_pad_mask.squeeze(-1),attn_mask=attn_mask, cache=cache)
        # print(f"encoder_layer: enc_output: {enc_output[0,:,0]},")
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class TimePositionalEncoding(nn.Module):
    """Temporal encoding in THP, ICML 2020
    """

    def __init__(self, d_model, device='cpu'):
        super().__init__()
        i = torch.arange(0, d_model, 1, device=device)
        div_term = (2 * (i // 2).float() * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('div_term', div_term)

    def forward(self, x):
        """Compute time positional encoding defined in Equation (2) in THP model.

        Args:
            x (tensor): time_seqs, [batch_size, seq_len]

        Returns:
            temporal encoding vector, [batch_size, seq_len, model_dim]

        """
        result = x.unsqueeze(-1) * self.div_term
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result
    
class TimeShiftedPositionalEncoding(nn.Module):
    """Time shifted positional encoding in SAHP, ICML 2020
    """

    def __init__(self, d_model, max_len=5000, device='cpu'):
        super().__init__()
        # [max_len, 1]
        position = torch.arange(0, max_len, device=device).float().unsqueeze(1)
        # [model_dim //2 ]
        div_term = (torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model)).exp()

        self.layer_time_delta = nn.Linear(1, d_model // 2, bias=False)

        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

    def forward(self, x, interval):
        """

        Args:
            x: time_seq, [batch_size, seq_len]
            interval: time_delta_seq, [batch_size, seq_len]

        Returns:
            Time shifted positional encoding defined in Equation (8) in SAHP model

        """
        phi = self.layer_time_delta(interval.unsqueeze(-1))
        aa = len(x.size())
        if aa > 1:
            length = x.size(1)
        else:
            length = x.size(0)

        arc = (self.position[:length] * self.div_term).unsqueeze(0)

        pe_sin = torch.sin(arc + phi)
        pe_cos = torch.cos(arc + phi)
        pe = torch.cat([pe_sin, pe_cos], dim=-1)

        return pe


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. Supports both full-sequence training and incremental inference with cache.
    """

    def __init__(self, d_model, d_rnn, num_layers=1):
        super().__init__()
        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=num_layers, batch_first=True)
        # self.rnn = nn.GRU(d_model, d_rnn, num_layers=num_layers, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)
        self.num_layers = num_layers

    def _init_cache(self, cache, batch_size, device):
        """
        Initialize hidden and cell state. If cache is None or contains None, create new zero states.
        """
        if cache is not None and cache[0] is not None:
            return cache
        h0 = torch.zeros(self.num_layers, batch_size, self.rnn.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.rnn.hidden_size, device=device)
        return (h0, c0)
        # return h0

    def forward(
        self,
        data: torch.Tensor,                         
        non_pad_mask: torch.Tensor,                 
        pad_seq_len: bool = True,
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_pack_seq: bool = True 
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        device = data.device
        print("rnn_in",)
        B, L, _ = data.shape
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()  # (B,)

        assert (lengths > 0).all(), "All sequences must have at least one non-padding token."
        if L != 1 and cache is not None and cache[0] is not None:
            raise RuntimeError(
                f"Full-sequence mode (L={L}) does not support non-empty cache. "
                f"Please reset cache to None when feeding full sequences."
            )

        cache = self._init_cache(cache, B, device)

        if L == 1 or not use_pack_seq:
            out, cache = self.rnn(data, cache)
            # out = self.projection(out)
            out = out * non_pad_mask
            return out, cache

        # 打包序列
        packed = nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        packed_out, cache = self.rnn(packed, cache)

        # 解包序列
        if pad_seq_len:
            out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)[0]
        else:
            out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)[0]

        out = self.projection(out)
        out = out * non_pad_mask

        return out, cache
