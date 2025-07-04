import torch.nn as nn
import torch
import math
from .subLayers import MultiHeadAttention, PositionwiseFeedForward
from src.utils.mask_utils import remove_all_zero_rows

class EncoderLayer(nn.Module):
    """ Compose with two layers """
    # non_pad_mask 是一个掩码张量，用于确保填充位置不会影响计算
    def __init__(self, d_model, d_inner, n_head, d_k, d_v,attn_type, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before, attn_type=attn_type)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(                         # 根据点积注意力，为0的填充部分计算后为0
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask # 此处*是相同位置数值相乘

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
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()
        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask, pad_seq_len=True):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        # 移除全0行
        valid_data, valid_non_pad_mask, valid_lengths, rows_mask, original_shape = remove_all_zero_rows(
            data, non_pad_mask, lengths
        )

        # 经过 RNN
        out = self.rnn_layer_forward(valid_data, valid_non_pad_mask, valid_lengths, original_shape, pad_seq_len)

        # 恢复到原始 batch 的形状
        restored_out = torch.zeros(
            (original_shape[0], out.shape[1], out.shape[2]), dtype=out.dtype
        ).to(data.device)
        restored_out[rows_mask] = out

        return restored_out

    def rnn_layer_forward(self, data, non_pad_mask, lengths, original_shape, pad_seq_len=True):
        pad_seq_len = bool(pad_seq_len)

        # 打包有效序列
        packed = nn.utils.rnn.pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        ##
        batch_size = data.size(0)
        device = data.device
        h0 = torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)
        c0 = torch.zeros(1, batch_size, self.rnn.hidden_size, device=device)
        #
        packed_out, _ = self.rnn(packed, (h0, c0))
 

        # 解包序列，是否 pad 到原始长度
        if pad_seq_len:
            out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=original_shape[1])[0]
        else:
            out = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)[0]
        # 映射回 d_model
        out = self.projection(out)
        out = out*non_pad_mask
        return out

