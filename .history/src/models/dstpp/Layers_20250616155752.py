import torch.nn as nn

from .SubLayers import MultiHeadAttention, PositionwiseFeedForward


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

    def __init__(self, d_model, max_len=5000, device='cpu'):
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