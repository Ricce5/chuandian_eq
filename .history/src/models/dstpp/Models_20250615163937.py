import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import src.data.constants as Constants
from .Layers import EncoderLayer

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    non_pad_mask = seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
    non_pad_mask[:,:,0] = 1.0  # ensure the first dimension is 1.0
    return n


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq, dim=2):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()[:2]
    subsequent_mask = torch.triu(
        torch.ones((dim, len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1).permute(1,2,0)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1,-1)  # b x ls x ls
    return subsequent_mask



class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device,attn_type, loc_dim):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event loc embedding
        self.event_emb = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.ModuleList([   # list改为List
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_loc)
        
        slf_attn_mask = slf_attn_mask[:,:,:,0] # shape: (batch, seq_len, seq_len)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output

class Encoder_type(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device,attn_type,
            num_event_types_pad,pad_token_id):
        super().__init__()

        self.d_model = d_model
        self.num_event_types_pad = num_event_types_pad
       
        self.pad_token_id = pad_token_id
        print(self.num_event_types_pad,self.pad_token_id)
        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # event loc embedding
        self.event_emb = nn.Embedding(self.num_event_types_pad, 
                                           self.d_model,
                                           padding_idx=self.pad_token_id)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.ModuleList([   # list改为List
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask


    def forward(self, event_type, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_type, dim=1)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type.unsqueeze(-1), seq_q=event_type.unsqueeze(-1))
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        enc_output = self.event_emb(event_type)
        
        slf_attn_mask = slf_attn_mask[:,:,:,0] # shape: (batch, seq_len, seq_len)

        for enc_layer in self.layer_stack:
            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output

class Encoder_ST(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device, loc_dim,attn_type,CosSin = False):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        self.event_emb_temporal = nn.Sequential(
          nn.Linear(1, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.event_emb_loc = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type,normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])
        # self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)

        enc_output_loc = self.event_emb_loc(event_loc)

        enc_output = enc_output_temporal+enc_output_loc
        
        slf_attn_mask = slf_attn_mask[:,:,:,0]

        for index in range(len(self.layer_stack)):
            enc_output_loc, _ = self.layer_stack_loc[index](
                enc_output_loc,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            enc_output_temporal, _ = self.layer_stack_temporal[index](
                enc_output_temporal,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            enc_output, _ = self.layer_stack[index](
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        
        return enc_output, enc_output_temporal, enc_output_loc
    




class Encoder_STM(nn.Module):

    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, device, loc_dim, attn_type):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # 时间编码的位置向量
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device)

        # 空间嵌入
        self.event_emb_loc = nn.Sequential(
            nn.Linear(loc_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 震级嵌入
        self.event_emb_magnitude = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # 编码层
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])
        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type= attn_type, normalize_before=False)
            for _ in range(n_layers)])
        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type= attn_type, normalize_before=False)
            for _ in range(n_layers)])
        self.layer_stack_mag = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type= attn_type, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        self.position_vec = self.position_vec.to(time)
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def forward(self, event_loc, event_time, event_magnitude, non_pad_mask):
        # 注意力掩码
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        slf_attn_mask = slf_attn_mask[:, :, :, 0]

        # 特征嵌入
        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)
        enc_output_loc = self.event_emb_loc(event_loc)
        enc_output_mag = self.event_emb_magnitude(event_magnitude)

        # 融合后嵌入
        enc_output = enc_output_temporal + enc_output_loc + enc_output_mag

        # 编码器层处理
        for index in range(len(self.layer_stack)):
            enc_output_loc, _ = self.layer_stack_loc[index](
                enc_output_loc, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

            enc_output_temporal, _ = self.layer_stack_temporal[index](
                enc_output_temporal, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

            enc_output_mag, _ = self.layer_stack_mag[index](
                enc_output_mag, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

            enc_output, _ = self.layer_stack[index](
                enc_output, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)

        return enc_output, enc_output_temporal, enc_output_loc, enc_output_mag


class Encoder_SE(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout,device, loc_dim,attn_type,CosSin = False):
        super().__init__()

        self.d_model = d_model
        self.loc_dim = loc_dim

        # position vector, used for temporal encoding
        self.register_buffer(
                "position_vec",
                torch.tensor([math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)])
            )


        # event loc embedding
        self.event_emb_loc = nn.Sequential(
          nn.Linear(self.loc_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
        )

        self.event_emb_mag = nn.Sequential(
          nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
        )

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type,normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_loc = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])

        self.layer_stack_temporal = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask.float().expand_as(result)

    def forward(self, event_loc,event_mag, event_time, non_pad_mask):
        """ Encode event sequences via masked self-attention. """

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = get_subsequent_mask(event_loc, dim=self.loc_dim)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_loc, seq_q=event_loc)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)

        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        enc_output_temporal = self.temporal_enc(event_time, non_pad_mask)

        enc_output_loc = self.event_emb_loc(event_loc)
        enc_output_mag = self.event_emb_mag(event_mag)
        enc_output_mark = enc_output_mag+ enc_output_loc
        enc_output = enc_output_temporal+ enc_output_mark
        slf_attn_mask = slf_attn_mask[:,:,:,0]

        for index in range(len(self.layer_stack)):
            enc_output_mark, _ = self.layer_stack_loc[index](
                enc_output_mark,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            enc_output_temporal, _ = self.layer_stack_temporal[index](
                enc_output_temporal,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

            enc_output, _ = self.layer_stack[index](
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        
        return enc_output, enc_output_temporal, enc_output_mark



def remove_all_zero_rows(b_x, non_pad_mask, lengths):
    original_shape = b_x.shape
    rows_mask = torch.nonzero(lengths).squeeze(-1)
    if rows_mask.numel() == 0:
        raise ValueError("All rows are zero-length, nothing to process.")  # 检查是否有有效行
    valid_lengths = lengths[rows_mask]
    valid_rows = b_x[rows_mask]  # 仅保留有效的行
    valid_non_pad_mask = non_pad_mask[rows_mask]  # 仅保留有效行的 non_pad_mask
    
    return valid_rows, valid_non_pad_mask, valid_lengths, rows_mask, original_shape

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


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,device=None,loc_dim=2, attn_type= 'full'):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim,
            attn_type=attn_type
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        
        enc_output = self.encoder(event_loc, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)

        return enc_output, non_pad_mask
    


class Transformer_type(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,dropout_post_rnn=0.3,
            device=None, attn_type= 'full',
            num_event_types_pad=None, pad_token_id=-100):
        super().__init__()

        self.encoder = Encoder_type(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            attn_type=attn_type,
            num_event_types_pad=num_event_types_pad,
            pad_token_id=pad_token_id
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)
        self.dropout = nn.Dropout(dropout)

    def forward(self, event_type, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        
        enc_output = self.encoder(event_type, event_time, non_pad_mask)
        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output = self.dropout(enc_output)
        return enc_output, non_pad_mask

class Transformer_ST(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,dropout_post_rnn=0.3,
            device=None,loc_dim=2,CosSin=False,attn_type='full'):
        super().__init__()

        self.encoder = Encoder_ST(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim,
            CosSin = CosSin,
            attn_type=attn_type
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)
        self.rnn_temporal = RNN_layers(d_model, d_rnn)
        self.rnn_spatial = RNN_layers(d_model, d_rnn)
        self.dropout = nn.Dropout(dropout_post_rnn)

    def forward(self, event_loc, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        enc_output, enc_output_temporal, enc_output_loc = self.encoder(event_loc, event_time, non_pad_mask)
        
        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
        enc_output_loc = self.rnn_spatial(enc_output_loc, non_pad_mask)

        enc_output_all = torch.cat((enc_output_temporal, enc_output_loc, enc_output),dim=-1)
        enc_output_all = self.dropout(enc_output_all)  
        return enc_output_all, non_pad_mask


class Transformer_STM(nn.Module):
    """一个融合空间、时间和震级信息的序列到序列模型，带有注意力机制"""

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
            device=None, loc_dim=2,attn_type='full'):
        super().__init__()

        # 使用增强后的编码器
        self.encoder = Encoder_STM(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim=loc_dim,
            attn_type=attn_type
        )

        # 可学习参数
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # RNN层分别处理三种特征编码
        self.rnn = RNN_layers(d_model, d_rnn)
        self.rnn_temporal = RNN_layers(d_model, d_rnn)
        self.rnn_spatial = RNN_layers(d_model, d_rnn)
        self.rnn_magnitude = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc, event_time, event_magnitude):
        """
        输入:
            event_loc:       (batch, seq_len, loc_dim)
            event_time:      (batch, seq_len)
            event_magnitude: (batch, seq_len)
        输出:
            enc_output_all:  (batch, seq_len, d_rnn * 4)
            non_pad_mask:    (batch, seq_len, 1)
        """

        non_pad_mask = get_non_pad_mask(event_time)

        # 编码器返回四个向量：融合的编码 + 各自的子编码
        enc_output, enc_output_temporal, enc_output_loc, enc_output_mag = self.encoder(
            event_loc, event_time, event_magnitude, non_pad_mask
        )

        # 三类特征应有差异
        assert (enc_output != enc_output_temporal).any() & \
               (enc_output != enc_output_loc).any() & \
               (enc_output != enc_output_mag).any()

        # RNN 层处理
        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
        enc_output_loc = self.rnn_spatial(enc_output_loc, non_pad_mask)
        enc_output_mag = self.rnn_magnitude(enc_output_mag, non_pad_mask)

        # 拼接最终特征输出
        enc_output_all = torch.cat(
            (enc_output_temporal, enc_output_loc, enc_output_mag, enc_output),
            dim=-1
        )  # 输出维度: batch * seq_len * (d_rnn * 4)

        return enc_output_all, non_pad_mask

class Transformer_SE(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
            device=None,loc_dim=2,CosSin=False,attn_type='full'):
        super().__init__()

        self.encoder = Encoder_SE(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim = loc_dim,
            CosSin = CosSin,
            attn_type=attn_type
        )

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)
        self.rnn_temporal = RNN_layers(d_model, d_rnn)
        self.rnn_mark = RNN_layers(d_model, d_rnn)

    def forward(self, event_loc,event_mag, event_time):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_loc: batch*seq_len*2;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim
        """

        non_pad_mask = get_non_pad_mask(event_time)
        enc_output, enc_output_temporal, enc_output_mark = self.encoder(event_loc,event_mag, event_time, non_pad_mask)

        
        enc_output = self.rnn(enc_output, non_pad_mask)
        enc_output_temporal = self.rnn_temporal(enc_output_temporal, non_pad_mask)
        enc_output_mark = self.rnn_mark(enc_output_mark, non_pad_mask)

        enc_output_all = torch.cat((enc_output_temporal, enc_output_mark, enc_output),dim=-1)
        # 输出的特征维数为d_model*3
        return enc_output_all, non_pad_mask