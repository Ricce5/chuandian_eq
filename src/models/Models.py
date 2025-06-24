import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP, AttentionPooling
from .transformer import Transformer, Transformer_ST, Transformer_STM, Transformer_SE
# from .TppModels import THP
from src.utils.registrable import Registrable
from src.utils.mask_utils import get_last_valid_step


def default_batch_to_input(bx):
    t_n_seq = bx[:, :, 1]
    mag_seq = bx[:, :, 2:3]
    loc_seq = bx[:, :, 3:5]
    features = torch.concat((loc_seq, mag_seq), dim=-1)
    return features, t_n_seq


class BaseTransformerModel(nn.Module,Registrable):
    def __init__(self, transformer, mlp, device):
        super().__init__()
        self.transformer = transformer.to(device)
        self.mlp = mlp.to(device)
        self.device = device

    def forward(self, x):
        x = x.float()
        features, t_n_seq = self._batch_to_input(x)
        features_dict = {
            "event_time": t_n_seq,  # 事件时间序列
            "event_mark": features,    # 其他特征（如地理位置和幅度）
        }

        enc_out, non_pad_mask = self.transformer( features_dict)
        enc_last, _ = get_last_valid_step(enc_out, non_pad_mask)
        out = self.mlp(enc_last)
        return out.squeeze(1)

    def _batch_to_input(self, x):
        raise NotImplementedError("Child class must implement this method.")



class classifier(BaseTransformerModel):
    def __init__(self, args, device):
        transformer = Transformer_ST(
            d_model=args.d_model, d_rnn=args.d_rnn, d_inner=args.d_inner,
            n_layers=args.n_layers, n_head=args.n_head, d_k=args.d_k, d_v=args.d_v,
            dropout=args.t_dropout, dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device, dim=args.dim, attn_type=args.attn_type
        )
        mlp = MLP(args.mlp_hdw, input_size=3 * args.d_model, output_size=args.mlp_out, dropout_rate=args.mlp_dropout)
        super().__init__(transformer, mlp, device)

    def _batch_to_input(self, x):
        return default_batch_to_input(x)



class Classifier_SE(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        # Transformer 初始化
        self.transformer = Transformer_SE(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            loc_dim=args.loc_dim,
            attn_type=args.attn_type,
        ).to(self.device)

        self.mlp = MLP(
            hidden_layers_width=args.mlp_hdw,
            input_size=3*args.d_rnn,
            output_size=args.mlp_out,
            dropout_rate=args.mlp_dropout).to(self.device)
    

    def forward(self, x):
        x = x.float()
        B, S, F_ = x.shape

        # Transformer部分
        loc_seq, mag_seq, t_n_seq = self._batch_to_model_input(x)
        enc_out, non_pad_mask = self.transformer(loc_seq,mag_seq, t_n_seq)
        enc_last, _ = self._process_transformer_out(non_pad_mask, enc_out, x)

        out = self.mlp(enc_last)
        # out = torch.sigmoid(out) # 使用bce with logits
        return out.squeeze(1)
    
    @staticmethod
    def _process_transformer_out(non_pad_mask, enc_out, x):
        lengths = non_pad_mask.sum(dim=1)
        lengths = torch.clamp(lengths, min=1)  
        last_step_index = (lengths.squeeze() - 1).long() 

        batch_idx = torch.arange(non_pad_mask.size(0), device=non_pad_mask.device)  # [B*N]

        # 获取最后一个有效时间步的 mask 和输出
        last_non_pad_mask = non_pad_mask[batch_idx, last_step_index]       # shape: [B*N, 1]
        enc_last = enc_out[batch_idx, last_step_index]                     # shape: [B*N, D]

        return enc_last, last_non_pad_mask

    @staticmethod
    def _batch_to_model_input(bx):
        """
        输入格式:
        ["t", "t_nl", "Magnitude", "Latitude", "Longitude", "Depth"]
        输出: event_loc, event_time, event_magnitude
        """
        t_seq = bx[:, :, 0]              # 原始时间戳（可忽略）
        t_n_seq = bx[:, :, 1]            # 归一化时间
        mag_seq = bx[:, :, [2]]            # 震级
        loc_seq = bx[:, :, 3:5]          # 纬度，经度
        return loc_seq, mag_seq, t_n_seq



class Classifier_STM(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        # 使用包含震级的 Transformer_STM 模型
        self.transformer = Transformer_STM(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            loc_dim=args.loc_dim,
            attn_type=args.attn_type,
        ).to(self.device)

        # MLP 输出分类结果
        self.mlp = MLP(
            hidden_layers_width=args.mlp_hdw,
            input_size=4 * args.d_model,  #  rnn层投影回了d_model
            output_size=args.mlp_out,
            dropout_rate=args.mlp_dropout
        ).to(self.device)

    def forward(self, x):
        x = x.float()
        B, S, F_ = x.shape

        # 拆解输入数据
        loc_seq, time_seq, mag_seq = self._batch_to_model_input(x)

        # Transformer STM 编码器
        enc_out, non_pad_mask = self.transformer(loc_seq, time_seq, mag_seq)
        # 获取每个序列最后有效时间步的输出
        enc_last, _ = self._process_transformer_out(non_pad_mask, enc_out, x)

        # 全连接分类器
        out = self.mlp(enc_last)
        return out.squeeze(1)

    @staticmethod
    def _process_transformer_out(non_pad_mask, enc_out, x):
        length = non_pad_mask.sum(dim=1)  
        last_step_index = (length.squeeze() - 1).long()
        batch_idx = torch.arange(non_pad_mask.size(0), device=non_pad_mask.device)
        enc_last = enc_out[batch_idx, last_step_index]
        return enc_last, non_pad_mask[batch_idx, last_step_index]

    @staticmethod
    def _batch_to_model_input(bx):
        """
        输入格式:
        ["t", "t_nl", "Magnitude", "Latitude", "Longitude", "Depth"]
        输出: event_loc, event_time, event_magnitude
        """
        t_seq = bx[:, :, 0]              # 原始时间戳（可忽略）
        t_n_seq = bx[:, :, 1]            # 归一化时间
        mag_seq = bx[:, :, [2]]            # 震级
        loc_seq = bx[:, :, 3:5]          # 纬度，经度
        return loc_seq, t_n_seq, mag_seq



class ClfAttnPl(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        self.transformer = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            loc_dim=args.dim,
            attn_type=args.attn_type,
        ).to(self.device)

        self.pooling = AttentionPooling(input_dim=3*args.d_model, hidden_dim=3*args.d_model).to(self.device)

        self.mlp = MLP(
            hidden_layers_width=args.mlp_hdw,
            input_size=3 * args.d_model,
            output_size=args.mlp_out,
            dropout_rate=args.mlp_dropout).to(self.device)

    def forward(self, x):
        x = x.float()
        B, S, F_ = x.shape

        f_seq, t_n_seq = self._batch_to_model_input(x)
        enc_out, non_pad_mask = self.transformer(f_seq, t_n_seq)  # [B, L, D], [B, L, 1]
        mask = non_pad_mask.squeeze(-1)  # [B, L]

        pooled_out, attn_weights = self.pooling(enc_out, mask)  # [B, D], [B, L]

        out = self.mlp(pooled_out)
        return out.squeeze(1)

    @staticmethod
    def _batch_to_model_input(bx):
        B, S, F_ = bx.shape
        t_seq = bx[:, :, 0]
        t_n_seq = bx[:, :, 1]
        mag_seq = bx[:, :, 2:3]
        loc_seq = bx[:, :, 3:5]
        dep_seq = bx[:, :, 5]
        f_seq = torch.concat((loc_seq, mag_seq), dim=-1)
        return f_seq, t_n_seq


class ClfAttnPl_T(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        self.transformer = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            device=device,
            loc_dim=args.dim,
            attn_type=args.attn_type,
        ).to(self.device)

        self.pooling = AttentionPooling(input_dim=3*args.d_model+1, hidden_dim=args.d_model).to(self.device)

        self.mlp = MLP(
            hidden_layers_width=args.mlp_hdw,
            input_size=3 * args.d_model+1,
            output_size=args.mlp_out,
            dropout_rate=args.mlp_dropout).to(self.device)

    def forward(self, x):
        x = x.float()
        B, S, F_ = x.shape

        f_seq, t_n_seq = self._batch_to_model_input(x)
        enc_out, non_pad_mask = self.transformer(f_seq, t_n_seq)  # [B, L, D], [B, L, 1]
        mask = non_pad_mask.squeeze(-1)  # [B, L]
        pooling_in = torch.cat((enc_out, t_n_seq.unsqueeze(-1)), dim=-1)
        pooled_out, attn_weights = self.pooling(pooling_in, mask)  # [B, D], [B, L]

        out = self.mlp(pooled_out)
        return out.squeeze(1)

    @staticmethod
    def _batch_to_model_input(bx):
        B, S, F_ = bx.shape
        t_seq = bx[:, :, 0]
        t_n_seq = bx[:, :, 1]
        mag_seq = bx[:, :, 2:3]
        loc_seq = bx[:, :, 3:5]
        dep_seq = bx[:, :, 5]
        f_seq = torch.concat((loc_seq, mag_seq), dim=-1)
        return f_seq, t_n_seq
    

class Regressor(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        # Transformer 初始化
        self.transformer = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn = getattr(args, 'rnn_dropout', 0), 
            device=device,
            loc_dim=args.dim,
            attn_type=args.attn_type,
        ).to(self.device)

        self.mlp = MLP(
            hidden_layers_width=args.mlp_hdw,
            input_size=3*args.d_model,  
            output_size=args.mlp_out,
            dropout_rate=args.mlp_dropout).to(self.device)


    def forward(self, x):
        x = x.float()
        B, S, F_ = x.shape

        # Transformer部分
        f_seq, t_n_seq = self._batch_to_model_input(x)
        enc_out, non_pad_mask = self.transformer(f_seq, t_n_seq)
        enc_last, _ = self._process_transformer_out(non_pad_mask, enc_out, x)

        out = self.mlp(enc_last)
        out = F.softplus(out)  
        return out.squeeze(1)
    
    @staticmethod
    def _process_transformer_out(non_pad_mask, enc_out, x):
        length = non_pad_mask.sum(dim=1)  
        last_step_index = (length.squeeze() - 1).long() 

        batch_idx = torch.arange(non_pad_mask.size(0), device=non_pad_mask.device)  # [B*N]

        # 获取最后一个有效时间步的 mask 和输出
        last_non_pad_mask = non_pad_mask[batch_idx, last_step_index]       # shape: [B*N, 1]
        enc_last = enc_out[batch_idx, last_step_index]                     # shape: [B*N, D]

        return enc_last, last_non_pad_mask

    @staticmethod
    def _batch_to_model_input(bx):
        # ["t", "t_nl", "Magnitude", "Latitude", "Longitude", "Depth"]
        B, S, F_ = bx.shape
        t_seq = bx[:, :, 0]
        t_n_seq = bx[:, :, 1]
        mag_seq = bx[:, :, 2:3]
        loc_seq = bx[:, :, 3:5]
        dep_seq = bx[:, :, 5]
        f_seq = torch.concat((loc_seq, mag_seq), dim=-1)
        return f_seq, t_n_seq


class Counter(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device

        # Transformer 初始化
        self.transformer = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn = getattr(args, 'rnn_dropout', 0), 
            device=device,
            loc_dim=args.dim,
            attn_type=args.attn_type,
        ).to(self.device)

        self.mlp = MLP(
            hidden_layers_width=args.mlp_hdw,
            input_size=3*args.d_model,  
            output_size=args.mlp_out,
            dropout_rate=args.mlp_dropout).to(self.device)

        self.fc = nn.Linear(1, 1).to(self.device)

    def forward(self, x):
        x = x.float()
        B, S, F_ = x.shape

        # Transformer部分
        f_seq, t_n_seq = self._batch_to_model_input(x)
        enc_out, non_pad_mask = self.transformer(f_seq, t_n_seq)
        enc_last, _ = self._process_transformer_out(non_pad_mask, enc_out, x)

        out = self.mlp(enc_last)
        out = F.softplus(out)  
        out = self.fc(out)
        return out.squeeze(1)
    
    @staticmethod
    def _process_transformer_out(non_pad_mask, enc_out, x):
        length = non_pad_mask.sum(dim=1)  
        last_step_index = (length.squeeze() - 1).long() 

        batch_idx = torch.arange(non_pad_mask.size(0), device=non_pad_mask.device)  # [B*N]

        # 获取最后一个有效时间步的 mask 和输出
        last_non_pad_mask = non_pad_mask[batch_idx, last_step_index]       # shape: [B*N, 1]
        enc_last = enc_out[batch_idx, last_step_index]                     # shape: [B*N, D]

        return enc_last, last_non_pad_mask

    @staticmethod
    def _batch_to_model_input(bx):
        # ["t", "t_nl", "Magnitude", "Latitude", "Longitude", "Depth"]
        B, S, F_ = bx.shape
        t_seq = bx[:, :, 0]
        t_n_seq = bx[:, :, 1]
        mag_seq = bx[:, :, 2:3]
        loc_seq = bx[:, :, 3:5]
        dep_seq = bx[:, :, 5]
        f_seq = torch.concat((loc_seq, mag_seq), dim=-1)
        return f_seq, t_n_seq


class LSTM(nn.Module):
    def __init__(self, args,device):
        super().__init__()
        self.device = device
        self.hidden_size = args.lstm_hidden_size
        self.num_layers = args.lstm_num_layers
        self.feature_size =  len(args.feature_cols)
        self.lstm = nn.LSTM(self.feature_size, self.hidden_size, self.num_layers, batch_first=True).to(self.device)
        self.fc = nn.Linear(self.hidden_size,1).to(self.device)
        self.dropout = nn.Dropout(p=args.lstm_dropout)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0]
        if hidden is None:
            h_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = x.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0, c_0 = hidden

        output, (h_0, c_0) = self.lstm(x, (h_0, c_0))
        output = self.dropout(output)
        output = self.fc(output)
        return output[:, -1, :].squeeze(1)