import torch
import torch.nn as nn
from .Layers import MLP, CNN, GCN
from .dstpp.Models import Transformer, Transformer_ST, Transformer_STM


class Classifier(nn.Module):
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
            device=device,
            loc_dim=args.dim,
            CosSin=True,
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
        # out = torch.sigmoid(out) # 使用bce with logits
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
            input_size=4 * args.d_rnn,  # 因为STM输出是4段拼接 (temporal, loc, mag, fused)
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
        mag_seq = bx[:, :, 2]            # 震级
        loc_seq = bx[:, :, 3:5]          # 纬度，经度
        return loc_seq, t_n_seq, mag_seq
