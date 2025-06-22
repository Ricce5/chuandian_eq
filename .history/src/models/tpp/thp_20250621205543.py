import torch
import torch.nn as nn
from src.models.base_model import EventPredictionBaseModel
from src.models.transformer import Transformer_ST
from src.models.layers import ScaledSoftplus

class THP(EventPredictionBaseModel):
    def __init__(self, args, device):
        super(THP, self).__init__(args, device)

        # 构建 Transformer 编码器
        self.transformer = Transformer_ST(
            d_model=args.d_model,
            d_rnn=args.d_rnn,
            d_inner=args.d_inner,
            n_layers=args.n_layers,
            n_head=args.n_head,
            d_k=args.d_k,
            d_v=args.d_v,
            dropout=args.t_dropout,
            dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
            device=device,
            dim=args.dim,
            attn_type=args.attn_type
        ).to(self.device)

        # 强度因子
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = nn.Linear(args.d_model, self.num_event_types, bias=True).to(self.device)
        self.softplus = ScaledSoftplus(self.num_event_types).to(self.device)

    def forward(self, batch):
        t_seq = batch.arrival_times
        type_seq = batch.type_seq
        enc_out, _ = self.transformer(type_seq, t_seq)
        return enc_out
