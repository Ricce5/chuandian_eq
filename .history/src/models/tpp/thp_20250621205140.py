import torch
import torch.nn as nn
from src.models.base_model import EventPredictionBaseModel
from src.models.transformer import Transformer_ST
from src.models.layers import ScaledSoftplus
from .thinning import EventSampler

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

        # 强度网络层
        self.layer_intensity_hidden = nn.Linear(args.d_model, self.num_event_types, bias=True).to(self.device)
        self.softplus = ScaledSoftplus(self.num_event_types).to(self.device)
        
        # 事件采样器
        self.event_sampler = EventSampler(
            num_sample=getattr(args, 'num_sample', 1),
            num_exp=getattr(args, 'num_exp', 500),
            over_sample_rate=getattr(args, 'over_sample_rate', 5.0),
            patience_counter=getattr(args, 'patience_counter', 5),
            num_samples_boundary=getattr(args, 'num_samples_boundary', 5),
            dtime_max=getattr(args, 'dtime_max', 5),
            device=self.device
        )

    def forward(self, batch):
        t_seq = batch.arrival_times
        type_seq = batch.type_seq
        enc_out, _ = self.transformer(type_seq, t_seq)
        return enc_out

    def log_likelihood(self, batch):
        time_delta_seqs = batch.inter_times
        type_seq = batch.type_seq
        seq_mask = batch.non_pad_mask
        enc_out = self.forward(batch)

        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(enc_out) + factor_intensity_base
        lambda_at_event = self.softplus(intensity_states)

        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])
        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out, sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)

        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
            lambda_at_event=lambda_at_event,
            lambdas_loss_samples=lambda_t_sample,
            time_delta_seq=time_delta_seqs[:, 1:],
            seq_mask=seq_mask[:, 1:],
            type_seq=type_seq[:, 1:]
        )

        loss = -(event_ll - non_event_ll).sum()
        return loss, num_events
