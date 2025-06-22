
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import ScaledSoftplus
from src.models.transformer.transformers import Transformer_type ,Transformer, Transformer_ST, Transformer_STM, Transformer_SE
from .tpp_base import TppBase

class THP(TppBase):
    def __init__(self, args,base_model,head, device):
        super().__init__(args, device)

        # 与事件强度相关的参数和操作
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = head.to(self.device)    # nn.Linear(args.d_model, self.num_event_types, bias=True).to(self.device)
        self.softplus = ScaledSoftplus(self.num_event_types).to(self.device)

        # Transformer网络
        # self.base_model = Transformer_type(
        #     d_model=args.d_model,
        #     d_rnn=args.d_rnn,
        #     d_inner=args.d_inner,
        #     n_layers=args.n_layers,
        #     n_head=args.n_head,
        #     d_k=args.d_k,
        #     d_v=args.d_v,
        #     dropout=args.t_dropout,
        #     dropout_post_rnn=getattr(args, 'rnn_dropout', 0),
        #     device=device,
        #     attn_type=args.attn_type,
        #     num_event_types_pad=self.num_event_types + 1,
        #     pad_token_id=args.pad_token_id
        # ).to(self.device)
        self.base_model = base_model.to(self.device)

    def forward(self, batch):
        # from src.models.input_adapters import Type_T_BatchInputAdapter
        # inputs = Type_T_BatchInputAdapter()(batch)
        enc_out, _ = self.base_model(batch)
        # return enc_out
        return enc_out

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """实现特定的计算逻辑：根据样本时间更新状态"""
        event_states = event_states[:, :, None, :]
        sample_dtimes = sample_dtimes[..., None]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(event_states) + factor_intensity_base
        return intensity_states

    def log_likelihood(self, batch):
        """计算日志似然"""
        time_delta_seqs = batch.inter_times
        type_seq = batch.type_seq
        seq_mask = batch.non_pad_mask
        enc_out = self.forward(batch[:, :-1])

        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(enc_out) + factor_intensity_base
        lambda_at_event = self.softplus(intensity_states)
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out, sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=seq_mask[:, 1:],
                                                                        type_seq=type_seq[:, 1:])

        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events
