import torch
import torch.nn as nn
import torch.nn.functional as F
from .thinning import EventSampler
from src.models.transformer.transformers import Transformer_type

class TppBase(nn.Module):
    def __init__(self, args, device):
        """初始化基类模型，提供通用功能"""
        super().__init__()
        self.device = device
        self.use_mc_samples = getattr(args, 'use_mc_samples', True)  # Use Monte Carlo samples for non-event LL
        self.pad_token_id = getattr(args, 'pad_token_id', -100)
        self.eps = torch.finfo(torch.float32).eps
        self.num_event_types = getattr(args, 'num_event_types', 1)
        self.loss_integral_num_sample_per_step = getattr(args, 'integral_num_sample', 100)  

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
        """Forward pass, to be implemented by the child class."""
        raise NotImplementedError('This method needs to be implemented by the subclass.')

    def make_dtime_loss_samples(self, time_delta_seq):
        """生成时间点样本"""
        dtimes_ratio_sampled = torch.linspace(
            start=0.0, end=1.0, steps=self.loss_integral_num_sample_per_step, device=self.device
        )[None, None, :]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled
        return sampled_dtimes

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """计算事件序列的似然"""
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),
            target=type_seq,
            ignore_index=self.pad_token_id,
            reduction='none',
        )

        # Compute non-event LL
        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else:
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """计算每个采样时间的状态，需在子类中实现"""
        raise NotImplementedError('This method needs to be implemented by the subclass.')

    def compute_intensities_at_sample_times(self, batch, sample_dtimes, **kwargs):
        """计算采样时刻的强度"""
        enc_out = self.forward(batch)
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)
        lambdas = self.softplus(encoder_output)
        return lambdas

    def predict_one_step_at_every_event(self, batch):
        """每个事件的一步预测"""
        batch = batch[:, :-1]
        time_seq, time_delta_seq, event_seq = batch.arrival_times, batch.inter_times, batch.type_seq
        dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max, time_delta_seq + self.event_sampler.dtime_max)

        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
            time_seq, time_delta_seq, event_seq, batch, dtime_boundary,
            self.compute_intensities_at_sample_times, compute_last_step_only=False
        )

        intensities_at_times = self.compute_intensities_at_sample_times(batch, accepted_dtimes)
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        types_pred = torch.argmax(intensities_weighted, dim=-1)
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)
        return dtimes_pred, types_pred

