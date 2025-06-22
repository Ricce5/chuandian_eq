import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.thinning import EventSampler  # 引入 EventSampler

class TppBaseModel(nn.Module):
    def __init__(self, args, device):
        super(TppBaseModel, self).__init__()
        
        self.device = device
        self.eps = torch.finfo(torch.float32).eps
        self.pad_token_id = getattr(args, 'pad_token_id', -100)
        self.num_event_types = getattr(args, 'num_event_types', 1)
        self.loss_integral_num_sample_per_step = getattr(args, 'integral_num_sample', 100)
        self.use_mc_samples = getattr(args, 'use_mc_samples', True)

        # 事件采样器初始化
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
        """
        Forward function to be overridden in the specific model
        """
        raise NotImplementedError

    def log_likelihood(self, batch):
        """
        Compute the log likelihood of the event sequence
        """
        raise NotImplementedError

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        dtimes_ratio_sampled = torch.linspace(start=0.0, end=1.0, steps=self.loss_integral_num_sample_per_step, device=self.device)[None, None, :]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled
        return sampled_dtimes

    def compute_states_at_sample_times(self, event_states, sample_dtimes):
        """Compute the hidden states at sampled times.

        Args:
            event_states (tensor): [batch_size, seq_len, hidden_size].
            sample_dtimes (tensor): [batch_size, seq_len, num_samples].

        Returns:
            tensor: hidden state at each sampled time.
        """
        event_states = event_states[:, :, None, :]
        sample_dtimes = sample_dtimes[..., None]

        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(event_states) + factor_intensity_base
        return intensity_states

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """Compute the loglikelihood of the event sequence

        Args:
            time_delta_seq (tensor): [batch_size, seq_len], time_delta_seq from model input.
            lambda_at_event (tensor): [batch_size, seq_len, num_event_types], unmasked intensity at
            (right after) the event.
            lambdas_loss_samples (tensor): [batch_size, seq_len, num_sample, num_event_types],
            intensity at sampling times.
            seq_mask (tensor): [batch_size, seq_len], sequence mask vector to mask the padded events.
            type_seq (tensor): [batch_size, seq_len], sequence of mark ids, with padded events having a mark of self.pad_token_id

        Returns:
            tuple: event loglike, non-event loglike, intensity at event with padding events masked
        """  
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),
            target=type_seq,
            ignore_index=self.pad_token_id,
            reduction='none'
        )

        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else:
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events

    def predict_one_step_at_every_event(self, batch):
        """One-step prediction for every event in the sequence.

        Args:
            time_seqs (tensor): [batch_size, seq_len].
            time_delta_seqs (tensor): [batch_size, seq_len].
            type_seqs (tensor): [batch_size, seq_len].

        Returns:
            tuple: tensors of dtime and type prediction, [batch_size, seq_len].
        """
        batch = batch[:, :-1]
        time_seq, time_delta_seq, event_seq = batch.arrival_times, batch.inter_times, batch.type_seq
        dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max, time_delta_seq + self.event_sampler.dtime_max)

        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(
            time_seq,
            time_delta_seq,
            event_seq,
            batch,
            dtime_boundary,
            self.compute_intensities_at_sample_times,
            compute_last_step_only=False
        )

        intensities_at_times = self.compute_intensities_at_sample_times(batch, accepted_dtimes)
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        types_pred = torch.argmax(intensities_weighted, dim=-1)
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)
        return dtimes_pred, types_pred
