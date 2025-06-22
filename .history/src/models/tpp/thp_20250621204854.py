import torch
import torch.nn as nn
import torch.nn.functional as F

class TppBaseModel(nn.Module):
    def __init__(self, args, device):
        super(TppBaseModel, self).__init__()
        self.device = device
        self.eps = torch.finfo(torch.float32).eps
        self.pad_token_id = getattr(args, 'pad_token_id', -100)
        self.num_event_types = getattr(args, 'num_event_types', 1)
        self.loss_integral_num_sample_per_step = getattr(args, 'integral_num_sample', 100)
        self.use_mc_samples = getattr(args, 'use_mc_samples', True)

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
        """
        Generate time point samples for loss calculation
        """
        dtimes_ratio_sampled = torch.linspace(start=0.0, end=1.0, steps=self.loss_integral_num_sample_per_step, device=self.device)[None, None, :]
        sampled_dtimes = time_delta_seq[:, :, None] * dtimes_ratio_sampled
        return sampled_dtimes

    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """
        Compute the loglikelihood of the event sequence
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
