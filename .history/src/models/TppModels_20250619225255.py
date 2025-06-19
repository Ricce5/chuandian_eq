import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import MLP, CNN, AttentionPooling,ScaledSoftplus
from src.models.transformer.transformers import Transformer_type ,Transformer, Transformer_ST, Transformer_STM, Transformer_SE
from src.models.thinning import EventSampler

class THP(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        self.use_mc_samples = getattr(args, 'use_mc_samples', True)  # Use Monte Carlo samples for non-event LL
        self.pad_token_id = getattr(args, 'pad_token_id', -100)
        self.eps = torch.finfo(torch.float32).eps
        self.num_event_types = getattr(args, 'num_event_types', 1)
        self.loss_integral_num_sample_per_step = getattr(args, 'integral_num_sample', 100)  

        # self.transformer = Transformer_ST(
        #     d_model=args.d_model,
        #     d_rnn=args.d_rnn,
        #     d_inner=args.d_inner,
        #     n_layers=args.n_layers,
        #     n_head=args.n_head,
        #     d_k=args.d_k,
        #     d_v=args.d_v,
        #     dropout=args.t_dropout,
        #     dropout_post_rnn = getattr(args, 'rnn_dropout', 0), 
        #     device=device,
        #     loc_dim=args.dim,
        #     CosSin=True,
        #     attn_type=args.attn_type,
        # ).to(self.device)
        self.transformer = Transformer_type(
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
            attn_type=args.attn_type,
            num_event_types_pad=self.num_event_types+1,
            pad_token_id=args.pad_token_id
        ).to(self.device)
       
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = nn.Linear(args.d_model, getattr(args, 'num_event_types', 1), bias=True).to(self.device)  # 3*d_model
        self.softplus = ScaledSoftplus(self.num_event_types).to(self.device)

        self.event_sampler = EventSampler(num_sample= getattr(args, 'num_sample', 1),
                                              num_exp=getattr(args, 'num_exp', 500),
                                              over_sample_rate=getattr(args, 'over_sample_rate', 5.0),
                                              patience_counter=getattr(args, 'patience_counter', 5),
                                              num_samples_boundary=getattr(args, 'num_samples_boundary', 5),
                                              dtime_max=getattr(args, 'dtime_max', 5),
                                              device=self.device)

    def forward(self, batch):
        # fea_seq = torch.cat([batch.loc,batch.mag[...,None]],dim=-1)
        t_seq = batch.arrival_times
        type_seq = batch.type_seq
        # enc_out, _ = self.transformer(fea_seq,t_seq)
        enc_out, _ = self.transformer(type_seq,t_seq)
        return enc_out
    
    def log_likelihood(self, batch): 
        time_delta_seqs = batch.inter_times
        type_seq = batch.type_seq
        seq_mask = batch.non_pad_mask
        enc_out = self.forward(batch[:, :-1]) 

        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(
            enc_out) + factor_intensity_base
        lambda_at_event = self.softplus(intensity_states)
        # print("lambda_at_event stats:", lambda_at_event.mean().item(), lambda_at_event.max().item())
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        state_t_sample = self.compute_states_at_sample_times(event_states=enc_out,
                                                             sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=seq_mask[:, 1:],
                                                                        type_seq=type_seq[:, 1:])

        loss = - (event_ll - non_event_ll).sum()
        return loss, num_events

    def make_dtime_loss_samples(self, time_delta_seq):
        """Generate the time point samples for every interval.

        Args:
            time_delta_seq (tensor): [batch_size, seq_len].

        Returns:
            tensor: [batch_size, seq_len, n_samples]
        """
        # [1, 1, n_samples]
        dtimes_ratio_sampled = torch.linspace(start=0.0,
                                              end=1.0,
                                              steps=self.loss_integral_num_sample_per_step,
                                              device=self.device)[None, None, :]

        # [batch_size, max_len, n_samples]
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
        # [batch_size, seq_len, 1, hidden_size]
        event_states = event_states[:, :, None, :]

        # [batch_size, seq_len, num_samples, 1]
        sample_dtimes = sample_dtimes[..., None]

        # [1, 1, 1, num_event_types]
        factor_intensity_decay = self.factor_intensity_decay[None, None, ...]
        factor_intensity_base = self.factor_intensity_base[None, None, ...]

        # update time decay based on Equation (6)
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(
            event_states) + factor_intensity_base

        return intensity_states
    
    def compute_loglikelihood(self, time_delta_seq, lambda_at_event, lambdas_loss_samples, seq_mask, type_seq):
        """Compute the loglikelihood of the event sequence based on Equation (8) of NHP paper.

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
        # First, add an epsilon to every marked intensity for stability
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        log_marked_event_lambdas = lambda_at_event.log()
        total_sampled_lambdas = lambdas_loss_samples.sum(dim=-1)

        # Compute event LL - [batch_size, seq_len]
        event_ll = -F.nll_loss(
            log_marked_event_lambdas.permute(0, 2, 1),  # mark dimension needs to come second, not third to match nll_loss specs
            target=type_seq,
            ignore_index=self.pad_token_id,  # Padded events have a pad_token_id as a value
            reduction='none', # Does not aggregate, and replaces what would have been the log(marked intensity) with 0.
        )

        # Compute non-event LL [batch_size, seq_len]
        # interval_integral = length_interval * average of sampled lambda(t)
        if self.use_mc_samples:
            non_event_ll = total_sampled_lambdas.mean(dim=-1) * time_delta_seq * seq_mask
        else: # Use trapezoid rule
            non_event_ll = 0.5 * (total_sampled_lambdas[..., 1:] + total_sampled_lambdas[..., :-1]).mean(dim=-1) * time_delta_seq * seq_mask

        num_events = torch.masked_select(event_ll, event_ll.ne(0.0)).size()[0]
        return event_ll, non_event_ll, num_events
    
    
    def compute_intensities_at_sample_times(self,
                                            batch,
                                            sample_dtimes,
                                            **kwargs):
        """Compute hidden states at sampled times.

        Args:
            time_seqs (tensor): [batch_size, seq_len], times seqs.
            time_delta_seqs (tensor): [batch_size, seq_len], time delta seqs.
            type_seqs (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_samples], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, seq_len, num_samples, num_event_types], intensity at all sampled times.
        """

        compute_last_step_only = kwargs.get('compute_last_step_only', False)

        # [batch_size, seq_len, num_samples]
        enc_out = self.forward(batch)

        # [batch_size, seq_len, num_samples, hidden_size]
        encoder_output = self.compute_states_at_sample_times(enc_out, sample_dtimes)

        if compute_last_step_only:
            lambdas = self.softplus(encoder_output[:, -1:, :, :])
        else:
            # [batch_size, seq_len, num_samples, num_event_types]
            lambdas = self.softplus(encoder_output)
        return lambdas
    
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

        # remove the last event, as the prediction based on the last event has no label
        # note: the first dts is 0
        # [batch_size, seq_len]
        time_seq, time_delta_seq, event_seq = batch.arrival_times, batch.inter_times, batch.type_seq
        # [batch_size, seq_len]
        dtime_boundary = torch.max(time_delta_seq * self.event_sampler.dtime_max,
                                    time_delta_seq + self.event_sampler.dtime_max) 

        # [batch_size, seq_len, num_sample]
        accepted_dtimes, weights = self.event_sampler.draw_next_time_one_step(time_seq,
                                                                                time_delta_seq,
                                                                                event_seq,
                                                                                batch,
                                                                                dtime_boundary,
                                                                                self.compute_intensities_at_sample_times,
                                                                                compute_last_step_only=False)  # make it explicit

        # We should condition on each accepted time to sample event mark, but not conditioned on the expected event time.
        # 1. Use all accepted_dtimes to get intensity.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_at_times = self.compute_intensities_at_sample_times(batch,accepted_dtimes)

        # 2. Normalize the intensity over last dim and then compute the weighted sum over the `num_sample` dimension.
        # Each of the last dimension is a categorical distribution over all marks.
        # [batch_size, seq_len, num_sample, num_marks]
        intensities_normalized = intensities_at_times / intensities_at_times.sum(dim=-1, keepdim=True)

        # 3. Compute weighted sum of distributions and then take argmax.
        # [batch_size, seq_len, num_marks]
        intensities_weighted = torch.einsum('...s,...sm->...m', weights, intensities_normalized)

        # [batch_size, seq_len]
        types_pred = torch.argmax(intensities_weighted, dim=-1)

        # [batch_size, seq_len]
        dtimes_pred = torch.sum(accepted_dtimes * weights, dim=-1)  # compute the expected next event time
        return dtimes_pred, types_pred