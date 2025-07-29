
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.distributions as dist
from src.models.layers import ScaledSoftplus
from .tpp_cif_model import TppCifModel
import src.data
from typing import Optional, Dict, Any, Union, List, Tuple
from src.utils.mask_utils  import masked_select_per_row

class THP_Cif(TppCifModel):
    def __init__(self, args,base_model,head):
        super().__init__(args,device=base_model.device)
        self.input_magnitude = True
        self.predict_magnitude = True
        self.num_extra_features = 0
        self.num_components = args.num_components
        self.register_buffer("tau_mean", torch.tensor(args.tau_mean, dtype=torch.float32)) 
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("mag_mean", torch.tensor(args.mag_mean, dtype=torch.float32))
        self.register_buffer("time_max", torch.tensor(args.time_max, dtype=torch.float32))
        self.register_buffer("richter_b", torch.tensor(args.richter_b_mle, dtype=torch.float32))
        self.register_buffer(
            "mag_completeness", torch.tensor(args.mag_completeness, dtype=torch.float32)
        )
        
        self.base_model = base_model
        self.device = self.base_model.device
        self.base_model.input_adapter.model = self
        self.base_model = base_model
        self.factor_intensity_base = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        self.factor_intensity_decay = nn.Parameter(torch.empty([1, self.num_event_types], device=self.device)).to(self.device)
        nn.init.xavier_normal_(self.factor_intensity_base)
        nn.init.xavier_normal_(self.factor_intensity_decay)

        self.layer_intensity_hidden = head.to(self.device)    # nn.Linear(args.d_model, self.num_event_types, bias=True).to(self.device)
        self.softplus = ScaledSoftplus(self.num_event_types).to(self.device)
     
    def set_attn_type(self, new_type: str):
        self.base_model.set_attn_type(new_type)

    def set_attn_dropout(self, p: float):
        self.base_model.set_attn_dropout(p)

    def forward(
            self,
            batch: src.data.Batch,
            caches: Optional[Dict[str, Any]] = None
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        enc_out, non_pad_mask, new_cache = self.base_model(batch, caches=caches)
        return enc_out, non_pad_mask, new_cache
       
    
    def get_context(
            self,
            batch: src.data.Batch,
            caches: Optional[Dict[str, Any]] = None
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        enc_out, _, new_cache = self.base_model(batch, caches=caches)
        context = F.pad(enc_out[:, :-1, :], (0, 0, 1, 0))
        if caches is not None:
            return context, new_cache 
        else:
            return context

    def get_magnitude_dist(self, context):
        log_rate = self.hypernet_mag(context).squeeze(-1)  # (B, L)
        b = self.richter_b * torch.ones_like(log_rate)
        mag_min = self.mag_completeness * torch.ones_like(log_rate)
        return dist.GutenbergRichter(b=b, mag_min=mag_min)


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
        # [batch_size, seq_len, num_samples, num_event_types]
        intensity_states = factor_intensity_decay * sample_dtimes + self.layer_intensity_hidden(event_states) + factor_intensity_base
        return intensity_states 
    


    def nll_loss(self, batch):
        time_delta_seqs = batch.inter_times
        type_seq = batch.type_seq
        seq_mask = batch.non_pad_mask
        #
        context = self.get_context(batch)
        #
        factor_intensity_decay = self.factor_intensity_decay[None, ...]
        factor_intensity_base = self.factor_intensity_base[None, ...]
        intensity_states = factor_intensity_decay * time_delta_seqs[:, 1:, None] + self.layer_intensity_hidden(self.forward(batch[:, :-1])) + factor_intensity_base
        lambda_at_event = self.softplus(intensity_states)
        # 
        sample_dtimes = self.make_dtime_loss_samples(time_delta_seqs[:, 1:])

        state_t_sample = self.compute_states_at_sample_times(event_states=self.forward(batch[:, :-1]), sample_dtimes=sample_dtimes)
        lambda_t_sample = self.softplus(state_t_sample)
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(lambda_at_event=lambda_at_event,
                                                                        lambdas_loss_samples=lambda_t_sample,
                                                                        time_delta_seq=time_delta_seqs[:, 1:],
                                                                        seq_mask=seq_mask[:, 1:],
                                                                        type_seq=type_seq[:, 1:])

        loss = - (event_ll - non_event_ll).sum()
        return loss