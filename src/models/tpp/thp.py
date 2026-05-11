# transformer Hawkes Process (THP) model with intensity free implementation 
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.distributions as dist
from .tpp_model import TPPModel  
from .common.inter_time_decoding import WeibullMixtureDecoder
from .common.sequence_ops import build_sample_batch, evaluate_compensator_from_model
import src.data
from typing import Optional, Dict, Any, Union, List, Tuple

class THP(TPPModel):
    def __init__(self, args,base_model, hypernet_time,hypernet_mag):
        super().__init__()
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

        self.hypernet_time =   hypernet_time # nn.Linear(args.d_model, self.num_time_params).to(self.device)
        self.inter_time_decoder = WeibullMixtureDecoder(
            num_components=self.num_components,
            parametrization="legacy",
            scale_range="positive",
            normalize_mixture_logits=True,
        )

        # RNN input features
        if self.input_magnitude:
            self.hypernet_mag = hypernet_mag # nn.Linear(args.d_model, self.num_mag_params).to(self.device)

        self.base_model = base_model

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

    def get_current_state(
            self,
            batch: src.data.Batch,
            caches: Optional[Dict[str, Any]] = None
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
         enc_out, _, new_cache = self.base_model(batch, caches=caches)
         current_state = enc_out[:,[-1], :] 
         if caches is not None:
            return current_state, new_cache
         else:
            return current_state
     
        
    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        return self.inter_time_decoder.from_context(context, self.hypernet_time)

    def get_magnitude_dist(self, context):
        log_rate = self.hypernet_mag(context).squeeze(-1)  # (B, L)
        b = self.richter_b * torch.ones_like(log_rate)
        mag_min = self.mag_completeness * torch.ones_like(log_rate)
        return dist.GutenbergRichter(b=b, mag_min=mag_min)

    def nll_loss(self, batch: src.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        context = self.get_context(batch)  # (B, L, C)
        inter_time_dist = self.get_inter_time_dist(context)
        log_like = self.time_log_likelihood(
            batch=batch,
            inter_time_dist=inter_time_dist,
            state=context,
            dist_from_state=self.get_inter_time_dist,
            pdf_inter_times=batch.inter_times,
            survival_inter_times=batch.inter_times,
        )
        return -log_like / (batch.t_end - batch.t_nll_start)  # negated as NLL


    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        return_sequences: bool = False,
    ) -> Union[src.data.Batch, List[src.data.Sequence]]:
        self.base_model.eval()  
        if self.input_magnitude != self.predict_magnitude:
            raise ValueError("Sampling is impossible if input_magnitude != predict_magnitude")
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")


        if past_seq is not None:
            t_start = past_seq.t_end
            time_remaining =  t_start - past_seq.arrival_times[-1]
            buffer_batch = src.data.Batch.init_sample_batch(past_seq=past_seq, batch_size=batch_size, max_sample_len=700)
            sample_batch = buffer_batch.get_sample_batch()
            current_state = self.get_current_state(sample_batch)
           
        else:
            current_state = torch.zeros(batch_size, 1, self.args.d_model, device=self.device)
            raise ValueError("past_seq must be provided for sampling")
            time_remaining = None

        t_end = t_start + duration
        inter_time_list = []  # accumulate in a list to avoid frequent concatenation
        if self.predict_magnitude:
            mag_list = []

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(current_state)

            if time_remaining is None:
                next_inter_times = inter_time_dist.sample()  # (B, 1)
            else:
                next_inter_times = inter_time_dist.sample_conditional(lower_bound=time_remaining)
                next_inter_times -= time_remaining
                time_remaining = None
  

            next_inter_times.clamp_max_(t_end - t_start)
            inter_time_list.append(next_inter_times)  # Comment in English.
            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample()  # (B, 1)
                mag_list.append(next_mag)
    
            buffer_batch.update_sample_batch(next_inter_times=next_inter_times, next_mag=next_mag if self.predict_magnitude else None)
            sample_batch = buffer_batch.get_sample_batch()  


            current_state = self.get_current_state(sample_batch)
            current_state = current_state.detach()  # important: prevent graph growth

   
            total_time = torch.cat(inter_time_list, dim=1).sum(-1).min()
            generated = total_time >= (t_end - t_start)

     
        inter_times = torch.cat(inter_time_list, dim=1)  # (B, L)
        if self.predict_magnitude:
            magnitudes = torch.cat(mag_list, dim=1)  # (B, L)
        else:
            magnitudes = None


        batch = build_sample_batch(
            inter_times=inter_times,
            t_start=t_start,
            t_end=t_end,
            device=self.device,
            magnitudes=magnitudes,
            time_dtype=torch.float32,
            epsilon=1e-5,
            check_last_surv_nonnegative=True,
        )

        if return_sequences:
            return batch.to_list()
        else:
            return batch

    @torch.inference_mode()
    def sample_with_cache(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        return_sequences: bool = False,
    ) -> Union[src.data.Batch, List[src.data.Sequence]]:
        self.base_model.eval()
        if self.input_magnitude != self.predict_magnitude:
            raise ValueError("Sampling is impossible if input_magnitude != predict_magnitude")

        cache = self.base_model.encoder.init_cache()
        if past_seq is not None:
            t_start = past_seq.t_end
            time_remaining =  t_start - past_seq.arrival_times[-1]
            buffer_batch = src.data.Batch.init_sample_batch(past_seq=past_seq, batch_size=batch_size, max_sample_len=1500)
            sample_batch = buffer_batch.get_sample_batch()
            current_state, cache = self.get_current_state(sample_batch, cache)
            # print("current_state:", torch.sum(current_state),current_state.shape)

        else:
            current_state = torch.zeros(batch_size, 1, self.args.d_model, device=self.device)
            raise ValueError("past_seq must be provided for sampling")
            time_remaining = None

        t_end = t_start + duration
        inter_time_list = []  # accumulate in a list to avoid frequent concatenation
        if self.predict_magnitude:
            mag_list = []

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(current_state)

            if time_remaining is None:
                next_inter_times = inter_time_dist.sample()  # (B, 1)
            else:
                next_inter_times = inter_time_dist.sample_conditional(lower_bound=time_remaining)
                next_inter_times -= time_remaining
                time_remaining = None
  

            next_inter_times.clamp_max_(t_end - t_start)
            inter_time_list.append(next_inter_times)  # Comment in English.

            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample()  # (B, 1)
                mag_list.append(next_mag)
    
            buffer_batch.update_sample_batch(next_inter_times=next_inter_times, next_mag=next_mag if self.predict_magnitude else None)
            tmp_batch = buffer_batch.get_tmp_batch()  
            current_state,cache = self.get_current_state(tmp_batch, cache)
            current_state = current_state.detach()  # important: prevent graph growth
            cache =cache.detach()

            total_time = torch.cat(inter_time_list, dim=1).sum(-1).min()
            generated = total_time >= (t_end - t_start)

     
        inter_times = torch.cat(inter_time_list, dim=1)  # (B, L)
        if self.predict_magnitude:
            magnitudes = torch.cat(mag_list, dim=1)  # (B, L)
        else:
            magnitudes = None

        batch = build_sample_batch(
            inter_times=inter_times,
            t_start=t_start,
            t_end=t_end,
            device=self.device,
            magnitudes=magnitudes,
            time_dtype=torch.float32,
            epsilon=1e-5,
            check_last_surv_nonnegative=True,
        )

        if return_sequences:
            return batch.to_list()
        else:
            return batch

    def evaluate_compensator(
        self, sequence: src.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return evaluate_compensator_from_model(
            model=self,
            sequence=sequence,
            num_grid_points=num_grid_points,
        )



        




 
