from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.distributions.utils import clamp_preserve_gradients
import numpy as np
import src
import src.distributions as dist
from .tpp_model import TPPModel
from functools import partial
from src.models.mha.mha_time import MHATime
from src.models.mamba.block import Block
from src.models.mamba.mixer_seq import MixerModel
from src.models.mha.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import InferenceParams


class MixerTPP(TPPModel):
    """Neural TPP model with an recurrent encoder.

    Args:
        input_magnitude: Should magnitude be used as model input?
        predict_magnitude: Should the model predict the magnitude?
        num_extra_features: Number of extra features to use as input.
        context_size: Size of the RNN hidden state.
        num_components: Number of mixture components in the output distribution.
        rnn_type: Type of the RNN. Possible choices {'GRU', 'RNN'}
        dropout_proba: Dropout probability.
        tau_mean: Mean inter-event times in the dataset.
        mag_mean: Mean earthquake magnitude in the dataset.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness of the catalog.
        learning_rate: Learning rate used in optimization.
    """

    def __init__(self, args,base_model, hypernet_time,hypernet_mag):
        super().__init__()

        self.predict_magnitude = True
        self.num_extra_features = None
        self.context_size = args.d_model
        self.num_components = args.num_components
        self.register_buffer("tau_mean", torch.tensor(args.tau_mean, dtype=torch.float32))
        self.register_buffer("tau_min", torch.tensor(args.tau_min, dtype=torch.float32))  
        self.register_buffer("tau_max", torch.tensor(args.tau_max, dtype=torch.float32))  
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("mag_mean", torch.tensor(args.mag_mean, dtype=torch.float32))
        self.register_buffer("time_max", torch.tensor(args.time_max, dtype=torch.float32))
        self.register_buffer("richter_b", torch.tensor(args.richter_b_mle, dtype=torch.float32))
        self.register_buffer(
            "mag_completeness", torch.tensor(args.mag_completeness, dtype=torch.float32)
        )
        self.input_magnitude = True
        self.base_model = base_model
        self.device = self.base_model.device
        self.base_model.input_adapter.model = self

        self.hypernet_time = hypernet_time
        self.hypernet_mag = hypernet_mag
        self.dropout = nn.Dropout(p=args.dropout)

        self.num_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # magnitude features 取true或false
            + 0 if self.num_extra_features is None else self.num_extra_features
        )


    def get_context(self, batch, inference_params=None):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        hidden_states = self.base_model(batch, inference_params=inference_params)  # (B, L, C)
        enc_output = hidden_states  *batch.input_mask[:, :, None]
        enc_output = enc_output[:, :-1, :] 
        output = F.pad(enc_output, (0, 0, 1, 0)) 
        output = self.dropout(output)
        return output  

    def get_current_state(self,
            batch: src.data.Batch, 
            inference_params: Optional[InferenceParams] = None,
            ):
        """Get the current state of the model for inference."""
        current_state = self.encoder(batch, inference_params=inference_params)
        return current_state
    

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        params = self.hypernet_time(context)
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
        # params = clamp_preserve_gradients(params, -6.0, 6.0)
        scale, shape, weight_logits = torch.split(
            params,
            [self.num_components, self.num_components, self.num_components],
            dim=-1,
        )
        scale = F.softplus(scale.clamp_min(-5.0))
        shape = F.softplus(shape.clamp_min(-5.0))
        weight_logits = F.log_softmax(weight_logits, dim=-1)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def forward(self, batch):
        enc_output = self.get_context(batch)  # (B, L, C)
        return enc_output

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
        # Inter-event times
        inter_time_dist = self.get_inter_time_dist(context)
        log_pdf = inter_time_dist.log_prob(batch.inter_times.clamp_min(1e-10))  # (B, L) 避免0处概率为0
        log_like = (log_pdf * batch.nll_event_mask).sum(-1) # 对nll区间的事件，上次事件到当前事件的时间间隔的对数概率
        # Survival time from last event until t_end
        arange = torch.arange(batch.batch_size)
        last_surv_context = context[arange, batch.end_idx, :] # end_idx对应生存时间
        last_surv_dist = self.get_inter_time_dist(last_surv_context)
        last_log_surv = last_surv_dist.log_survival(
            batch.inter_times[arange, batch.end_idx]
        )
        log_like = log_like + last_log_surv.squeeze(-1)  # (B,)

        # Remove survival time from t_prev to t_nll_start  # 对第一个事件，计算条件概率，条件是在t_nll_start-t_prev存活
        if torch.any(batch.t_nll_start != batch.t_start):
            prev_surv_context = context[arange, batch.start_idx, :]
            prev_surv_dist = self.get_inter_time_dist(prev_surv_context)
            prev_surv_time = batch.inter_times[arange, batch.start_idx] - (       # nll区间上一个事件到nll区间开始时间
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_log_surv = prev_surv_dist.log_survival(prev_surv_time)
            log_like = log_like - prev_log_surv
        return -log_like / (batch.t_end - batch.t_nll_start)  # (B,)  取了负值


    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        return_sequences: bool = False,
    ) -> Union[src.data.Batch, List[src.data.Sequence]]:
        if self.input_magnitude != self.predict_magnitude:
            raise ValueError("Sampling is impossible if input_magnitude != predict_magnitude")
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")

        past_seq_len = len(past_seq)
        max_generation_len = 2000
        max_seqlen = past_seq_len + max_generation_len

        inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            key_value_memory_dict= self.encoder.allocate_inference_cache(batch_size=batch_size, max_seqlen=max_seqlen),
        )
        if past_seq is not None:
            t_start = past_seq.t_end
            past_batch = src.data.Batch.from_list([past_seq])
            buffer_batch = src.data.Batch.init_sample_batch(past_seq=past_seq, batch_size=batch_size, max_sample_len=1500)
            sample_batch = buffer_batch.get_sample_batch()
            current_state = self.get_current_state(sample_batch)
            inference_params.seqlen_offset += 1
            current_state = current_state.expand(batch_size, -1, -1)  # (B, 1, C)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
        else:
            current_state = torch.zeros(batch_size, 1, self.context_size, device=self.device, dtype=torch.float16)
            time_remaining = None

        t_end = t_start + duration
        inter_time_list = []
        if self.predict_magnitude:
            mag_list = []

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(current_state)
            if time_remaining is None:
                next_inter_times = inter_time_dist.sample()
            else:
                next_inter_times = inter_time_dist.sample_conditional(lower_bound=time_remaining)
                next_inter_times -= time_remaining
                time_remaining = None

            next_inter_times.clamp_max_(t_end - t_start)
            inter_time_list.append(next_inter_times)

            rnn_input_list = [self.encode_time(next_inter_times)] 
            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample()
                mag_list.append(next_mag)
                rnn_input_list.append(self.encode_magnitude(next_mag))

            rnn_input = torch.cat(rnn_input_list, dim=-1).contiguous()
            current_state = self.get_current_state(rnn_input, inference_params=inference_params, dt_input=next_inter_times)
            inference_params.seqlen_offset += 1
            current_state = self.dropout(current_state)
            current_state = current_state.detach()

            total_time = torch.cat(inter_time_list, dim=1).sum(-1).min()
            generated = total_time >= (t_end - t_start)

        inter_times = torch.cat(inter_time_list, dim=1)
        magnitudes = torch.cat(mag_list, dim=1) if self.predict_magnitude else None

        duration = t_end - t_start
        unclipped_arrival_times = inter_times.cumsum(-1)
        epsilon = 1e-5
        padding_mask = unclipped_arrival_times > duration - epsilon
        inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
        end_idx = (1 - padding_mask.long()).sum(-1)
        last_surv_time = duration - inter_times.sum(-1)
        if (last_surv_time < 0).any():
            print("Min last_surv_time:", last_surv_time.min().item())
            raise ValueError("last_surv_time < 0 detected")
        inter_times[torch.arange(batch_size), end_idx] = last_surv_time

        batch = src.data.Batch(
            inter_times=inter_times,
            arrival_times=inter_times.cumsum(-1),
            t_start=torch.full([batch_size], t_start, device=self.device, dtype=torch.float16),
            t_end=torch.full([batch_size], t_end, device=self.device, dtype=torch.float16),
            t_nll_start=torch.full([batch_size], t_start, device=self.device, dtype=torch.float16),
            mask=padding_mask.float(),
            start_idx=torch.zeros(batch_size, device=self.device).long(),
            end_idx=end_idx,
            mag=magnitudes,
        )

        return batch.to_list() if return_sequences else batch



    def evaluate_compensator(
        self, sequence: src.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = src.data.Batch.from_list([sequence])
        context = self.get_context(batch).squeeze(0)  # (L, C)
        inter_time_dist = self.get_inter_time_dist(context)

        # Evaluate each log survival function at times x = [eps, ..., tau_i]
        x = batch.inter_times * torch.linspace(1e-4, 1, num_grid_points)[:, None]
        log_surv = inter_time_dist.log_survival(x)
        # Compute the cumulative sum of log survival functions to get the compensator
        surv_offsets = torch.cat(
            [torch.tensor([0.0]), log_surv[-1].cumsum(dim=-1)[:-1]]
        )
        compensator = -(log_surv + surv_offsets).T.reshape(-1)

        # Shift the inter-event times x to get the global times
        offsets = torch.cat([torch.tensor([0.0]), sequence.arrival_times])
        grid = (x + offsets).T.reshape(-1)
        return grid, compensator
