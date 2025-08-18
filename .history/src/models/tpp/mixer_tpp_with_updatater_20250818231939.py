from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Gamma
from src.distributions.utils import clamp_preserve_gradients
import numpy as np
import src
import src.distributions as dist
from .tpp_model import TPPModel
from functools import partial
from src.models.mha.mha_time import MHATime
from src.models.mamba.block import Block
from src.models.mamba.mixer_seq import MixerModel
from src.models.mamba.scan_wrapper import BoundedSelectiveScanWrapper
from src.models.mha.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import InferenceParams


class MixerTPP(TPPModel):
    """Neural TPP model with an recurrent encoder.

    Args:
        input_magnitude: Should magnitude be used as model input?
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

    def __init__(self, base_model, hypernet_time, hypernet_mag, dropout, predict_b, ssm_filter=None):
        super().__init__()

        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        self.num_extra_features = None
        self.input_magnitude = True
        self.base_model = base_model
        self.device = self.base_model.device

        self.hypernet_time = hypernet_time
        self.hypernet_mag = hypernet_mag
        self.ssm_filter = ssm_filter
        self.dropout = nn.Dropout(dropout)
        rb = torch.as_tensor(self.base_model.input_adapter.richter_b, device=device, dtype=dtype)
        mc = torch.as_tensor(self.base_model.input_adapter.mag_completeness, device=device, dtype=dtype)
        self.register_buffer("richter_b", rb)
        self.register_buffer("mag_completeness", mc)

        self.num_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # magnitude features 取true或false
            + 0 if self.num_extra_features is None else self.num_extra_features
        )
        self.predict_b = predict_b
        
       

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
        current_state = self.base_model(batch, inference_params=inference_params)
        return current_state[:, -1:, :]
    

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        params = self.hypernet_time(context)
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
        # params = clamp_preserve_gradients(params, -6.0, 6.0)
        num_components = params.shape[-1] // 3
        scale, shape, weight_logits = torch.split(params, [num_components, num_components, num_components], dim=-1)

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
    
    def _get_b_pred(self, context, predict_b: Optional[bool] = None):
        if predict_b:
            b_delta = self.hypernet_mag(context)
            if self.ssm_filter is not None:
                b_pred = self.ssm_filter(b_delta).squeeze(-1)
            else:
                b_min, b_max = 0.5, 2.0
                b_pred = b_min + (b_max - b_min) * torch.sigmoid(b_delta)
        else:
            b_pred = context.new_full(context.shape[:2], float(self.richter_b))
        return b_pred

    def get_magnitude_dist(self, context, predict_b: Optional[bool] = None, return_b: bool = False):
        b_pred = self._get_b_pred(context, predict_b)
        mag_min = context.new_full(context.shape[:2], float(self.mag_completeness))
        gr = dist.GutenbergRichter(b=b_pred, mag_min=mag_min)
        if return_b:
            return gr, b_pred
        return gr  

    def compute_b_value(self, seq: Optional[src.data.Sequence] = None, predict_b: bool = False):
        past_batch = src.data.Batch.from_list([seq])
        context = self.get_context(past_batch)
        b_pred = self._get_b_pred(context, predict_b)
        return b_pred

    def get_updater_b_distribution(self,batch):
        
        return dist.GutenbergRichter(b=b_pred)

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        predict_b: Optional[bool] = None,      # True=预测 b，False=常数 b（仍计算震级似然）
        mag_weight: float = 1.0,       # 震级似然权重
        reduction: str = "per_time",   # "sum" | "mean" | "per_event" | "per_time" | "none"
        eps: float = 1e-10,
    ):
        """
        分别返回时间NLL、震级NLL和加权总NLL。
        Returns:
            dict(time=..., mag=..., total=...)  # 张量或标量，取决于 reduction
        """
        if predict_b is None:
            predict_b = self.predict_b
        device = batch.inter_times.device

        def _reduce(x, reduction: str):
            # x: (B,)
            if reduction == "sum":
                return x.sum()
            elif reduction == "mean":
                return x.mean()
            elif reduction == "per_event":
                num_events = batch.nll_event_mask.sum(-1)  # (B,)
                return (x / num_events.clamp_min(1)).to(x.dtype)
            elif reduction == "per_time":
                span = (batch.t_end - batch.t_nll_start)  # (B,)
                return (x / span.clamp_min(eps)).to(x.dtype)
            elif reduction == "none":
                return x  # (B,)
            else:
                raise ValueError(f"Unknown reduction: {reduction}")

        # ---------- Context ----------
        context = self.get_context(batch)  # (B, L, C)

        # ---------- Time part ----------
        inter_time_dist = self.get_inter_time_dist(context)
        log_pdf_time = inter_time_dist.log_prob(batch.inter_times.clamp_min(eps))  # (B, L)
        log_like_time = (log_pdf_time * batch.nll_event_mask).sum(-1)  # (B,)

        # last survival
        arange = torch.arange(batch.batch_size, device=device)
        last_surv_context = context[arange, batch.end_idx, :]         # (B, C)
        last_surv_dist = self.get_inter_time_dist(last_surv_context)  # batched
        last_surv_time = batch.inter_times[arange, batch.end_idx].clamp_min(eps)
        last_log_surv = last_surv_dist.log_survival(last_surv_time).squeeze(-1)  # (B,)
        log_like_time = log_like_time + last_log_surv

        # subtract survival from t_prev to t_nll_start
        if torch.any(batch.t_nll_start != batch.t_start):
            prev_surv_context = context[arange, batch.start_idx, :]
            prev_surv_dist = self.get_inter_time_dist(prev_surv_context)
            prev_surv_time = batch.inter_times[arange, batch.start_idx] - (
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_log_surv = prev_surv_dist.log_survival(prev_surv_time.clamp_min(eps)).squeeze(-1)
            log_like_time = log_like_time - prev_log_surv

        nll_time = -log_like_time  # (B,)

        # ---------- Magnitude part ----------
        mag_dist, b_pred = self.get_magnitude_dist(context, predict_b=predict_b, return_b=True)
        mask = batch.nll_event_mask.bool()           # (B, L)
        log_like_mag = mag_dist.log_likelihood(batch.mag, mask)
        print(torch.sum(log_like_mag))
        nll_mag = -log_like_mag                      # (B,)
        # ---------- Combine ----------
        nll_total = nll_time + mag_weight * nll_mag  # (B,)

        # ---------- Reductions (same rule for all) ----------
        out = {
            "time": _reduce(nll_time, reduction),
            "mag":  _reduce(nll_mag,  reduction),
            "total": _reduce(nll_total, reduction),
        }
        return out



    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        return_sequences: bool = False,
        predict_b: Optional[bool] = None,
    ) -> Union[src.data.Batch, List[src.data.Sequence]]:

        if predict_b is None:
            predict_b = self.predict_b
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")

        past_seq_len = len(past_seq)
        max_sample_len = 2000
        max_seqlen = past_seq_len + max_sample_len

        inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            key_value_memory_dict= self.base_model.encoder.allocate_inference_cache(batch_size=batch_size, max_seqlen=max_seqlen),
        )
        if past_seq is not None:
            t_start = past_seq.t_end
            buffer_batch = src.data.Batch.init_sample_batch(past_seq=past_seq, batch_size=batch_size, max_sample_len=max_sample_len)
            sample_batch = buffer_batch.get_tmp_batch()
            current_state = self.get_current_state(sample_batch)
            inference_params.seqlen_offset += 1
            current_state = current_state.expand(batch_size, -1, -1)  # (B, 1, C)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
        else:
            dtype = next(self.parameters()).dtype
            current_state = torch.zeros(batch_size, 1, self.context_size, device=self.device, dtype=dtype)
            time_remaining = None

        t_end = t_start + duration
        inter_time_list = []
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


            if self.ssm_filter is not None:
                mag_dist = self.get_magnitude_dist(context= current_state,predict_b= predict_b)
            else:
                raise NotImplemented
            next_mag = mag_dist.sample()
            mag_list.append(next_mag)

            buffer_batch.update_sample_batch(next_inter_times=next_inter_times, next_mag=next_mag )
            sample_batch = buffer_batch.get_tmp_batch()  

            current_state = self.get_current_state(sample_batch, inference_params=inference_params)
            inference_params.seqlen_offset += 1
            current_state = self.dropout(current_state)
            current_state = current_state.detach()

            total_time = torch.cat(inter_time_list, dim=1).sum(-1).min()
            generated = total_time >= (t_end - t_start)

        inter_times = torch.cat(inter_time_list, dim=1)
        magnitudes = torch.cat(mag_list, dim=1)

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
