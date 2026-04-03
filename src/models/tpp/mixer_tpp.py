from typing import List, Optional, Tuple, Union

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from  src.distributions import Gamma
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
from src.models.mamba.scan_wrapper import  SelectiveScanWrapper
import math 

logger = logging.getLogger(__name__)



class MixerTPP(TPPModel):
    """MixerTPP: Neural Temporal Point Process model with a mixer architecture.

    Args:
        base_model: The backbone mixer model for sequence encoding.
        hypernet_time: Hypernetwork for time distribution parameters.
        hypernet_mag: Hypernetwork for magnitude distribution parameters.
        dropout: Dropout probability.
        predict_b: Whether to predict the Gutenberg-Richter b-value.
        use_b_updater: Whether to use b-value updater distribution.
        loss_weights: Dictionary of loss weights for time, magnitude, and b-value.
        loss_reduction: Reduction method for loss ('sum', 'mean', etc.).
        b_range: Tuple specifying min and max b-value.
        use_adaptive_loss_weights: Whether to use adaptive weighting for b loss.
    """

    def __init__(self, base_model, hypernet_time, hypernet_mag, dropout, 
                 predict_b, use_b_updater=False,
                 loss_weights=None, loss_reduction=None,b_range=None,
                 use_adaptive_loss_weights=False, bg_model=None, b_filter=None, b_init=1.0):
        super().__init__()

        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        # Infer context size (d_model) from a few possible places on the provided base_model.
        def _infer_d_model(bm):
            # Common locations: bm.encoder.d_model, bm.encoder.backbone.d_model, bm.encoder.config.d_model, bm.d_model
            enc = getattr(bm, "encoder", None)
            if enc is not None:
                d = getattr(enc, "d_model", None)
                if d is not None:
                    return d
                cfg = getattr(enc, "config", None)
                if cfg is not None:
                    d = getattr(cfg, "d_model", None)
                    if d is not None:
                        return d
            # try base_model itself
            d = getattr(bm, "d_model", None)
            return d

        self.context_size = _infer_d_model(base_model)
        self.num_extra_features = None
        self.input_magnitude = True
        self.base_model = base_model
        self.device = self.base_model.device

        self.hypernet_time = hypernet_time
        self.hypernet_mag = hypernet_mag
        self.dropout = nn.Dropout(dropout)
        rb = torch.as_tensor(self.base_model.input_adapter.richter_b, device=device, dtype=dtype)
        mc = torch.as_tensor(self.base_model.input_adapter.mag_completeness, device=device, dtype=dtype)
        self.register_buffer("richter_b", rb)
        self.register_buffer("mag_completeness", mc)

        self.num_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # Comment in English.
            + 0 if self.num_extra_features is None else self.num_extra_features
        )
        self.predict_b = predict_b
        self.use_b_updater = use_b_updater
        if loss_weights is None:
            self.weights = {
            "time_weight": 1.0,
            "mag_weight": 1.0,
            "b_weight": 1.0,
            "b_smooth_weight": 0.0,
            }
        else:
            self.weights = loss_weights
            self.weights.setdefault("time_weight", 1.0)
            self.weights.setdefault("mag_weight", 1.0)
            self.weights.setdefault("b_weight", 1.0)
            self.weights.setdefault("b_smooth_weight", 0.0)
        if use_adaptive_loss_weights:
            self.log_sigma2_b = nn.Parameter(torch.zeros(()))
        self.reduction = loss_reduction if loss_reduction is not None else "sum"
        if b_range is not None:
            self.b_min,self.b_max = b_range
        else:
            self.b_min, self.b_max = 0.5, 2.0

        self.b_filter = b_filter
        self.b_init = b_init
        if self.b_filter is not None:
            self.dt_input_proj = nn.Linear(1, self.context_size, bias=False).to(device=device, dtype=dtype)
            self.dt_min = 1e-2
            self.dt_max = 10
        # optional proportional background model (expects ProportionalBGModel-like API)
        self.bg_model = bg_model



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
            return_all: bool = False,
            ):
        """Get the current state of the model for inference."""
        state_all = self.base_model(batch, inference_params=inference_params)
        if inference_params is not None:
            inference_params.seqlen_offset += 1
        if return_all:
            return state_all
        return state_all[:, -1:, :]
    

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
        # weight_logits = F.log_softmax(weight_logits, dim=-1)
        component_dist = dist.Weibull(scale=scale, shape=shape)
        mixture_dist = Categorical(logits=weight_logits)
        return dist.MixtureSameFamily(
            mixture_distribution=mixture_dist,
            component_distribution=component_dist,
        )

    def forward(self, batch):
        enc_output = self.get_context(batch)  # (B, L, C)
        return enc_output
    
    def dt_to_delta(self, inter_times):
        inter_times = inter_times.clamp_min(0)
        weight = self._bounded_weight_tanh(self.dt_min, self.dt_max)
        delta = torch.matmul(inter_times.unsqueeze(-1), weight)     
        return   delta



    def _bounded_weight_tanh(self, min_val: float = 0.01, max_val: float = 1 ):
        """
        Returns a bounded positive projection weight tensor in range [min_val, max_val].
        """
        raw_weight = self.dt_input_proj.weight.view(1, -1)  
        bounded_weight = min_val + (max_val - min_val) * 0.5 * (torch.tanh(raw_weight) + 1)
        return bounded_weight  



    def _get_b_pred(self, context, predict_b: Optional[bool] = None, inter_times: Optional[torch.Tensor] = None, filter_params=None):
        """
        context: (B, L, D)
        inter_times: (B, L) 
        return: (B, L)
        """
        predict_b = self.predict_b if predict_b is None else predict_b
        updated_filter_params = filter_params
        if predict_b:
            use_filter = (self.b_filter is not None) and (inter_times is not None)
            if use_filter:
                delta= self.dt_to_delta(inter_times)
                delta = delta.expand_as(context)
                if filter_params is None:
                    context_ssm = self.b_filter(context, delta)  # (B, L, D)
                else:
                    ssm_state = filter_params
                    if ssm_state.shape[0] != context.shape[0]:
                        ssm_state = ssm_state.expand(context.shape[0], -1, -1, -1).contiguous()
                    context_ssm, updated_filter_params = self.b_filter(
                        context,
                        delta,
                        ssm_state=ssm_state,
                        return_state=True,
                    )  # (B, L, D)
                b_raw = self.hypernet_mag(context_ssm).squeeze(-1)  # (B, L)
                b_raw = b_raw + self.b_init
            else:
                b_raw = self.hypernet_mag(context).squeeze(-1)  # (B, L)
            clamped = b_raw.clamp(self.b_min, self.b_max)
            b_pred = b_raw + (clamped - b_raw).detach()
        else:
            b_pred = context.new_full(context.shape[:2], float(self.richter_b))
        # print(f"b_pred min/max: {b_pred.min().item():.4f}/{b_pred.max().item():.4f}")
        if filter_params is None:
            return b_pred
        return b_pred, updated_filter_params

    def get_magnitude_dist(
        self,
        context,
        predict_b: Optional[bool] = None,
        return_b: bool = False,
        inter_times: Optional[torch.Tensor] = None,
        filter_params=None,
    ):
        if filter_params is not None:
            b_pred, filter_params = self._get_b_pred(
                context, predict_b, inter_times=inter_times, filter_params=filter_params
            )
        else:
            b_pred = self._get_b_pred(context, predict_b, inter_times=inter_times)
        mag_min = context.new_full(context.shape[:2], float(self.mag_completeness))
        gr = dist.GutenbergRichter(b=b_pred, mag_min=mag_min)
        if return_b:
            if filter_params is not None:
                return gr, b_pred, filter_params
            return gr, b_pred
        if filter_params is not None:
            return gr, filter_params
        return gr  

    def compute_b_value(self, seq: Optional[src.data.Sequence] = None, predict_b: bool = False):
        past_batch = src.data.Batch.from_list([seq])
        context = self.get_context(past_batch)
        b_pred = self._get_b_pred(context, predict_b, inter_times=past_batch.inter_times)
        return b_pred

    def get_updater_b_distribution(self,batch):
        gamma =  Gamma(batch.a_t, batch.s_t * torch.log(torch.tensor(10.0, device=batch.device)))
        return gamma

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        predict_b: Optional[bool] = None,   
        use_b_updater: Optional[bool] = None,  
        weights: dict = None,                  
        reduction: str = None,           # "sum" | "mean" | "per_event" | "per_time" | "none"
        eps: float = 1e-10,
    ):
        """
        Return time NLL, magnitude NLL, and weighted total NLL.
        Returns:
            dict(time=..., mag=..., total=...)
        """

        weights = self.weights if weights is None else weights
        predict_b = self.predict_b if predict_b is None else predict_b
        use_b_updater = self.use_b_updater if use_b_updater is None else use_b_updater
        reduction = self.reduction if reduction is None else reduction
        device = batch.inter_times.device

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
        mag_dist, b_pred = self.get_magnitude_dist(context, predict_b=predict_b, return_b=True, inter_times=batch.inter_times)
        mask = batch.nll_event_mask.bool()           # (B, L)
        log_like_mag = mag_dist.log_likelihood(batch.mag, mask)
        nll_mag = -log_like_mag                      # (B,)

        # ---------- Combine ----------
        nll_total = weights["time_weight"] * nll_time + weights["mag_weight"] * nll_mag   # (B,)

        # ---------- b smoothness penalty ----------
        b_smooth = None
        if predict_b and weights.get("b_smooth_weight", 0.0) > 0:
            mask_pairs = batch.nll_event_mask.bool()
            mask_pairs = mask_pairs[:, 1:] & mask_pairs[:, :-1]
            b_diff_sq = (b_pred[:, 1:] - b_pred[:, :-1]) ** 2
            delta_t = batch.inter_times[:, 1:].clamp_min(eps)
            b_smooth = (b_diff_sq / delta_t * mask_pairs).sum(-1)
            nll_total = nll_total + weights["b_smooth_weight"] * b_smooth

        # ---------- b part ----------
        if use_b_updater:
            b_updater_dist = self.get_updater_b_distribution(batch)
            log_like_b = b_updater_dist.log_likelihood(b_pred, batch.nll_event_mask.bool())
            nll_b = -log_like_b
            if hasattr(self, "log_sigma2_b"):
                nll_b_all = nll_b * torch.exp(-self.log_sigma2_b) + self.log_sigma2_b
            else:
                nll_b_all = weights["b_weight"] *nll_b
            nll_total += nll_b_all

        # ---------- background proportional model part ----------
        if getattr(self, "bg_model", None) is not None:
            log_h_intensity = inter_time_dist.log_hazard(batch.inter_times.clamp_min(eps))
            nll_bg = self.bg_model.nll_change(batch, log_h_intensity)  # (B,)
            nll_total = nll_total + nll_bg
        # ---------- Reductions (same rule for all) ---------
        out = {
            "time": nll_time,
            "mag": nll_mag,
            "total": nll_total,
        }

        if b_smooth is not None:
            out["b_smooth"] = b_smooth

        if use_b_updater:
            out["b"] = nll_b
        if getattr(self, "bg_model", None) is not None:
            # nll_bg was computed as per-batch tensor earlier when bg_model present
            out["bg"] = nll_bg
        return self.reduce_nll_dict(out, batch, reduction=reduction, eps=eps)



    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        return_sequences: bool = False,
        predict_b: Optional[bool] = None,
        max_sample_len: int = 1000,
    ) -> Union[src.data.Batch, List[src.data.Sequence]]:

        predict_b = self.predict_b if predict_b is None else predict_b  
        past_seq_len = len(past_seq) if past_seq is not None else 0
        max_seqlen = past_seq_len + max_sample_len

        inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            key_value_memory_dict= self.base_model.encoder.allocate_inference_cache(batch_size=batch_size, max_seqlen=max_seqlen),
        )
        filter_params = None
        if self.b_filter is not None:
            filter_params = self.b_filter.allocate_inference_cache(batch_size=1)

        if past_seq is not None:
            t_start = past_seq.t_end
            buffer_batch = src.data.Batch.init_sample_batch(past_seq=past_seq.init_sample_sequence(), batch_size=batch_size, max_sample_len=max_sample_len)
            past_batch = src.data.Batch.from_list([past_seq])
            state_all = self.get_current_state(
                past_batch[:, :-1], inference_params=inference_params, return_all=True
            )  # (1, L, C)
            current_state = state_all[:, -1:, :].expand(batch_size, -1, -1)  # (B, 1, C)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
            # warm up SSM filter state over the observed history
            if self.b_filter is not None and filter_params is not None:
                _, filter_params = self._get_b_pred(
                    state_all,
                    predict_b,
                    inter_times=past_batch.inter_times[:, 1:],
                    filter_params=filter_params,
                )
        else:
            dtype = next(self.parameters()).dtype
            current_state = torch.zeros(batch_size, 1, self.context_size, device=self.device, dtype=dtype)
            buffer_batch = src.data.Batch.init_sample_batch(
                past_seq=None, batch_size=batch_size, max_sample_len=max_sample_len
            )
            time_remaining = None

        t_end = t_start + duration
        inter_time_list = []
        running_time = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
        mag_list = []

        generated = False
        while not generated:
            inter_time_dist = self.get_inter_time_dist(current_state)
            if time_remaining is None:
                next_inter_times = inter_time_dist.sample()
                if getattr(self, "bg_model", None) is not None:
                    t_last_event = t_start + running_time
                    dt = next_inter_times.squeeze(-1)
                    bg_inter_time = self.bg_model.sample_nhpp_inverse(
                        batch_size,
                        t0=t_last_event,
                        dt=dt,
                    )
                    if (bg_inter_time < 0.0).any():
                        raise ValueError(
                            f"Sampled inter-event time should be non-negative. Got minimum value {bg_inter_time.min()}"
                        )
                    next_inter_times = bg_inter_time.unsqueeze(-1)
            else:
                lower_bound = torch.as_tensor(
                    time_remaining,
                    device=self.device,
                    dtype=running_time.dtype,
                )
                next_inter_times = inter_time_dist.sample_conditional(lower_bound=lower_bound)
                if getattr(self, "bg_model", None) is not None:
                    if past_seq is None:
                        raise ValueError("past_seq must be provided when time_remaining is not None.")
                    t_last_event = torch.as_tensor(
                        past_seq.arrival_times[-1],
                        device=self.device,
                        dtype=running_time.dtype,
                    ).repeat(batch_size)
                    dt = (next_inter_times - lower_bound).squeeze(-1)
                    bg_inter_time = self.bg_model.sample_nhpp_inverse(
                        batch_size,
                        t0=t_last_event + lower_bound,
                        dt=dt,
                    )
                    if (bg_inter_time < 0.0).any():
                        raise ValueError(
                            f"Sampled inter-event time should be non-negative. Got minimum value {bg_inter_time.min()}"
                        )
                    next_inter_times = bg_inter_time.unsqueeze(-1) + lower_bound
                time_remaining = lower_bound

            next_inter_times.clamp_max_(t_end - t_start)
            if time_remaining is not None:
                delta = next_inter_times-time_remaining
                inter_time_list.append(delta)
                assert (delta).min() >= 0
                time_remaining = None
            else:
                inter_time_list.append(next_inter_times)

            running_time += inter_time_list[-1].squeeze(-1)

            if self.b_filter is not None:
                current_inter_times = inter_time_list[-1]
                if current_inter_times.ndim == 1:
                    current_inter_times = current_inter_times.unsqueeze(-1)
                mag_dist, filter_params = self.get_magnitude_dist(
                    context=current_state,
                    predict_b=predict_b,
                    inter_times=current_inter_times,
                    filter_params=filter_params,
                )
            else:
                mag_dist = self.get_magnitude_dist(context=current_state, predict_b=predict_b)
            next_mag = mag_dist.sample()
            mag_list.append(next_mag)

            buffer_batch.update_sample_batch(next_inter_times=next_inter_times, next_mag=next_mag)
            tmp_batch = buffer_batch.get_tmp_batch()

            current_state = self.get_current_state(tmp_batch, inference_params=inference_params)
            generated = running_time.min() >= (t_end - t_start)-1e-6

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
            logger.error("Min last_surv_time: %s", last_surv_time.min().item())
            raise ValueError("last_surv_time < 0 detected")
        inter_times[torch.arange(batch_size), end_idx] = last_surv_time

        batch = src.data.Batch(
            inter_times=inter_times,
            arrival_times=inter_times.cumsum(-1),
            t_start=torch.full([batch_size], t_start, device=self.device, dtype=torch.float32),
            t_end=torch.full([batch_size], t_end, device=self.device, dtype=torch.float32),
            t_nll_start=torch.full([batch_size], t_start, device=self.device, dtype=torch.float32),
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
