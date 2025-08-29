from typing import List, Optional, Tuple, Union

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

    def __init__(self, base_model, hypernet_time, hypernet_mag, dropout, 
                 predict_b, ssm_filter=None, use_b_updater=False,
                 loss_weights=None, loss_reduction=None,b_range=None):
        super().__init__()

        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        self.context_size = getattr(base_model, "d_model", None)
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
        self.use_b_updater = use_b_updater
        if loss_weights is None:
            self.weights = {
            "time_weight": 1.0,
            "mag_weight": 1.0,
            "b_weight": 1.0
            }
        else:
            self.weights = loss_weights
            self.weights.setdefault("time_weight", 1.0)
            self.weights.setdefault("mag_weight", 1.0)
            self.weights.setdefault("b_weight", 1.0)
        self.reduction = loss_reduction if loss_reduction is not None else "sum"
        if b_range is not None:
            self.b_min,self.b_max = b_range
        else:
            self.b_min, self.b_max = 0.5, 2.0

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
        if inference_params is not None:
            inference_params.seqlen_offset += 1
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
    
    def _get_b_pred(self, context, predict_b: Optional[bool] = None):
        if predict_b:
            b_delta = self.hypernet_mag(context)
            if self.ssm_filter is not None:
                b_pred = self.ssm_filter(b_delta)
            else:
                b_pred = self.b_min + (self.b_max - self.b_min) * 0.5 * (torch.tanh(b_delta) + 1)
                b_pred = b_pred
            b_pred = b_pred.squeeze(-1)
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
        gamma =  Gamma(batch.a_t, batch.s_t * torch.log(torch.tensor(10.0, device=batch.device)))
        return gamma

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        predict_b: Optional[bool] = None,      # True=预测 b，False=常数 b（仍计算震级似然）
        use_b_updater: Optional[bool] = None,  # 是否使用b的更新器
        weights: dict = None,                   # 字典化配置各部分权重
        reduction: str = None,           # "sum" | "mean" | "per_event" | "per_time" | "none"
        eps: float = 1e-10,
    ):
        """
        分别返回时间NLL、震级NLL和加权总NLL。
        Returns:
            dict(time=..., mag=..., total=...)  # 张量或标量，取决于 reduction
        """

        weights = self.weights if weights is None else weights
        predict_b = self.predict_b if predict_b is None else predict_b
        use_b_updater = self.use_b_updater if use_b_updater is None else use_b_updater
        reduction = self.reduction if reduction is None else reduction
        device = batch.inter_times.device

        def _reduce(x, mode: str):
            # x: (B,)
            if mode == "sum":
                return x.sum()
            elif mode == "mean":
                return x.mean()
            elif mode == "per_event":
                num_events = batch.nll_event_mask.sum(-1)  # (B,)
                return (x / num_events.clamp_min(1)).to(x.dtype)
            elif mode == "per_time":
                span = (batch.t_end - batch.t_nll_start)  # (B,)
                return (x / span.clamp_min(eps)).to(x.dtype)
            elif mode == "none":
                return x  # (B,)
            else:
                raise ValueError(f"Unknown mode: {mode}")

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
        nll_mag = -log_like_mag                      # (B,)

        # ---------- Combine ----------
        nll_total = weights["time_weight"] * nll_time + weights["mag_weight"] * nll_mag   # (B,)
        # ---------- b part ----------
        if use_b_updater:
            b_updater_dist = self.get_updater_b_distribution(batch)
            log_like_b = b_updater_dist.log_likelihood(b_pred, batch.nll_event_mask.bool())
            nll_b = -log_like_b
            nll_total += weights["b_weight"] * nll_b

        # ---------- Reductions (same rule for all) ---------
        out = {
            "time": _reduce(nll_time, reduction),
            "mag":  _reduce(nll_mag,  reduction),
            "total": _reduce(nll_total, reduction),
        }

        # 只有在 use_b_updater 为 True 时，才返回 "b" 字段
        if use_b_updater:
            out["b"] = _reduce(nll_b, reduction)
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

        predict_b = self.predict_b if predict_b is None else predict_b  
        past_seq_len = len(past_seq) if past_seq is not None else 0
        max_sample_len = 1000
        max_seqlen = past_seq_len + max_sample_len

        inference_params = InferenceParams(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            key_value_memory_dict= self.base_model.encoder.allocate_inference_cache(batch_size=batch_size, max_seqlen=max_seqlen),
        )
        if past_seq is not None:
            t_start = past_seq.t_end
            buffer_batch = src.data.Batch.init_sample_batch(past_seq=past_seq.init_sample_sequence(), batch_size=batch_size, max_sample_len=max_sample_len)
            current_state = self.get_current_state(src.data.Batch.from_list([past_seq])[:,:-1],inference_params=inference_params)
            current_state = current_state.expand(batch_size, -1, -1)  # (B, 1, C)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
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
            else:
                next_inter_times = inter_time_dist.sample_conditional(lower_bound=time_remaining)

            next_inter_times.clamp_max_(t_end - t_start)
            if time_remaining is not None:
                delta = next_inter_times-time_remaining
                inter_time_list.append(delta)
                assert (delta).min() >= 0
                time_remaining = None
            else:
                inter_time_list.append(next_inter_times)

            running_time += inter_time_list[-1].squeeze(-1)

            if self.ssm_filter is None:
                mag_dist = self.get_magnitude_dist(context= current_state,predict_b= predict_b)
            else:
                raise NotImplemented
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
            print("Min last_surv_time:", last_surv_time.min().item())
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
