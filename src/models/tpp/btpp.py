from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import src
import src.distributions as dist
from .tpp_model import TPPModel
from .common.inter_time_decoding import WeibullMixtureDecoder
from .common.sequence_ops import build_sample_batch, evaluate_compensator_from_model
from functools import partial
from src.models.mha.mha_time import MHATime
from src.models.mamba.block import Block
from src.models.mha.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.utils.generation import InferenceParams

class BlockTPP(TPPModel):
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

    def __init__(self, args,device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.input_magnitude = True
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
        self.register_buffer("time_mean", torch.tensor(args.time_mean, dtype=torch.float32))
        self.register_buffer("richter_b", torch.tensor(args.richter_b_mle, dtype=torch.float32))
        self.register_buffer(
            "mag_completeness", torch.tensor(args.mag_completeness, dtype=torch.float32)
        )

        # Decoder for the time distribution
        self.num_time_params = 3 * self.num_components
        self.hypernet_time = nn.Linear(self.context_size, self.num_time_params)
        self.inter_time_decoder = WeibullMixtureDecoder(
            num_components=self.num_components,
            parametrization="legacy",
            scale_range="positive",
            normalize_mixture_logits=True,
        )

        # RNN input features
        if self.input_magnitude:
            # Decoder for magnitude
            self.num_mag_params = 1  # (1 rate)
            self.hypernet_mag = nn.Linear(self.context_size, self.num_mag_params)


        self.num_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  # Comment in English.
            + 0 if self.num_extra_features is None else self.num_extra_features
        )
        D = args.d_model  # batch size, sequence length, embedding dim
        H = 4               # number of heads
        rotary_emb_dim = D // H // 2  # typically half of head_dim
        rotary_emb_scale_base = 1024
        mlp_hidden_dim = 256
        self.layer_idx = 0
        
        self.block = Block(
                dim=D,
                mixer_cls=partial(MHATime,
                                num_heads=H,
                                rotary_emb_dim=rotary_emb_dim,
                                rotary_emb_scale_base=rotary_emb_scale_base,
                                rotary_emb_time_center= self.time_mean/self.tau_mean,
                                causal=True,
                                layer_idx=0),
                mlp_cls=partial(GatedMLP,
                            hidden_features=mlp_hidden_dim,
                            out_features=D),
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32= True,
        ).cuda()  
        self.input_proj = nn.Linear(self.num_inputs, self.context_size)
        self.dropout =  nn.Dropout(args.rnn_dropout)
        self.norm_f = nn.LayerNorm(self.context_size, elementwise_affine=False)
        self.to(self.device)

    def encode_time(self, inter_times):  # apply log transform and centering
        log_tau = torch.log(torch.clamp_min(inter_times, 1e-10)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def normalize_inter_times(self, inter_times):
        """Normalize inter-event times to the range [0, 1]."""
        return (inter_times - self.tau_min) / (self.tau_max - self.tau_min + 1e-10)


    def encode_magnitude(self, mag):  # apply centering
        return mag.unsqueeze(-1) - self.mag_mean

    def encode_extra_features(self, extra_feat):
        return extra_feat



    def get_context(self, batch,inference_params=None):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        feat_list = [self.encode_time(batch.inter_times)]  
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        features = torch.cat(feat_list, dim=-1).contiguous() * batch.input_mask[:, :, None]
        dt_input = self.normalize_inter_times(batch.inter_times)* batch.input_mask
        t_input =   batch.arrival_times * batch.input_mask/self.tau_mean
        hidden_states = self.input_proj(features)
        hidden_states, residual = self.block(
            hidden_states,
            inference_params=inference_params,
            times=t_input
        )

        rnn_output = hidden_states  *batch.input_mask[:, :, None]
        rnn_output = rnn_output[:, :-1, :] 
        output = F.pad(rnn_output, (0, 0, 1, 0)) 
        output = self.dropout(output)
        return output  

    def get_current_state(self, input, inference_params=None, dt_input=None):
        """Get the current state of the model for inference."""
        hidden_states = self.input_proj(input)  # (B, L, C)
        dt_input = self.normalize_inter_times(dt_input)
        current_state,residual = self.block(hidden_states, inference_params=inference_params, dt_input=dt_input)  # (B, L, C)
        return current_state
    

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        return self.inter_time_decoder.from_context(context, self.hypernet_time)

    def forward(self, batch):
        feat_list = [self.encode_time(batch.inter_times)]  # inter-event time from previous to current event
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        features = torch.cat(feat_list, dim=-1).contiguous() * batch.input_mask[:, :, None]
        dt_input = self.normalize_inter_times(batch.inter_times) * batch.input_mask
        rnn_output = self.block(features.contiguous())  # (B, L, C)
        return rnn_output

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
            key_value_memory_dict={self.layer_idx: self.block.allocate_inference_cache(batch_size=batch_size, max_seqlen=max_seqlen)},
        )
        if past_seq is not None:
            t_start = past_seq.t_end
            past_batch = src.data.Batch.from_list([past_seq])
            current_state = self.get_context(past_batch, inference_params)[:, [-1], :]  # (1, 1, C)
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

        batch = build_sample_batch(
            inter_times=inter_times,
            t_start=t_start,
            t_end=t_end,
            device=self.device,
            magnitudes=magnitudes,
            time_dtype=torch.float16,
            epsilon=1e-5,
            check_last_surv_nonnegative=True,
        )

        return batch.to_list() if return_sequences else batch



    def evaluate_compensator(
        self, sequence: src.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return evaluate_compensator_from_model(
            model=self,
            sequence=sequence,
            num_grid_points=num_grid_points,
        )
