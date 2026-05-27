#  pytorch implementation of the RECAST model
# ref: https://zenodo.org/records/8161777 Using Deep Learning for Flexible and Scalable Earthquake Forecasting
import inspect
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import src
import src.distributions as dist

from ..common.inter_time_decoding import WeibullMixtureDecoder
from .sampling import RecurrentTPPSamplingMixin
from .utils import run_rnn_with_chunking
from ..tpp_model import TPPModel


class RecurrentTPP(RecurrentTPPSamplingMixin, TPPModel):
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

    def __init__(self, args,device=None,bg_model=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.input_magnitude = True
        self.predict_magnitude = True
        self.num_extra_features = None
        self.context_size = args.d_model
        self.num_components = args.num_components
        self.num_rnn_layers = getattr(args, "num_rnn_layers", 1)
        self.scale_range = getattr(args, "scale_range", "positive")
        if self.scale_range not in ["positive", "decay"]:
            raise ValueError("scale_range must be one of ['positive', 'decay']")
        self.register_buffer("tau_mean", torch.tensor(args.tau_mean, dtype=torch.float32))  # average inter-event interval
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("mag_mean", torch.tensor(args.mag_mean, dtype=torch.float32))
        self.register_buffer("time_max", torch.tensor(args.time_max, dtype=torch.float32))
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
            scale_range=self.scale_range,
            normalize_mixture_logits=True,
        )

        # RNN input features
        if self.input_magnitude:
            # Decoder for magnitude
            self.num_mag_params = 1  # (1 rate)
            self.hypernet_mag = nn.Linear(self.context_size, self.num_mag_params)

        if args.rnn_type not in ["RNN", "GRU", ]:
            raise ValueError(
                f"rnn_type must be one of ['RNN', 'GRU'] " f"(got {args.rnn_type})"
            )
        self.num_rnn_inputs = (
            1  # inter-event times
            + int(self.input_magnitude)  
            + 0 if self.num_extra_features is None else self.num_extra_features
        )
        self.rnn = getattr(nn, args.rnn_type)(
            self.num_rnn_inputs, self.context_size, num_layers=self.num_rnn_layers, batch_first=True,
        )
        self.rnn_chunk_len = int(getattr(args, "rnn_chunk_len", 60000))
        if self.rnn_chunk_len <= 0:
            self.rnn_chunk_len = 60000
        # from src.utils.utils import init_rnn_weights 
        # init_rnn_weights(self.rnn,seed=42) 
        # from src.utils.utils import print_weight_sum
        # print_weight_sum(self.rnn, name="RNN weights", verbose=True)
        self.dropout = nn.Dropout(args.rnn_dropout)
        self.bg_model = bg_model
        self.weights = self._resolve_loss_weights(args)
        self.reduction = getattr(args, "loss_reduction", "per_time")
        self.to(self.device)

    @staticmethod
    def _resolve_loss_weights(args) -> dict[str, float]:
        raw_cfg = dict(getattr(args, "loss_weights", {}) or {})
        if not raw_cfg:
            for legacy_key in ("bg_weight", "bg_kl_weight", "bg_norm_weight"):
                if hasattr(args, legacy_key):
                    raw_cfg[legacy_key] = getattr(args, legacy_key)
        weights = {
            "bg_weight": float(raw_cfg.pop("bg_weight", 1.0)),
            "bg_kl_weight": float(raw_cfg.pop("bg_kl_weight", 1.0)),
            "bg_norm_weight": float(raw_cfg.pop("bg_norm_weight", 0.0)),
        }
        for key, value in weights.items():
            if value < 0.0:
                raise ValueError(f"{key} must be >= 0, got {value}.")
        if raw_cfg:
            for stale_key in ("time_weight", "mag_weight", "b_weight", "b_smooth_weight"):
                raw_cfg.pop(stale_key, None)
        if raw_cfg:
            unknown = ", ".join(sorted(raw_cfg.keys()))
            raise ValueError(f"Unknown loss_weights keys for rtpp: {unknown}")
        return weights

    def encode_time(self, inter_times):  # apply log transform and centering
        log_tau = torch.log(torch.clamp_min(inter_times, 1e-10)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def encode_magnitude(self, mag):  # apply centering
        return mag.unsqueeze(-1) - self.mag_mean

    def encode_extra_features(self, extra_feat):
        return extra_feat

    def _run_rnn(self, features: torch.Tensor, hidden_state=None):
        """Run recurrent backbone with optional sequence chunking."""
        return run_rnn_with_chunking(
            rnn=self.rnn,
            features=features,
            context_size=self.context_size,
            chunk_len=self.rnn_chunk_len,
            hidden_state=hidden_state,
        )

    def get_context(self, batch):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        # print(f"batch.inter_times { torch.sum(batch.inter_times*batch.input_mask[:, :, None]) }{batch.inter_times.shape}, {batch.inter_times[0,:]}")
        # print(f"batch.mag { torch.sum(batch.mag[:,:-1]) }{torch.sum(batch.mag*batch.input_mask)}{batch.mag.shape}, {batch.mag[0,:]}")
        # print(f"batch.input_mask { torch.sum(batch.input_mask) }{batch.input_mask.shape}, {batch.input_mask[0,:]}")
        feat_list = [self.encode_time(batch.inter_times)]  # inter-event time from previous to current event
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        features = torch.cat(feat_list, dim=-1).contiguous() * batch.input_mask[:, :, None]
        # print(f"mag_mean {self.mag_mean}, tau_mean {self.tau_mean}")
        # print(f"rnn_in { torch.sum(features*batch.input_mask[:, :, None]) }{features.shape}{features[0,:]}")
        # torch.save(features, 'features2.pth')
        # torch.save(batch.arrival_times, 'arrival_times2.pth')
        rnn_output, _ = self._run_rnn(features)
        rnn_output = rnn_output * batch.input_mask[:, :, None]
        # print(f"rnn_out { torch.sum(rnn_output) }{rnn_output.shape}")
        rnn_output = rnn_output[:, :-1, :]  
        output = F.pad(rnn_output, (0, 0, 1, 0))  
        output = self.dropout(output)
        return output  

    def get_inter_time_dist(self, context):
        """Get the distribution over the inter-event times given the context."""
        return self.inter_time_decoder.from_context(context, self.hypernet_time)

    def forward(self, batch):
        feat_list = [self.encode_time(batch.inter_times)]  # inter-event time from previous to current event
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        features = torch.cat(feat_list, dim=-1).contiguous() * batch.input_mask[:, :, None]
        rnn_output, hidden = self._run_rnn(features)
        return rnn_output, hidden

    
    def get_magnitude_dist(self, context):
        log_rate = self.hypernet_mag(context).squeeze(-1)  # (B, L)
        b = self.richter_b * torch.ones_like(log_rate)
        mag_min = self.mag_completeness * torch.ones_like(log_rate)
        return dist.GutenbergRichter(b=b, mag_min=mag_min)

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 0.0,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.
            reduction: Reduction mode passed to ``reduce_nll_dict``.
            return_dict: If true, return a dict with ``time`` and ``total``.
                If false, return only the ``total`` component for compatibility.
            eps: Numerical epsilon used in hazard-related computations.

        Returns:
            Dict of reduced NLL components by default:
                ``{"time": ..., "total": ...}`` (and optional ``"bg"``).
            If ``return_dict=False``, returns only the reduced ``total`` tensor.
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
        
        reduction = self.reduction if reduction is None else reduction

        nll_time = -log_like  # (B,)
        nll_total = nll_time
        out = {
            "time": nll_time,
            "total": nll_total,
        }
        if getattr(self, "bg_model", None) is not None:
            log_h_intensity = inter_time_dist.log_hazard(batch.inter_times.clamp_min(eps))
            nll_change_sig = inspect.signature(self.bg_model.nll_change)
            supports_include_kl = "include_kl" in nll_change_sig.parameters
            if supports_include_kl:
                nll_bg_raw = self.bg_model.nll_change(batch, log_h_intensity, include_kl=False)  # (B,)
            else:
                nll_bg_raw = self.bg_model.nll_change(batch, log_h_intensity)  # (B,)
            bg_kl = self.bg_model.kl_term(batch, eps=eps) if hasattr(self.bg_model, "kl_term") else None
            nll_bg = nll_bg_raw
            if bg_kl is not None and not supports_include_kl:
                nll_bg = nll_bg_raw - bg_kl
            bg_norm = None
            if self.weights["bg_norm_weight"] > 0.0 and hasattr(self.bg_model, "normalizing_term"):
                bg_norm = self.bg_model.normalizing_term(
                    batch,
                    log_h_intensity,
                    eps=eps,
                )
            nll_total = nll_total + self.weights["bg_weight"] * nll_bg
            if bg_kl is not None:
                nll_total = nll_total + self.weights["bg_kl_weight"] * bg_kl
            if bg_norm is not None:
                nll_total = nll_total + self.weights["bg_norm_weight"] * bg_norm
            out["bg"] = nll_bg
            if bg_kl is not None:
                out["bg_kl"] = bg_kl
            if bg_norm is not None:
                out["bg_norm"] = bg_norm
            out["total"] = nll_total

        out = self.reduce_nll_dict(out, batch, reduction=reduction, eps=eps)
        if return_dict is False:
            return out["total"]
        return out


    def _sampling_state_step(
        self,
        rnn_input: torch.Tensor,
        current_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rnn_output, next_hidden = self.rnn(rnn_input, current_hidden)
        next_state = self.dropout(rnn_output[:, -1:, :].contiguous())
        return next_state, next_hidden

    def _get_context_and_hidden(
        self,
        batch: src.data.Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_list = [self.encode_time(batch.inter_times)]
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        if self.num_extra_features is not None and hasattr(batch, "extra_features"):
            feat_list.append(self.encode_extra_features(batch.extra_features))
        features = torch.cat(feat_list, dim=-1).contiguous() * batch.input_mask[:, :, None]
        rnn_output, hidden = self._run_rnn(features)
        rnn_output = rnn_output * batch.input_mask[:, :, None]
        rnn_output = rnn_output[:, :-1, :]
        context = F.pad(rnn_output, (0, 0, 1, 0))
        return self.dropout(context), hidden
