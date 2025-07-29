
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.distributions as dist
from src.models.layers import ScaledSoftplus
from .tpp_cif_model import TppCifModel
import src.data
from typing import Optional, Dict, Any, Union, List, Tuple

class THP_Cif(TppCifModel):
    def __init__(self, args,base_model):
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


 


