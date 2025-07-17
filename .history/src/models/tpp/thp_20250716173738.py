
import torch
import torch.nn as nn
import torch.nn.functional as F
import src.distributions as dist
from torch.distributions import Categorical
from .tpp_base import TPPModel  
import src.data
from typing import Optional, Dict, Any, Union, List, Tuple

class THP(TPPModel):
    def __init__(self, args,base_model,head, device):
        super().__init__(args, device)

        super().__init__()
        self.input_magnitude = True
        self.predict_magnitude = True
        self.num_extra_features = 0
        self.num_components = args.num_components
        self.register_buffer("tau_mean", torch.tensor(1.0, dtype=torch.float32))  # 平均事件间隔
        self.register_buffer("log_tau_mean", self.tau_mean.log())
        self.register_buffer("mag_mean", torch.tensor(args.mag_mean, dtype=torch.float32))
        self.register_buffer("time_max", torch.tensor(args.time_max, dtype=torch.float32))
        self.register_buffer("richter_b", torch.tensor(args.richter_b, dtype=torch.float32))
        self.register_buffer(
            "mag_completeness", torch.tensor(args.mag_completeness, dtype=torch.float32)
        )

    
        self.base_model = base_model
        self.num_time_params = 3 * self.num_components
        self.hypernet_time = nn.Linear(args.d_model, self.num_time_params)

        # RNN input features
        if self.input_magnitude:
            # Decoder for magnitude
            self.num_mag_params = 1  # (1 rate)
            self.hypernet_mag = nn.Linear(args.d_model, self.num_mag_params)

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
        params = self.hypernet_time(context)
        # Very small params may lead to numerical problems, clamp to avoid this
        # params = clamp_preserve_gradients(params, -6.0, np.inf)
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
    # log_rate没有被使用
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
        self.transformer.eval()  
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
        inter_time_list = []  # 用列表累积，避免频繁 cat
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
            inter_time_list.append(next_inter_times)  # 不再循环中 cat
            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample()  # (B, 1)
                mag_list.append(next_mag)
    
            buffer_batch.update_sample_batch(next_inter_times=next_inter_times, next_mag=next_mag if self.predict_magnitude else None)
            sample_batch = buffer_batch.get_sample_batch()  


            current_state = self.get_current_state(sample_batch)
            current_state = current_state.detach()  # 关键：防止图增长

   
            total_time = torch.cat(inter_time_list, dim=1).sum(-1).min()
            generated = total_time >= (t_end - t_start)

     
        inter_times = torch.cat(inter_time_list, dim=1)  # (B, L)
        if self.predict_magnitude:
            magnitudes = torch.cat(mag_list, dim=1)  # (B, L)
        else:
            magnitudes = None


        unclipped_arrival_times = inter_times.cumsum(-1)  # (B, L)
        epsilon = 1e-5
        padding_mask = unclipped_arrival_times > duration-epsilon
        inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
        end_idx = (1 - padding_mask.long()).sum(-1)
        last_surv_time = duration - inter_times.sum(-1)
        if (last_surv_time < 0).any():
            print("Min last_surv_time:", last_surv_time.min().item())
            print("Any negative?", (last_surv_time < 0).any().item())
            raise ValueError("last_surv_time < 0 detected")

        inter_times[torch.arange(batch_size), end_idx] = last_surv_time

        batch = src.data.Batch(
            inter_times=inter_times,
            arrival_times=inter_times.cumsum(-1),
            t_start=torch.full([batch_size], t_start, device=self.device).float(),
            t_end=torch.full([batch_size], t_end, device=self.device).float(),
            t_nll_start=torch.full([batch_size], t_start, device=self.device).float(),
            mask=padding_mask.float(),
            start_idx=torch.zeros(batch_size, device=self.device).long(),
            end_idx=end_idx,
            mag=magnitudes,
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
        self.transformer.eval()
        if self.input_magnitude != self.predict_magnitude:
            raise ValueError("Sampling is impossible if input_magnitude != predict_magnitude")
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")

        cache = self.transformer.init_cache()
        if past_seq is not None:
            t_start = past_seq.t_end
            time_remaining =  t_start - past_seq.arrival_times[-1]
            buffer_batch = src.data.Batch.init_sample_batch(past_seq=past_seq, batch_size=batch_size, max_sample_len=1000)
            sample_batch = buffer_batch.get_sample_batch()
            current_state, cache = self.get_current_state(sample_batch, cache)
            # print("current_state:", torch.sum(current_state),current_state.shape)

        else:
            current_state = torch.zeros(batch_size, 1, self.args.d_model, device=self.device)
            raise ValueError("past_seq must be provided for sampling")
            time_remaining = None

        t_end = t_start + duration
        inter_time_list = []  # 用列表累积，避免频繁 cat
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
            inter_time_list.append(next_inter_times)  # 不再循环中 cat

            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample()  # (B, 1)
                mag_list.append(next_mag)
    
            buffer_batch.update_sample_batch(next_inter_times=next_inter_times, next_mag=next_mag if self.predict_magnitude else None)
            tmp_batch = buffer_batch.get_tmp_batch()  
            current_state,cache = self.get_current_state(tmp_batch, cache)
            current_state = current_state.detach()  # 关键：防止图增长
            cache =cache.detach()

            total_time = torch.cat(inter_time_list, dim=1).sum(-1).min()
            generated = total_time >= (t_end - t_start)

     
        inter_times = torch.cat(inter_time_list, dim=1)  # (B, L)
        if self.predict_magnitude:
            magnitudes = torch.cat(mag_list, dim=1)  # (B, L)
        else:
            magnitudes = None

        unclipped_arrival_times = inter_times.cumsum(-1)  # (B, L)
        epsilon = 1e-5
        padding_mask = unclipped_arrival_times > duration-epsilon
        inter_times = torch.masked_fill(inter_times, padding_mask, 0.0)
        end_idx = (1 - padding_mask.long()).sum(-1)
        last_surv_time = duration - inter_times.sum(-1)
        if (last_surv_time < 0).any():
            print("Min last_surv_time:", last_surv_time.min().item())
            print("Any negative?", (last_surv_time < 0).any().item())
            raise ValueError("last_surv_time < 0 detected")

        inter_times[torch.arange(batch_size), end_idx] = last_surv_time

        batch = src.data.Batch(
            inter_times=inter_times,
            arrival_times=inter_times.cumsum(-1),
            t_start=torch.full([batch_size], t_start, device=self.device).float(),
            t_end=torch.full([batch_size], t_end, device=self.device).float(),
            t_nll_start=torch.full([batch_size], t_start, device=self.device).float(),
            mask=padding_mask.float(),
            start_idx=torch.zeros(batch_size, device=self.device).long(),
            end_idx=end_idx,
            mag=magnitudes,
        )

        if return_sequences:
            return batch.to_list()
        else:
            return batch
        
    # 时间变换定理，任何TPP可转化为单位泊松过程
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



        




 


