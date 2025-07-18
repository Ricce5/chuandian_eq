from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import src
import src.distributions as dist

from .tpp_model import TPPModel


class RecurrentTPP(TPPModel):
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
        self.register_buffer("tau_mean", torch.tensor(args.tau_mean, dtype=torch.float32))  # 平均事件间隔
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
            + int(self.input_magnitude)  # magnitude features 取true或false
            + 0 if self.num_extra_features is None else self.num_extra_features
        )
        self.rnn = getattr(nn, args.rnn_type)(
            self.num_rnn_inputs, self.context_size, batch_first=True
        )
        # from src.utils.utils import init_rnn_weights 
        # init_rnn_weights(self.rnn,seed=42) 
        # from src.utils.utils import print_weight_sum
        # print_weight_sum(self.rnn, name="RNN weights", verbose=True)
        self.dropout = nn.Dropout(args.rnn_dropout)
        self.to(self.device)

    def encode_time(self, inter_times):  # 做log变换并中心化
        log_tau = torch.log(torch.clamp_min(inter_times, 1e-10)).unsqueeze(-1)
        return log_tau - self.log_tau_mean

    def encode_magnitude(self, mag): # 中心化
        return mag.unsqueeze(-1) - self.mag_mean

    def encode_extra_features(self, extra_feat):
        return extra_feat

    def get_context(self, batch):
        """Get context embedding for each event in the batch of padded sequences.

        Returns:
            context: Context vectors, shape (batch_size, seq_len, context_size)
        """
        print(f"batch.inter_times { torch.sum(batch.inter_times*batch.input_mask[:, :, None]) }{batch.inter_times.shape}, {batch.inter_times[0,:]}")
        print(f"batch.mag { torch.sum(batch.mag[:,:-1]) }{torch.sum(batch.mag*batch.input_mask)}{batch.mag.shape}, {batch.mag[0,:]}")
        print(f"batch.input_mask { torch.sum(batch.input_mask) }{batch.input_mask.shape}, {batch.input_mask[0,:]}")
        feat_list = [self.encode_time(batch.inter_times)]  # 上一次事件到当前事件的时间间隔
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        features = torch.cat(feat_list, dim=-1).contiguous() * batch.input_mask[:, :, None]
        print(f"mag_mean {self.mag_mean}, tau_mean {self.tau_mean}")
        print(f"rnn_in { torch.sum(features*batch.input_mask[:, :, None]) }{features.shape}{features[0,:]}")
        torch.save(features, 'features2.pth')
        rnn_output = self.rnn(features.contiguous())[0]*batch.input_mask[:, :, None]
        print(f"rnn_out { torch.sum(rnn_output) }{rnn_output.shape}")
        rnn_output = rnn_output[:, :-1, :] # (B, L-1, C) 第i个时间点预测i+1个时间点的时间间隔  [0]对应所有时间步状态  对应上一次事件时context
        output = F.pad(rnn_output, (0, 0, 1, 0))  # (B, L, C) 在序列前面pad一位
        output = self.dropout(output)
        return output  # (B, L, C)  在RNN外dropout

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

    def forward(self, batch):
        feat_list = [self.encode_time(batch.inter_times)]  # 上一次事件到当前事件的时间间隔
        if self.input_magnitude:
            feat_list.append(self.encode_magnitude(batch.mag))
        features = torch.cat(feat_list, dim=-1).contiguous() * batch.input_mask[:, :, None]
        rnn_output = self.rnn(features.contiguous())
        return  rnn_output

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
        if self.input_magnitude != self.predict_magnitude:
            raise ValueError("Sampling is impossible if input_magnitude != predict_magnitude")
        if self.num_extra_features is not None:
            raise ValueError("Sampling is not currently supported for extra features")

        # 初始化状态
        if past_seq is not None:
            t_start = past_seq.t_end
            past_batch = src.data.Batch.from_list([past_seq])
            current_state = self.get_context(past_batch)[:, [-1], :]  # (1, 1, C)
            current_state = current_state.expand(batch_size, -1, -1)  # (B, 1, C)
            time_remaining = past_seq.t_end - past_seq.arrival_times[-1]
        else:
            current_state = torch.zeros(batch_size, 1, self.context_size, device=self.device)
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
            inter_time_list.append(next_inter_times)  

            rnn_input_list = [self.encode_time(next_inter_times)]

            if self.predict_magnitude:
                mag_dist = self.get_magnitude_dist(current_state)
                next_mag = mag_dist.sample()  # (B, 1)
                mag_list.append(next_mag)
                rnn_input_list.append(self.encode_magnitude(next_mag))

            rnn_input = torch.cat(rnn_input_list, dim=-1).contiguous()

            # RNN 更新状态
            current_state = self.rnn(rnn_input, current_state.transpose(0, 1).contiguous())[0]
            current_state = self.dropout(current_state)
            current_state = current_state.detach()  # 关键：防止图增长
            # print(f"current_state: {torch.sum(current_state)}")

            # 检查是否达到采样终点
            total_time = torch.cat(inter_time_list, dim=1).sum(-1).min()
            generated = total_time >= (t_end - t_start)

        # 合并列表成张量
        inter_times = torch.cat(inter_time_list, dim=1)  # (B, L)
        if self.predict_magnitude:
            magnitudes = torch.cat(mag_list, dim=1)  # (B, L)
        else:
            magnitudes = None

        # 时间修正与padding处理
        duration = t_end - t_start
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
