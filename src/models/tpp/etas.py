#  pytorch implementation of the ETAS model
# ref: https://zenodo.org/records/8161777 Using Deep Learning for Flexible and Scalable Earthquake Forecasting
import math
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from scipy.stats import poisson
from tqdm.auto import trange

import src
from src.data.batch import get_mask, pad_sequence,Batch
from src.data.sequence import Sequence

from .tpp_model import TPPModel

def _to_tensor(x, ref: torch.Tensor):
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

def branching_ratio(k=0.001, b=1, alpha=1, M_min=0, M_max=10):
    """Compute branching ratio of the ETAS model (Sornette & Werner)."""  # 每个事件触发事件数的期望
    # n=EM​[k10α(M−Mc​)]⋅∫0∞​(t+c)−pdt
    if b == alpha:
        branching_ratio = (
            k * b * np.log(10) * (M_max - M_min) / (1 - 10 ** (-b * (M_max - M_min)))
        )
    else:
        branching_ratio = k * b / (b - alpha)
        branching_ratio *= 1 - 10 ** (
            -(b - alpha) * (M_max - M_min) / (1 - 10 ** (-b * (M_max - M_min)))
        )
    if branching_ratio > 1:
        print("Branching ratio: ", branching_ratio)
    return branching_ratio


def gen_mag(shape=1, b=1, M_min=0, M_max=10):
    """Draw sample from the Gutenberg-Richter distribution."""
    u = np.random.random(shape)
    mag = (
        -1
        / b
        * np.log10(-u * (10 ** (-b * M_min) - 10 ** (-b * M_max)) + 10 ** (-b * M_min))
    )
    return mag


def productivity(m, k, alpha=1, Mc=3):
    """Compute productivity of the earthquake with given magnitude."""
    return k * 10 ** (alpha * (m - Mc))    


def omori_int(T1, T2, c, p):
    """Integral of Omori's law from T1 to T2.

    Used for the finite catalog correction in the productivity estimate (Brodsky 2011).
    """
    if p == 1:
        return np.log(T2 + c) - np.log(T1 + c)
    else:
        return ((T2 + c) ** (1 - p) - (T1 + c) ** (1 - p)) / (1 - p)


def omori_inv(T1, T2, c, p, size=1, t_max=1e10):
    """Draw sample from Omori's law using inverse transform."""
    u = np.random.random(size=size)
    F = lambda tau: omori_int(0, tau, c, p) / omori_int(0, t_max, c, p)
    u_prime = u * (F(T2) - F(T1)) + F(T1)
    return (
        (u_prime * omori_int(0, t_max, c, p) + c ** (1 - p) / (1 - p)) * (1 - p)
    ) ** (1 / (1 - p)) - c


class ETAS(TPPModel):
    """Epidemic-type aftershock sequence model (Ogata, 1988).

    Args:
        omori_p_init: Initial value of the p paramater of Omori's law.
        omori_c_init: Initial value of the c paramater of Omori's law.
        base_rate_init: Initial value of the background (immigrant) intensity.
        productivity_k_init: Initial value of the productivty parameter k.
        productivity_alpha_init: Initial value of the productivty parameter alpha.
        richter_b: Fixed b value of the Gutenberg-Richter distribution for magnitudes.
        mag_completeness: Magnitude of completeness.
        report_params: Whether to report the model parameters in the PyTorch Lightning
            progress bar during training.
        learning_rate: Learning rate use for optimization.
    """

    def __init__(
        self,
        omori_p_init: float = 1.08,
        omori_c_init: float = 0.1,
        base_rate_init: float = 0.02,
        productivity_k_init: float = 0.0073,
        productivity_alpha_init: float = 1.0,
        richter_b: float = 1.0,
        mag_completeness: float = 2.0,
        mag_max: float=10.0,
        report_params: bool = True,
        device: Optional[torch.device] = None,
        bg_model=None,
        fix_mu_zero: bool = False,
    ):
        super().__init__()
        self.fix_mu_zero = fix_mu_zero
        self.log_p = nn.Parameter(torch.tensor(math.log(omori_p_init)))
        self.log_c = nn.Parameter(torch.tensor(math.log(omori_c_init)))
        self.log_mu = nn.Parameter(torch.tensor(math.log(base_rate_init)))
        if self.fix_mu_zero:
            self.log_mu.requires_grad = False
        self.log_k = nn.Parameter(torch.tensor(math.log(productivity_k_init)))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(productivity_alpha_init)))
        self.register_buffer("M_c", torch.tensor(mag_completeness))
        self.register_buffer("M_m", torch.tensor(mag_max))
        self.register_buffer("b", torch.tensor(richter_b))
        if self.fix_mu_zero:
            self.register_buffer("mu_zero", torch.tensor(0.0))
        self.report_params = report_params
        self.device = device
        self.bg_model = bg_model
        self.to(device)

    @property
    def p(self):
        return torch.exp(self.log_p)

    @property
    def c(self):
        return torch.exp(self.log_c)

    @property
    def mu(self):
        if getattr(self, "fix_mu_zero", False):
            return self.mu_zero
        return torch.exp(self.log_mu)

    @property
    def k(self):
        return torch.exp(self.log_k)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

   

    def nll_loss(self, batch: Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        t = batch.arrival_times
        # t_select - arrival times of events for which intensity must be computed, shape (B, S)
        # (where S = L if t_start == t_nll_start, and S <= L otherwise)
        t_select, intensity_mask = masked_select_per_row(t, batch.nll_event_mask)
        # delta_t[0, i, j] = t_i - t_j
        delta_t = t_select.unsqueeze(-1) - t.unsqueeze(-2)  # (B, S, L)
        # prev_mask[0, i, j] = float(t_i < t_j)
        prev_mask = (delta_t > 0).float()  # (B, S, L) 当前事件之前的所有事件掩码
        # Logarithm of the intensity
        # omori[0, i, j] = contribution of event t_j on intensity at time t_i
        omori = (delta_t * prev_mask + self.c).pow(-self.p)  # (B, S, L)
        # productivity[0, j] = expected number of aftershocks after event t_j
        productivity = self.k * 10 ** (self.alpha * (batch.mag - self.M_c))  # (B, L)
        #
        intensity = (omori * productivity.unsqueeze(-2) * prev_mask).sum(-1) + self.mu  # (B, S)
        if self.bg_model is not None:
            f_intensity = self.bg_model.intensity(batch,t_query=t_select) # (B, S)
            print(f"intensity max: {intensity.max().item()}, f_intensity max: {f_intensity.max().item()},mu max: {self.mu.max().item()}")
            intensity += f_intensity
        
        log_intensity = (
            torch.log(
                intensity
            )* intensity_mask
        ).sum(-1)
        ####### 对数条件强度函数和条件强度函数积分的掩码是分开计算的
        # Integrated intensity
        one_minus_p = 1 - self.p
        t_end = batch.t_end.unsqueeze(-1)  # (B, 1)
        t_nll_start = batch.t_nll_start.unsqueeze(-1)  # (B, 1)
        # omori_int[0, j] = integral of the omori law from max(t_j, t_nll_start) to t_end 对每个事件计算
        omori_int = (
            (t_end - t + self.c).pow(one_minus_p)
            - ((t_nll_start - t).clamp_min(0.0) + self.c).pow(one_minus_p)
        ) / one_minus_p  # (B, L)
        # 屏蔽padding事件对积分的贡献
        survival_mask = get_mask(
            batch.inter_times,
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx,
        )
        integral = (omori_int * productivity * survival_mask).sum(-1)
        integral += (batch.t_end - batch.t_nll_start) * self.mu # (B,1)
        if self.bg_model is not None:
            f_integral = self.bg_model.intensity_integral(batch)  # (B,)
            integral += f_integral
        nll_total = -log_intensity + integral
        return nll_total / (batch.t_end - batch.t_nll_start)  # (B,)
    
    def h_intensity(self, batch: Batch, t_query: torch.Tensor=None) -> torch.Tensor:
        """Compute the intensity at given query times for each sequence in the batch.

        Args:
            batch: Batch of event sequences.
            t_query: Query times, shape (B, S)

        Returns:
            intensity: Intensity at each query time, shape (B, S)
        """
        t = batch.arrival_times[:, :-1]
        mag = batch.mag[:, :-1]
        if t_query is None:
            t_query = t
        delta_t = t_query.unsqueeze(-1) - t.unsqueeze(-2)  # (B, S, L)
        prev_mask = (delta_t > 0).float()  # (B, S, L)
        omori = (delta_t * prev_mask + self.c).pow(-self.p)  # (B, S, L)
        productivity = self.k * 10 ** (self.alpha * (mag - self.M_c))  # (B, L)
        h_intensity = (omori * productivity.unsqueeze(-2) * prev_mask).sum(-1) + self.mu  # (B, S)
        return h_intensity



    def training_step(self, batch, batch_idx):
        loss = self.nll_loss(batch).mean()
        self.log(
            "train_loss",
            loss,
            on_step=False,                      
            on_epoch=True,
            batch_size=batch.batch_size,
        )
        if self.report_params:
            for param_name in ["p", "c", "mu", "k", "alpha"]:
                self.log(
                    f"params/{param_name}",
                    getattr(self, param_name).item(),
                    on_step=False,
                    on_epoch=True,                           # 在每个epoch结束时记录参数
                    prog_bar=True,
                    batch_size=batch.batch_size,
                )
        return loss

    def sample_thinning(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[Sequence] = None,
        random_state: int = 123,
        max_length: int = 50_000,
        n_jobs: int = -1,
        return_sequences: bool = False,
        verbose: bool = False,         # 是否打印采样过程的详细信息
    ) -> Union[Batch, List[Sequence]]:
        """Generate a sample from the model (conditional or unconditional).

        Uses the thinning algorithm, which can be much slower than the default branching
        process sampler `sample`, especially for long sequences.

        Args:
            batch_size: number of sequences to generate.
            duration: length of the interval on which to simulate the tpp.
            t_start: start of the interval on which to simulate the tpp.
            past_seq: if provided, events are sampled conditioned on the past sequence.
            random_state: random seed, specified for reproducibility.
            max_length: if not none, discards samples with more than this many events.
                prevents runaway explosive sequences.
            n_jobs: Number of jobs that run sampling in parallel. -1 uses all cores.
            return_sequences: if true, returns samples as list[Sequence].
                if false, returns samples as Batch.
        """
        p, c, mu, k, alpha = [
            param.cpu().detach().numpy()
            for param in [self.p, self.c, self.mu, self.k, self.alpha]
        ]

        def get_intensity(t: float, t_past: np.ndarray, mag_past: np.ndarray):
            """Compute the intesity at time t given past events and magnitudes."""
            omori = (t - t_past + c) ** (-p)
            productivity = k * 10 ** (alpha * (mag_past - float(self.M_c)))
            return (omori * productivity).sum() + mu

        def bernoulli(success_proba: float):
            if success_proba < 0 or success_proba > 1:
                raise ValueError("Success probability must be in [0, 1] range")
            return np.random.uniform() < success_proba

        # Need to pass t_start as an argument due to weird Python interpreter behavior
        def sample_single_seq(t_start, seed):
            np.random.seed(seed)
            if past_seq is not None:
                # Recompute the arrival times in float64 precision
                past_tau = past_seq.inter_times.cpu().numpy().copy()
                arrival_times = np.cumsum(past_tau[:-1]) + past_seq.t_start
                magnitudes = past_seq.mag.cpu().numpy().copy()
                t_start = float(past_seq.t_end)
            else:
                arrival_times = np.array([], dtype=np.float64)
                magnitudes = np.array([], dtype=np.float64)
                t_start = t_start
            t_current = t_start
            t_end = t_start + duration
            # upper bound on the intensity - used to generate candidate events
            upper_bound = get_intensity(t_current, arrival_times, magnitudes)
            tau_current = 0.0   # tau_current 为距离上一次事件的时间
            inter_times = []
            while True:  # 事件生成循环
                tau = np.random.exponential(1.0 / upper_bound) # 指数分布生成候选事件时间间隔
                tau_current += tau
                t_current = t_current + tau
                if t_current > t_end:
                    break

                lambda_current = get_intensity(t_current, arrival_times, magnitudes) # 当前时间的强度
                p_accept = lambda_current / upper_bound
                if verbose:
                    print(
                        f"\nCandidate event at {t_current:.3f}, acceptance prob = {p_accept:.2f}",
                        end="",
                    )
                if bernoulli(p_accept):
                    arrival_times = np.append(arrival_times, t_current)
                    magnitudes = np.append(
                        magnitudes,
                        gen_mag(b=float(self.b), M_min=float(self.M_c)),
                    )
                    inter_times.append(tau_current)
                    tau_current = 0.0
                    if verbose:
                        print(f" -> accepted (mag = {magnitudes[-1]:.2f})", end="")
                # Update the upper bound for the next event
                upper_bound = get_intensity(t_current, arrival_times, magnitudes)
                if len(inter_times) > max_length:
                    print(
                        "Stopping generation since max_length exceeded (likely explosive process)."
                    )
                    return None

            # Use max to avoid numerical errors
            inter_times = np.append(inter_times, max(duration - np.sum(inter_times), 0)) # 将剩余的时间间隔添加到inter_times中
            valid_idx = (arrival_times > t_start) & (arrival_times <= t_end)
            return dict(
                inter_times=inter_times,
                t_start=t_start,
                mag=magnitudes[valid_idx],
            )

        sequences = []
        # Keep generating sequences in groups of size (batch_size - len(sequences))
        # until batch_size is reached. Some sequences might be too long because of
        # explosiveness - these are filtered out
        while len(sequences) < batch_size:
            num_seq_to_generate = batch_size - len(sequences)
            # Random seed is passed as an agument to the sampling function to ensure reproducibility
            new_sequences = Parallel(n_jobs=n_jobs)(
                delayed(sample_single_seq)(t_start, seed)  # delayed为joblib的延迟执行函数
                for seed in trange(random_state, num_seq_to_generate + random_state)
            )
            filtered = [
                Sequence(**seq) for seq in new_sequences if seq is not None
            ]
            sequences.extend(filtered)

        if return_sequences:
            return sequences
        else:
            return Batch.from_list(sequences)

    def sample(   # 基于分支过程模拟
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[Sequence] = None,
        random_state: int = 123,
        max_length: Optional[int] = 50_000,
        t_max: float = 1e10,  # maximum duration of the aftershock sequence. Important for p close to 1.
        n_jobs: int = -1,     # Number of jobs that run sampling in parallel. -1 uses all cores.
        return_sequences: bool = False,
    ) -> Union[Batch, List[Sequence]]:
        """Generate a sample from the model (conditional or unconditional).

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the interval on which to simulate the TPP.
            t_start: Start of the interval on which to simulate the TPP.
            past_seq: If provided, events are sampled conditioned on the past sequence.
            random_state: Random seed, specified for reproducibility.
            max_length: If not None, discards samples with more than this many events.
                Prevents runaway explosive sequences.
            t_max: Maximum time since parent at which an aftershock can be produced.
            n_jobs: Number of jobs that run sampling in parallel. -1 uses all cores.
            return_sequences: If True, returns samples as List[Sequence].
                If False, returns samples as Batch.

        Returns:
            batch: Sequences generated from the model.
        """
        p, c, mu, k, alpha, b, M_c = [
            param.cpu().detach().numpy()
            for param in [self.p, self.c, self.mu, self.k, self.alpha, self.b, self.M_c]
        ]
        if past_seq is not None:
            t_start = float(past_seq.t_end)
        else:
            t_start = t_start

        # Determine the branching ratio (and assert that it is smaller than one)
        branch = branching_ratio(k=k, b=b, alpha=alpha, M_min=M_c, M_max=self.M_m)
        if branch > 1:
            raise ValueError(
                f"The process is explosive: branching ratio {branch:.2f} is > 1."
            )

        def sample_single_seq(seed):   
            np.random.seed(seed)
            if past_seq is not None:
                # Recompute the arrival times in float64 precision
                past_tau = past_seq.inter_times.cpu().numpy().copy()
                arrival_times = np.cumsum(past_tau[:-1]) + past_seq.t_start
                magnitudes = past_seq.mag.cpu().numpy().copy()
                parent_catalog = np.column_stack((arrival_times, magnitudes))
            else:
                arrival_times = np.array([], dtype=np.float64)
                magnitudes = np.array([], dtype=np.float64)
                parent_catalog = []

            t_end = t_start + duration
         
            # Background events are sampled from a poisson distribution with mean mu*T  由条件强度函数背景地震率部分生成的事件
            #########
            Nback = poisson.rvs(mu * (duration))  # number of background events

            # background events occur randomly in the time domain
            if self.bg_model is None:   
                background_events = [np.random.uniform(t_start, t_end, Nback).T]
                background_events.append(gen_mag(shape=Nback, b=b, M_min=M_c))
                background_catalog = np.column_stack(background_events) # (Nback, 2)
            else:
                times_list = self.bg_model.sample_nhpp_inverse(
                    B=1,
                    t0=torch.tensor([t_start], device=self.device),
                    dt=torch.tensor([duration], device=self.device),
                    sample_sequence=True,
                    mu=float(self.mu.item()), 
                )
                t_back = np.array(times_list[0], dtype=np.float64)

                Nback = t_back.size
                if Nback > 0:
                    m_back = gen_mag(shape=Nback, b=b, M_min=M_c)
                    background_catalog = np.column_stack([t_back, m_back]).astype(np.float64)
                else:
                    background_catalog = np.empty((0, 2), dtype=np.float64)
            #########
            parent_catalog = (
                np.vstack((parent_catalog, background_catalog))
                if len(parent_catalog) > 0
                else background_catalog
            )
            offspring_catalog = []
            whole_catalog = []
            generation = 0

            num_generated = Nback
            while True:
                whole_catalog.append(parent_catalog)

                # Determine the number of offspring each parent will have:
                k_prime = k * omori_int(0, t_max, c, p)  # k'
                prod = productivity(parent_catalog[:, -1], k_prime, alpha, M_c) # parent_catalog[:, -1] 是父事件的震级

                # Determine how many of these events will be within the forecast interval:

                # Explanation:
                # This deserves some explanation. Condering the kernel k10**(alpha(M-M_c))*(t+c)**-p
                # We can recast the above expression as: N * p(t) where N is the number of aftershocks and p(t)
                # is a PDF for the arrival time (which integrates to 1).
                # Exanding the above expression yields:
                # N                   *           p(t)
                # k'*10**(alpha(M-Mc)) * (t+c)**-p / int((t+c)**-p)
                # where k' = k*int((t+c)**-p)    
                TAU1 = t_start - parent_catalog[:, 0]   # 父事件发生时间到模拟开始时间的间隔
                TAU1[TAU1 < 0] = 0  # 父事件发生在模拟事件之前
                TAU2 = t_end - parent_catalog[:, 0] # 父事件发生时间到模拟结束时间的间隔
                prod_in_interval = (
                    prod * omori_int(TAU1, TAU2, c, p) / omori_int(0, t_max, c, p)
                )

                # to streamline things we only consider the events that do have aftershocks
                N_aftershock = np.random.poisson(prod_in_interval)
                I = N_aftershock != 0  # indices of parents that have aftershocks
                short_parent_catalog = parent_catalog[I, :]
                short_N_aftershock = N_aftershock[I]

                for ieq, iNaft, itau1, itau2 in zip(
                    short_parent_catalog, short_N_aftershock, TAU1[I], TAU2[I]
                ):

                    # The cdf follows of the time distributions follows the integral of
                    # omori's law normalized by the integral out to infinity. Provided the
                    # temporal decay (p-value) is greater than 1 (p>1), the integral converges
                    # the inverse cdf can be used to generate random times:

                    t_parent = ieq[0]  # parent event time

                    dti = omori_inv(itau1, itau2, c, p, size=iNaft, t_max=t_max) # 生成iNaft个余震的时间间隔
                    t_aftershock = t_parent + dti  # new arrival time

                    aftershock_catalog = []
                    aftershock_catalog.append(t_aftershock)

                    # ...and magnitudes
                    m_aftershock = gen_mag(iNaft, b=b, M_min=M_c)
                    aftershock_catalog.append(m_aftershock)

                    aftershock_catalog = np.column_stack(aftershock_catalog)

                    # Tack on the aftershock sequence to the catalog of offsprings
                    offspring_catalog.append(aftershock_catalog)

                if not offspring_catalog:
                    break

                # make offspring catalog
                offspring_catalog = np.vstack(offspring_catalog)
                generation += 1
                parent_catalog = offspring_catalog
                offspring_catalog = []
                # Stop generation if the event sequence is too long
                num_generated += len(parent_catalog)
                if max_length is not None and num_generated > max_length:
                    print(f"Exceeded {max_length} events, discarding sequence")
                    return None

            whole_catalog = np.vstack(whole_catalog)
            whole_catalog = whole_catalog[whole_catalog[:, 0].argsort()]  # 对事件的到达时间进行排序

            arrival_times = whole_catalog[:, 0]
            magnitudes = whole_catalog[:, -1]

            valid_idx = (arrival_times > t_start) & (arrival_times <= t_end)
            fc_arrival_times = arrival_times[valid_idx]
            fc_magnitudes = magnitudes[valid_idx]

            inter_times = np.diff(fc_arrival_times, prepend=t_start, append=t_end)
            return Sequence(
                inter_times=inter_times,
                t_start=t_start,
                mag=fc_magnitudes,
            )

        sequences = []
        # Keep generating sequences in groups of size (batch_size - len(sequences)) until batch_size is reached
        # Some sequences might be too long because of explosiveness - these are filtered out
        np.random.seed(random_state)
        starting_seed = np.random.randint(0, 100000)
        while len(sequences) < batch_size:
            num_seq_to_generate = batch_size - len(sequences)
            new_sequences = Parallel(n_jobs=n_jobs)(
                delayed(sample_single_seq)(seed)
                for seed in trange(starting_seed, starting_seed + num_seq_to_generate)
            )
            # Filter out explosive sequences
            filtered = [seq for seq in new_sequences if seq is not None]
            sequences.extend(filtered)
            starting_seed += num_seq_to_generate # 更新随机种子，确保每次生成的序列都是不同的

        if return_sequences:
            return sequences
        else:
            return Batch.from_list(sequences)
        
    def print_params(self):
        """Print current ETAS model parameters in a readable format."""
        params = {
            "p": self.p.detach().cpu().item(),
            "c": self.c.detach().cpu().item(),
            "mu": self.mu.detach().cpu().item(),
            "k": self.k.detach().cpu().item(),
            "K":self.k/(self.p-1)*self.c**(1-self.p).detach().cpu().item(),
            "alpha": self.alpha.detach().cpu().item(),
            'alpha_e':(self.alpha * np.log(10)).detach().cpu().item(),
            "b": float(self.b.detach().cpu().item()),
            "M_c": float(self.M_c.detach().cpu().item()),
            "M_m": float(self.M_m.detach().cpu().item()),
        }
        print("ETAS model parameters:")
        for name, value in params.items():
            print(f"  {name} = {value}")

    def set_params(
        self,
        *,
        p: Optional[float] = None,
        c: Optional[float] = None,
        mu: Optional[float] = None,
        # code parametrization
        k: Optional[float] = None,
        alpha: Optional[float] = None,
        # paper parametrization
        K: Optional[float] = None,
        alpha_e: Optional[float] = None,
    ) -> None:
        """
        Set ETAS parameters.

        - current implementation
            g = k * 10^(alpha*(M-Mc)) * (t+c)^(-p)

        - Ogata/Zhuang normalized implementation
            g = K * exp(alpha_e*(M-Mc)) * (p-1)*c^(p-1) * (t+c)^(-p)

        Conversion (paper -> code):
            alpha = alpha_e / ln(10)
            k     = K * (p-1) * c^(p-1)   (depends on p,c)
        """
        with torch.no_grad():
            # 1) update p,c,mu first (because k conversion needs p,c)
            if p is not None:
                p_t = _to_tensor(p, self.log_p)
                self.log_p.copy_(torch.log(p_t))

            if c is not None:
                c_t = _to_tensor(c, self.log_c)
                self.log_c.copy_(torch.log(c_t))

            if mu is not None and not self.fix_mu_zero:
                mu_t = _to_tensor(mu, self.log_mu)
                self.log_mu.copy_(torch.log(mu_t))

            # 2) if paper params provided, convert -> (k, alpha)
            if (K is not None) or (alpha_e is not None):
                # if only one of them is provided, use current value for the other
                if K is None:
                    K_t = torch.exp(self.log_k)  # placeholder; will be overwritten below only if needed
                else:
                    K_t = _to_tensor(K, self.log_k)

                if alpha_e is None:
                    alpha_e_t = None
                else:
                    alpha_e_t = _to_tensor(alpha_e, self.log_alpha)

                # alpha conversion: alpha = alpha_e / ln(10)
                if alpha_e_t is not None:
                    alpha_t = alpha_e_t / math.log(10.0)
                    self.log_alpha.copy_(torch.log(alpha_t))

                # k conversion needs p,c
                if K is not None:
                    p_cur = torch.exp(self.log_p)   # after possible update above
                    c_cur = torch.exp(self.log_c)
                    k_t = K_t * (p_cur - 1.0) * c_cur.pow(p_cur - 1.0)
                    self.log_k.copy_(torch.log(k_t))

                # If paper params were used, we ignore direct k/alpha below to avoid conflict
                return

            # 3) otherwise: original code params (k, alpha)
            if k is not None:
                k_t = _to_tensor(k, self.log_k)
                self.log_k.copy_(torch.log(k_t))

            if alpha is not None:
                a_t = _to_tensor(alpha, self.log_alpha)
                self.log_alpha.copy_(torch.log(a_t))



def masked_select_per_row(matrix, mask):
    """Perform masked select on each row, and return the result as a padded tensor.

    Args:
        matrix: 2-d tensor from which values must be selected, shape [M, N]
        mask: Boolean matrix indicating what entries must be selected, shape [M, N]

    Returns:
        new_matrix: 2-d tensor, where each row contains the selected entries from the
            respective row of matrix + padding.
        new_mask: Float mask indicating what entries correspond to actual values
            (new_mask[i, j] = 1 => new_matrix[i, j] is not padding).

    Example:
        >>> matrix = torch.tensor([
                [0, 1, 2, 3, 4],
                [5, 6, 7, 8, 9],
            ])
        >>> mask = torch.tensor([
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1],
            ])
        >>> selected, new_mask = masked_select_per_row(matrix, mask)
        >>> print(selected)
        tensor([[1, 2, 3],
                [8, 9, 0]])
        >>> print(new_mask)
        tensor([[1., 1., 1.],
                [1., 1., 0.]])
    """
    assert matrix.shape == mask.shape and matrix.ndim == 2
    selected_rows = []
    for matrix_row, mask_row in zip(matrix, mask.bool()):
        selected_rows.append(matrix_row.masked_select(mask_row)) # masked_select:torch对象的方法，根据mask选择

    new_matrix = pad_sequence(selected_rows)
    new_mask = pad_sequence([torch.ones_like(s) for s in selected_rows])
    return new_matrix, new_mask.float()


# αe​=α10​ln10
#