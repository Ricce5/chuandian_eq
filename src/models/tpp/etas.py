#  pytorch implementation of the ETAS model
# Modify the ETAS model to use the ETAS background model
# ref: https://zenodo.org/records/8161777 Using Deep Learning for Flexible and Scalable Earthquake Forecasting
import logging
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

logger = logging.getLogger(__name__)

def _to_tensor(x, ref: torch.Tensor):
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)

def _compute_branching_ratio(k=0.001, b=1, alpha=1, M_min=0, M_max=10):
    """Compute branching ratio of the ETAS model (Sornette & Werner)."""
    # Magnitude-term only (truncated Gutenberg-Richter in [M_min, M_max]).
    delta_m = M_max - M_min
    if delta_m <= 0:
        raise ValueError("M_max must be greater than M_min.")

    denom = 1 - 10 ** (-b * delta_m)
    if np.isclose(denom, 0.0):
        raise ValueError("Invalid magnitude range or b value for branching ratio.")

    if np.isclose(b, alpha):
        branching_ratio = (
            k * b * np.log(10) * delta_m / denom
        )
    else:
        branching_ratio = k * b / (b - alpha)
        branching_ratio *= (1 - 10 ** (-(b - alpha) * delta_m)) / denom
    if branching_ratio > 1:
        logger.warning("Branching ratio: %s", branching_ratio)
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
    if np.isclose(p, 1.0):
        return np.log(T2 + c) - np.log(T1 + c)
    else:
        return ((T2 + c) ** (1 - p) - (T1 + c) ** (1 - p)) / (1 - p)


def omori_inv(T1, T2, c, p, size=1, t_max=1e10):
    """Draw samples from Omori law on [T1, T2] using inverse transform.

    Sampling is conditional on the interval [T1, T2], implemented via a global CDF
    transform with truncation on [0, t_max].
    """
    if t_max <= 0:
        raise ValueError("t_max must be positive.")
    if c <= 0:
        raise ValueError("c must be positive.")

    T1 = np.clip(np.asarray(T1, dtype=np.float64), 0.0, t_max)
    T2 = np.clip(np.asarray(T2, dtype=np.float64), 0.0, t_max)
    if np.any(T2 < T1):
        raise ValueError("T2 must be >= T1 after clipping to [0, t_max].")

    u = np.random.random(size=size)
    total_mass = omori_int(0.0, t_max, c, p)
    cdf_T1 = omori_int(0.0, T1, c, p) / total_mass
    cdf_T2 = omori_int(0.0, T2, c, p) / total_mass
    u_prime = u * (cdf_T2 - cdf_T1) + cdf_T1

    if np.isclose(p, 1.0):
        return c * np.exp(u_prime * (np.log(t_max + c) - np.log(c))) - c

    one_minus_p = 1.0 - p
    base = u_prime * total_mass * one_minus_p + c ** one_minus_p
    base = np.maximum(base, np.finfo(np.float64).tiny)
    return base ** (1.0 / one_minus_p) - c


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
        fix_mu: bool = False,
        fixed_mu_value: Optional[float] = None,
        loss_reduction: str = "per_time",
    ):
        super().__init__()
        self.fix_mu = fix_mu
        base_rate_init_t = torch.as_tensor(
            base_rate_init, device=device, dtype=torch.get_default_dtype()
        )
        self.log_p = nn.Parameter(torch.tensor(math.log(omori_p_init)))
        self.log_c = nn.Parameter(torch.tensor(math.log(omori_c_init)))
        self.log_mu = nn.Parameter(base_rate_init_t.log())
        if self.fix_mu:
            self.log_mu.requires_grad = False
            mu_value = (
                torch.as_tensor(
                    fixed_mu_value, device=device, dtype=base_rate_init_t.dtype
                )
                if fixed_mu_value is not None
                else torch.zeros(1, device=device, dtype=base_rate_init_t.dtype)
            )
            self.register_buffer("mu_fixed", mu_value)
        self.log_k = nn.Parameter(torch.tensor(math.log(productivity_k_init)))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(productivity_alpha_init)))
        self.register_buffer("M_c", torch.tensor(mag_completeness))
        self.register_buffer("M_m", torch.tensor(mag_max))
        self.register_buffer("b", torch.tensor(richter_b))
        self.report_params = report_params
        self.device = device
        self.bg_model = bg_model
        self.reduction = loss_reduction
        self.to(device)

    @property
    def p(self):
        return torch.exp(self.log_p)

    @property
    def c(self):
        return torch.exp(self.log_c)

    @property
    def mu(self):
        if getattr(self, "fix_mu", False):
            return self.mu_fixed
        return torch.exp(self.log_mu)

    @property
    def k(self):
        return torch.exp(self.log_k)

    @property
    def alpha(self):
        return torch.exp(self.log_alpha)

    @property
    def branching_ratio(self) -> float:
        """Magnitude-only branching term under truncated Gutenberg-Richter."""
        return float(
            _compute_branching_ratio(
                k=float(self.k.detach().cpu()),
                b=float(self.b.detach().cpu()),
                alpha=float(self.alpha.detach().cpu()),
                M_min=float(self.M_c.detach().cpu()),
                M_max=float(self.M_m.detach().cpu()),
            )
        )

    def effective_branching_ratio(self, t_max: float = 1e4) -> float:
        """Total offspring ratio including temporal Omori mass on [0, t_max]."""
        p = float(self.p.detach().cpu())
        c = float(self.c.detach().cpu())
        temporal_mass = float(omori_int(0.0, float(t_max), c, p))
        return self.branching_ratio * temporal_mass

   

    def nll_loss(
        self,
        batch: Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 1e-8,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        reduction = self.reduction if reduction is None else reduction
        t = batch.arrival_times                         # (B,L)
        # Mask of real (non-padding) events             
        survival_mask = get_mask(                       # (B,L)
            batch.inter_times,      
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx,
        )
        # t_select - arrival times of events for which intensity must be computed, shape (B, S)
        # (where S = L if t_start == t_nll_start, and S <= L otherwise)
        t_select, intensity_mask = masked_select_per_row(t, batch.nll_event_mask)
        # delta_t[0, i, j] = t_i - t_j
        delta_t = t_select.unsqueeze(-1) - t.unsqueeze(-2)       # (B, S, L)
        # prev_mask[0, i, j] = float(t_i < t_j)
        prev_mask = (delta_t > 0).float() 
        # Logarithm of the intensity
        # omori[0, i, j] = contribution of event t_j on intensity at time t_i
        omori = (delta_t * prev_mask + self.c).pow(-self.p)      # (B, S, L)
        # productivity[0, j] = expected number of aftershocks after event t_j
        masked_mag = (batch.mag - self.M_c) * survival_mask      # (B, L)
        productivity = self.k * 10 ** (self.alpha * masked_mag)  # (B, L)
        #
        intensity = (
            omori * productivity.unsqueeze(-2) * prev_mask * survival_mask.unsqueeze(-2)
        ).sum(-1) + self.mu  # (B, S)
        if self.bg_model is not None:
            f_intensity = self.bg_model.intensity(batch,t_query=t_select) # (B, S)
            if intensity.numel() > 0:
                logger.debug(
                    "intensity max: %s, f_intensity max: %s, mu max: %s",
                    intensity.max().item(),
                    f_intensity.max().item(),
                    self.mu.max().item(),
                )
            else:
                logger.debug("Empty event-selection window in nll_loss; skipping max() stats.")
            intensity += f_intensity
        
        # Numerical guard: zero/negative intensity leads to -inf log-likelihood and NaN gradients.
        intensity_safe = intensity.clamp_min(eps)
        log_intensity = (torch.log(intensity_safe) * intensity_mask).sum(-1)
        # Integrated intensity
        one_minus_p = 1 - self.p
        t_end = batch.t_end.unsqueeze(-1)  # (B, 1)
        t_nll_start = batch.t_nll_start.unsqueeze(-1)  # (B, 1)
        # omori_int[0, j] = integral of the omori law from max(t_j, t_nll_start) to t_end
        dt_end = (t_end - t).clamp_min(0.0)
        dt_start = (t_nll_start - t).clamp_min(0.0)
        omori_int = ((dt_end + self.c).pow(one_minus_p) - (dt_start + self.c).pow(one_minus_p)) / one_minus_p  # (B, L)
        # zero-out padded events to avoid propagating NaNs from invalid exponents
        omori_int = omori_int * survival_mask
        integral = (omori_int * productivity * survival_mask).sum(-1)
        integral += (batch.t_end - batch.t_nll_start) * self.mu # (B,1)
        if self.bg_model is not None:
            f_integral = self.bg_model.intensity_integral(batch)  # (B,)
            integral += f_integral
        nll_time = -log_intensity + integral

        # Keep a single optimization key: total.
        bg_kl = None
        if self.bg_model is not None and hasattr(self.bg_model, "kl_term"):
            bg_kl = self.bg_model.kl_term(batch, eps=eps)

        nll_total = nll_time if bg_kl is None else (nll_time + bg_kl)

        out_dict = {
            "time": nll_time,
            "total": nll_total,
        }
        if bg_kl is not None:
            out_dict["bg_kl"] = bg_kl

        out = self.reduce_nll_dict(
            out_dict,
            batch,
            reduction=reduction,
            eps=eps,
        )
        if return_dict:
            return out
        return out["total"]
    
    
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
        survival_mask = get_mask(
            batch.inter_times[:, :-1],
            start_idx=torch.zeros_like(batch.start_idx),
            end_idx=batch.end_idx - 1,
        )
        if t_query is None:
            t_query = t
        delta_t = t_query.unsqueeze(-1) - t.unsqueeze(-2)  # (B, S, L)
        prev_mask = (delta_t > 0).float()  # (B, S, L)
        omori = (delta_t * prev_mask + self.c).pow(-self.p)  # (B, S, L)
        masked_mag = (mag - self.M_c) * survival_mask
        productivity = self.k * 10 ** (self.alpha * masked_mag)  # (B, L)
        h_intensity = (omori * productivity.unsqueeze(-2) * prev_mask * survival_mask.unsqueeze(-2)).sum(-1) + self.mu  # (B, S)
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
                    on_epoch=True, 
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
        verbose: bool = False, 
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
            tau_current = 0.0 
            inter_times = []
            while True: 
                tau = np.random.exponential(1.0 / upper_bound) 
                tau_current += tau
                t_current = t_current + tau
                if t_current > t_end:
                    break

                lambda_current = get_intensity(t_current, arrival_times, magnitudes) 
                p_accept = lambda_current / upper_bound
                if verbose:
                    logger.info(
                        "Candidate event at %.3f, acceptance prob = %.2f",
                        t_current,
                        p_accept,
                    )
                if bernoulli(p_accept):
                    arrival_times = np.append(arrival_times, t_current)
                    magnitudes = np.append(
                        magnitudes,
                        gen_mag(
                            b=float(self.b),
                            M_min=float(self.M_c),
                            M_max=float(self.M_m),
                        ),
                    )
                    inter_times.append(tau_current)
                    tau_current = 0.0
                    if verbose:
                        logger.info("Accepted candidate event (mag = %.2f)", magnitudes[-1])
                # Update the upper bound for the next event
                upper_bound = get_intensity(t_current, arrival_times, magnitudes)
                if len(inter_times) > max_length:
                    logger.warning(
                        "Stopping generation since max_length exceeded (likely explosive process)."
                    )
                    return None

            # Use max to avoid numerical errors
            inter_times = np.append(inter_times, max(duration - np.sum(inter_times), 0)) 
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
                delayed(sample_single_seq)(t_start, seed) 
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

    def sample(  
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
        # Move scalar parameters to CPU before any NumPy math.
        p = float(self.p.detach().cpu())
        c = float(self.c.detach().cpu())
        mu = float(self.mu.detach().cpu())
        k = float(self.k.detach().cpu())
        alpha = float(self.alpha.detach().cpu())
        b = float(self.b.detach().cpu())
        M_c = float(self.M_c.detach().cpu())
        M_m = float(self.M_m.detach().cpu())
        if past_seq is not None:
            t_start = float(past_seq.t_end)
        else:
            t_start = t_start

        # Determine the effective branching ratio (and assert that it is smaller than one)
        # branch = self.effective_branching_ratio(t_max=t_max)
        # if branch > 1:
        #     raise ValueError(
        #         f"The process is explosive: branching ratio {branch:.2f} is > 1."
        #     )
        omori_norm = float(omori_int(0.0, t_max, c, p))

        def sample_single_seq(seed, bg_times: Optional[np.ndarray] = None):   
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
         
           
            #########
            Nback = poisson.rvs(mu * (duration)) if self.bg_model is None else 0

            # background events occur randomly in the time domain
            if self.bg_model is None:   
                background_events = [np.random.uniform(t_start, t_end, Nback).T]
                background_events.append(
                    gen_mag(shape=Nback, b=b, M_min=M_c, M_max=M_m)
                )
                background_catalog = np.column_stack(background_events) # (Nback, 2)
            else:
                # precomputed NHPP samples avoid rebuilding the CIF per sequence
                if bg_times is None:
                    times_list = self.bg_model.sample_nhpp_inverse(
                        B=1,
                        t0=torch.tensor([t_start], device=self.device),
                        dt=torch.tensor([duration], device=self.device),
                        sample_sequence=True,
                        mu=float(self.mu.item()), 
                    )
                    t_back = np.array(times_list[0], dtype=np.float64)
                else:
                    t_back = bg_times

                Nback = t_back.size
                if Nback > 0:
                    m_back = gen_mag(shape=Nback, b=b, M_min=M_c, M_max=M_m)
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
                prod = productivity(parent_catalog[:, -1], k_prime, alpha, M_c) 

                # Determine how many of these events will be within the forecast interval:

                # Explanation:
                # This deserves some explanation. Condering the kernel k10**(alpha(M-M_c))*(t+c)**-p
                # We can recast the above expression as: N * p(t) where N is the number of aftershocks and p(t)
                # is a PDF for the arrival time (which integrates to 1).
                # Exanding the above expression yields:
                # N                   *           p(t)
                # k'*10**(alpha(M-Mc)) * (t+c)**-p / int((t+c)**-p)
                # where k' = k*int((t+c)**-p)    
                TAU1 = np.clip(t_start - parent_catalog[:, 0], a_min=0.0, a_max=t_max)
                TAU2 = np.clip(t_end - parent_catalog[:, 0], a_min=0.0, a_max=t_max)
                prod_in_interval = prod * omori_int(TAU1, TAU2, c, p) / omori_norm

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

                    dti = omori_inv(itau1, itau2, c, p, size=iNaft, t_max=t_max) 
                    t_aftershock = t_parent + dti  # new arrival time

                    aftershock_catalog = []
                    aftershock_catalog.append(t_aftershock)

                    # ...and magnitudes
                    m_aftershock = gen_mag(iNaft, b=b, M_min=M_c, M_max=M_m)
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
                    logger.warning("Exceeded %s events, discarding sequence", max_length)
                    return None

            whole_catalog = np.vstack(whole_catalog)
            whole_catalog = whole_catalog[whole_catalog[:, 0].argsort()] 

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

            # pre-sample NHPP background events in one call to reuse cached grids
            bg_times_bulk: Optional[List[np.ndarray]] = None
            if self.bg_model is not None:
                times_list = self.bg_model.sample_nhpp_inverse(
                    B=num_seq_to_generate,
                    t0=torch.full((num_seq_to_generate,), t_start, device=self.device),
                    dt=torch.full((num_seq_to_generate,), duration, device=self.device),
                    sample_sequence=True,
                    mu=float(self.mu.item()),
                )
                def _to_numpy(x):
                    if torch.is_tensor(x):
                        return x.detach().cpu().numpy().astype(np.float64)
                    return np.asarray(x, dtype=np.float64)

                bg_times_bulk = [_to_numpy(t) for t in times_list]

            new_sequences = Parallel(n_jobs=n_jobs)(
                delayed(sample_single_seq)(seed, None if bg_times_bulk is None else bg_times_bulk[i])
                for i, seed in enumerate(trange(starting_seed, starting_seed + num_seq_to_generate))
            )
            # Filter out explosive sequences
            filtered = [seq for seq in new_sequences if seq is not None]
            sequences.extend(filtered)
            starting_seed += num_seq_to_generate 

        if return_sequences:
            return sequences
        else:
            return Batch.from_list(sequences)

    @staticmethod
    def _torch_gen_mag(shape, b, M_min, M_max, device, dtype):
        # Gutenberg-Richter inverse sampling (torch)
        u = torch.rand(shape, device=device, dtype=dtype)
        return (-1.0 / b) * torch.log10(
            -u * (10 ** (-b * M_min) - 10 ** (-b * M_max)) + 10 ** (-b * M_min)
        )

    @staticmethod
    def _torch_omori_int(T1, T2, c, p):
        # Integral of Omori kernel from T1 to T2
        # Handles p == 1
        one = torch.tensor(1.0, device=T1.device, dtype=T1.dtype)
        if torch.isclose(p, one):
            return torch.log(T2 + c) - torch.log(T1 + c)
        one_minus_p = one - p
        return ((T2 + c).pow(one_minus_p) - (T1 + c).pow(one_minus_p)) / one_minus_p

    @staticmethod
    def _torch_omori_inv(T1, T2, c, p, size, t_max, device, dtype):
        """
        Draw samples tau ~ Omori(tau+c)^(-p) restricted to [T1, T2]
        using inverse transform with global normalization integral(0,t_max).
        """
        u = torch.rand(size, device=device, dtype=dtype)

        zero = torch.zeros((), device=device, dtype=dtype)
        tmax_t = torch.tensor(t_max, device=device, dtype=dtype)
        T1 = torch.clamp(T1, min=0.0, max=float(t_max))
        T2 = torch.clamp(T2, min=0.0, max=float(t_max))
        if torch.any(T2 < T1):
            raise ValueError("T2 must be >= T1 after clipping to [0, t_max].")

        F0 = ETAS._torch_omori_int(zero, tmax_t, c, p)  # ∫_0^{t_max} g(t)

        # CDF in [0, t_max]
        F1 = ETAS._torch_omori_int(zero, T1, c, p) / F0
        F2 = ETAS._torch_omori_int(zero, T2, c, p) / F0
        u_prime = u * (F2 - F1) + F1

        one = torch.tensor(1.0, device=device, dtype=dtype)
        if torch.isclose(p, one):
            # tau = c * exp(u_prime * log((t_max+c)/c)) - c
            log_term = torch.log(tmax_t + c) - torch.log(c)
            return c * torch.exp(u_prime * log_term) - c

        one_minus_p = one - p
        # Inverse CDF:
        # u = ( (tau+c)^(1-p) - c^(1-p) ) / ( (tmax+c)^(1-p) - c^(1-p) )
        base = u_prime * F0 * one_minus_p + c.pow(one_minus_p)
        return base.pow(1.0 / one_minus_p) - c

    def sample_gpu_parallel(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional["Sequence"] = None,
        max_length: Optional[int] = 50_000,
        t_max: float = 1e10,
        return_sequences: bool = False,
        dtype: torch.dtype = torch.float64, 
    ) -> Union["Batch", List["Sequence"]]:
        """
        GPU-parallel branching-process sampler for ETAS.

        Fixes mismatch between offspring counts and Omori inverse sampling.
        Supports float32/float64 via dtype argument.
        """

        device = self.device
        if device is None:
            device = next(self.parameters()).device

        # ---- parameters on GPU ----
        p = self.p.to(device=device, dtype=dtype)
        c = self.c.to(device=device, dtype=dtype)
        mu = self.mu.to(device=device, dtype=dtype)
        k = self.k.to(device=device, dtype=dtype)
        alpha = self.alpha.to(device=device, dtype=dtype)
        b = self.b.to(device=device, dtype=dtype)

        M_c = self.M_c.to(device=device, dtype=dtype)
        M_m = self.M_m.to(device=device, dtype=dtype)

        # ---- effective branching ratio check (CPU float) ----
        branch = self.effective_branching_ratio(t_max=t_max)
        if branch > 1:
            raise ValueError(f"The process is explosive: branching ratio {branch:.2f} is > 1.")

        # ---- start/end per sequence ----
        if past_seq is not None:
            t0 = float(past_seq.t_end)
        else:
            t0 = float(t_start)

        t_start_vec = torch.full((batch_size,), t0, device=device, dtype=dtype)
        t_end_vec = t_start_vec + float(duration)

        catalogs = []

        # ---- conditioning history (replicate across batch) ----
        if past_seq is not None:
            past_tau = past_seq.inter_times.cpu().numpy().copy()
            tau_tensor = torch.tensor(past_tau[:-1], dtype=dtype, device=device)
            arrival_single = torch.cumsum(tau_tensor, dim=0) + float(past_seq.t_start)
            mags_single = torch.tensor(past_seq.mag, dtype=dtype, device=device)

            # replicate
            arrival = arrival_single.expand(batch_size, -1).reshape(-1)
            mags = mags_single.expand(batch_size, -1).reshape(-1)
            bid = torch.arange(batch_size, device=device, dtype=torch.int64).repeat_interleave(arrival_single.numel())

            catalogs.append((arrival, mags, bid))

        # ---- background events ----
        lam = mu * duration
        N_back = torch.poisson(lam.expand(batch_size)).to(torch.int64)
        max_back = int(N_back.max().item()) if N_back.numel() > 0 else 0

        if max_back > 0:
            u_time = torch.rand((batch_size, max_back), device=device, dtype=dtype)
            t_back = t_start_vec[:, None] + u_time * float(duration)  # (B, max_back)
            m_back = self._torch_gen_mag(
                (batch_size, max_back),
                b=float(b.item()),
                M_min=float(M_c.item()),
                M_max=float(M_m.item()),
                device=device,
                dtype=dtype,
            )
            mask_back = (torch.arange(max_back, device=device)[None, :] < N_back[:, None])

            t_back_flat = t_back[mask_back]
            m_back_flat = m_back[mask_back]
            bid_flat = torch.repeat_interleave(
                torch.arange(batch_size, device=device, dtype=torch.int64),
                N_back
            )
            catalogs.append((t_back_flat, m_back_flat, bid_flat))

        # ---- if no events at all ----
        if len(catalogs) == 0:
            empty = [
                Sequence(
                    inter_times=np.array([duration], dtype=np.float64),
                    t_start=float(t_start_vec[i].item()),
                    mag=np.array([], dtype=np.float64),
                )
                for i in range(batch_size)
            ]
            return empty if return_sequences else Batch.from_list(empty)

        # ---- merge catalog tensors ----
        t_parent = torch.cat([c_[0] for c_ in catalogs], dim=0)
        m_parent = torch.cat([c_[1] for c_ in catalogs], dim=0)
        b_parent = torch.cat([c_[2] for c_ in catalogs], dim=0)  # int64

        parent_catalog = (t_parent, m_parent, b_parent)
        events_all = [parent_catalog]
        total_events = t_parent.numel()

        # global normalization integral ∫_0^{t_max} g(t)
        zero = torch.zeros((), device=device, dtype=dtype)
        tmax_t = torch.tensor(t_max, device=device, dtype=dtype)
        omori_norm = self._torch_omori_int(zero, tmax_t, c, p)  # scalar

        # ---- branching process generations ----
        while True:
            t_parent, m_parent, b_parent = parent_catalog

            # interval boundaries relative to each parent
            TAU1 = (t_start_vec[b_parent] - t_parent).clamp(min=0.0, max=t_max)
            TAU2 = (t_end_vec[b_parent] - t_parent).clamp(min=0.0, max=t_max)

            # ----- offspring mean -----
            # Match CPU sampler:
            # k' = k * omori_norm
            # prod_in_interval = k' * 10^(alpha*(M-Mc)) * omori_int(TAU1,TAU2)/omori_norm
            # = k*10^(alpha*(M-Mc)) * omori_int(TAU1,TAU2)
            prod = k * omori_norm * (10.0 ** (alpha * (m_parent - M_c)))
            interval_mass = self._torch_omori_int(TAU1, TAU2, c, p)
            prod_in_interval = prod * interval_mass / omori_norm

            N_child = torch.poisson(prod_in_interval).to(torch.int64)
            has_child = N_child > 0
            if not has_child.any():
                break

            # explode parent indices
            parent_idx = torch.nonzero(has_child, as_tuple=False).squeeze(-1)
            repeat_idx = torch.repeat_interleave(parent_idx, N_child[has_child])
            n_child_total = repeat_idx.numel()

            # collect expanded parent info
            t_sel = t_parent[repeat_idx]
            bid_sel = b_parent[repeat_idx]
            tau1_sel = TAU1[repeat_idx]
            tau2_sel = TAU2[repeat_idx]

            # sample child times
            dti = self._torch_omori_inv(
                tau1_sel,
                tau2_sel,
                c,
                p,
                size=(n_child_total,),
                t_max=t_max,
                device=device,
                dtype=dtype,
            )
            t_child = t_sel + dti

            # sample child magnitudes
            m_child = self._torch_gen_mag(
                (n_child_total,),
                b=float(b.item()),
                M_min=float(M_c.item()),
                M_max=float(M_m.item()),
                device=device,
                dtype=dtype,
            )

            if (max_length is not None) and (total_events + n_child_total > max_length):
                # discard / truncate explosive
                break

            child_catalog = (t_child, m_child, bid_sel)
            events_all.append(child_catalog)
            parent_catalog = child_catalog
            total_events += n_child_total

        # ---- build sequences back from catalog ----
        all_t = torch.cat([ev[0] for ev in events_all], dim=0)
        all_m = torch.cat([ev[1] for ev in events_all], dim=0)
        all_b = torch.cat([ev[2] for ev in events_all], dim=0)  # int64

        seq_list: List[Sequence] = []
        for bidx in range(batch_size):
            mask = (all_b == bidx)
            if not mask.any():
                seq_list.append(Sequence(
                    inter_times=np.array([duration], dtype=np.float64),
                    t_start=float(t_start_vec[bidx].item()),
                    mag=np.array([], dtype=np.float64),
                ))
                continue

            times = all_t[mask]
            mags = all_m[mask]

            # restrict to forecast window
            in_win = (times > t_start_vec[bidx]) & (times <= t_end_vec[bidx])
            times = times[in_win]
            mags = mags[in_win]

            if times.numel() == 0:
                seq_list.append(Sequence(
                    inter_times=np.array([duration], dtype=np.float64),
                    t_start=float(t_start_vec[bidx].item()),
                    mag=np.array([], dtype=np.float64),
                ))
                continue

            sort_idx = torch.argsort(times)
            times = times[sort_idx]
            mags = mags[sort_idx]

            inter_times = torch.diff(torch.cat([
                t_start_vec[bidx:bidx+1],
                times,
                t_end_vec[bidx:bidx+1],
            ]))

            seq_list.append(Sequence(
                inter_times=inter_times.detach().cpu().numpy().astype(np.float64),
                t_start=float(t_start_vec[bidx].item()),
                mag=mags.detach().cpu().numpy().astype(np.float64),
            ))

        return seq_list if return_sequences else Batch.from_list(seq_list)



    
        
    def print_params(self):
        """Print current ETAS model parameters in a readable format."""
        params = {
            "p": self.p.detach().cpu().item(),
            "c": self.c.detach().cpu().item(),
            "mu": self.mu.detach().cpu().item(),
            "k": self.k.detach().cpu().item(),
            "K": (self.k / (self.p - 1) * self.c ** (1 - self.p)).detach().cpu().item(),
            "alpha": self.alpha.detach().cpu().item(),
            'alpha_e': (self.alpha * math.log(10)).detach().cpu().item(),
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
        # Gutenberg-Richter b
        b: Optional[float] = None,
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

        Notes:
            Do not mix code and paper parameterizations in one call.
            If only one paper parameter is provided (K or alpha_e), only its mapped
            code parameter is updated and the other remains unchanged.
        """
        use_code_param = (k is not None) or (alpha is not None)
        use_paper_param = (K is not None) or (alpha_e is not None)
        if use_code_param and use_paper_param:
            raise ValueError(
                "Ambiguous parameterization: provide either (k, alpha) or (K, alpha_e), not both."
            )

        with torch.no_grad():
            # 1) update p,c,mu first (because k conversion needs p,c)
            if p is not None:
                p_t = _to_tensor(p, self.log_p)
                self.log_p.copy_(torch.log(p_t))

            if c is not None:
                c_t = _to_tensor(c, self.log_c)
                self.log_c.copy_(torch.log(c_t))

            if mu is not None:
                if not self.fix_mu:
                    mu_t = _to_tensor(mu, self.log_mu)
                    self.log_mu.copy_(torch.log(mu_t))
                else:
                    mu_t = _to_tensor(mu, self.log_mu)
                    self.mu_fixed.copy_(mu_t)
            if b is not None:
                b_t = _to_tensor(b, self.b)
                self.b.copy_(b_t)

            # 2) if paper params provided, convert -> (k, alpha)
            if use_paper_param:
                # partial paper updates are allowed and applied independently
                if K is not None:
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
        selected_rows.append(matrix_row.masked_select(mask_row)) 

    new_matrix = pad_sequence(selected_rows)
    new_mask = pad_sequence([torch.ones_like(s) for s in selected_rows])
    return new_matrix, new_mask.float()



