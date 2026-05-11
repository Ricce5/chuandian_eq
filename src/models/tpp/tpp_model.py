# ref: https://zenodo.org/records/8161777 Using Deep Learning for Flexible and Scalable Earthquake Forecasting
from typing import Callable, Optional, Tuple

import torch

import src.data
import src.distributions as dist

class TPPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def reduce_nll(
        self,
        values: torch.Tensor,
        batch: src.data.Batch,
        *,
        reduction: str,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Reduce per-sequence NLL values with a shared convention."""
        if reduction == "sum":
            return values.sum()
        if reduction == "mean":
            return values.mean()
        if reduction == "per_event":
            num_events = batch.nll_event_mask.sum(-1)
            return (values / num_events.clamp_min(1)).to(values.dtype)
        if reduction == "per_time":
            span = batch.t_end - batch.t_nll_start
            return (values / span.clamp_min(eps)).to(values.dtype)
        if reduction in {"none", None}:
            return values
        raise ValueError(f"Unknown reduction mode: {reduction}")

    def reduce_nll_dict(
        self,
        values: dict[str, torch.Tensor],
        batch: src.data.Batch,
        *,
        reduction: str,
        eps: float = 1e-10,
    ) -> dict[str, torch.Tensor]:
        """Apply the same reduction rule to each per-sequence NLL component."""
        return {
            name: self.reduce_nll(value, batch, reduction=reduction, eps=eps)
            for name, value in values.items()
        }

    def time_log_likelihood(
        self,
        *,
        batch: src.data.Batch,
        inter_time_dist: dist.MixtureSameFamily,
        state: torch.Tensor,
        dist_from_state: Callable[[torch.Tensor], dist.MixtureSameFamily],
        pdf_inter_times: torch.Tensor,
        survival_inter_times: torch.Tensor,
    ) -> torch.Tensor:
        """Shared log-likelihood for trigger-time component of TPP models."""
        log_pdf = inter_time_dist.log_prob(pdf_inter_times.clamp_min(1e-10))
        log_like = (log_pdf * batch.nll_event_mask).sum(-1)

        arange = torch.arange(batch.batch_size, device=state.device)
        last_surv_context = state[arange, batch.end_idx, :]
        last_surv_dist = dist_from_state(last_surv_context)
        last_log_surv = last_surv_dist.log_survival(survival_inter_times[arange, batch.end_idx])
        log_like = log_like + last_log_surv.squeeze(-1)

        if torch.any(batch.t_nll_start != batch.t_start):
            prev_surv_context = state[arange, batch.start_idx, :]
            prev_surv_dist = dist_from_state(prev_surv_context)
            prev_surv_time = survival_inter_times[arange, batch.start_idx] - (
                batch.arrival_times[arange, batch.start_idx] - batch.t_nll_start
            )
            prev_log_surv = prev_surv_dist.log_survival(prev_surv_time)
            log_like = log_like - prev_log_surv

        return log_like

    def nll_loss(self, batch: src.data.Batch) -> torch.Tensor:
        """
        Compute negative log-likelihood (NLL) for a batch of event sequences.

        Args:
            batch: Batch of padded event sequences.

        Returns:
            nll: NLL of each sequence, shape (batch_size,)
        """
        raise NotImplementedError

    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
    ) -> src.data.Batch:
        """
        Sample a batch of sequences from the TPP model.

        Args:
            batch_size: Number of sequences to generate.
            duration: Length of the time interval on which the sequence is simulated.

        Returns:
            batch: Batch of padded event sequences.
        """
        raise NotImplementedError

    def evaluate_intensity(
        self, sequence: src.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the intensity for the given sequence (used for plotting).

        Args:
            sequence: Sequence for which to evaluate the intensity.
            num_grid_points: Number of points between consecutive events on which to
                evaluate the intensity.

        Returns:
            grid: Times for which the intensity is evaluated,
                shape (seq_len * num_grid_points,)
            intensity: Values of the conditional intensity on times in grid,
                shape (seq_len * num_grid_points,)
        """
        raise NotImplementedError

    def evaluate_compensator(
        self, sequence: src.data.Sequence, num_grid_points: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the compensator for the given sequence (used for plotting).

        Args:
            sequence: Sequence for which to evaluate the compensator.
            num_grid_points: Number of points between consecutive events on which to
                evaluate the compensator.

        Returns:
            grid: Times for which the intensity is evaluated,
                shape (seq_len * num_grid_points,)
            intensity: Values of the conditional intensity on times in grid,
                shape (seq_len * num_grid_points,)
        """
        raise NotImplementedError
    

  
    
