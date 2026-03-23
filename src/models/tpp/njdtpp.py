from __future__ import annotations

import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import src.data
import src.distributions as dist

from .tpp_model import TPPModel


class MLP(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int = 32,
        num_hidden: int = 2,
        sigma: float = 0.01,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if num_hidden == 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_out)])
        elif num_hidden > 0:
            self.linears = nn.ModuleList([nn.Linear(dim_in, dim_hidden)])
            self.linears.extend([nn.Linear(dim_hidden, dim_hidden) for _ in range(num_hidden - 1)])
            self.linears.append(nn.Linear(dim_hidden, dim_out))
        else:
            raise ValueError("num_hidden must be >= 0")

        for layer in self.linears:
            nn.init.normal_(layer.weight, mean=0.0, std=sigma)
            nn.init.uniform_(layer.bias, a=-sigma, b=sigma)

        self.activation = copy.deepcopy(activation) if activation is not None else nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.linears[:-1]:
            x = self.activation(layer(x))
        return self.linears[-1](x)


class NJDSDE(nn.Module):
    def __init__(
        self,
        dim_eta: int,
        dim_hidden: int = 32,
        num_hidden: int = 2,
        sigma: float = 0.01,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.dim_eta = dim_eta
        self.F = MLP(dim_eta, dim_eta, dim_hidden, num_hidden, sigma, activation)
        self.G = MLP(dim_eta, dim_eta, dim_hidden, num_hidden, sigma, activation)
        self.H = MLP(dim_eta, dim_eta * dim_eta, dim_hidden, num_hidden, sigma, activation)

    def f(self, y: torch.Tensor) -> torch.Tensor:
        return self.F(y)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return self.G(y)

    def h(self, y: torch.Tensor) -> torch.Tensor:
        return self.H(y).view(y.shape[0], self.dim_eta, self.dim_eta)


class NJDTPP(TPPModel):
    """
    NJDTPP-style model for continuous magnitude marks.

    The event-time intensity is modeled through a latent multi-dimensional
    intensity state, while the observed magnitude is treated as a continuous
    mark. Magnitudes affect the post-event jump through a continuous encoder and
    contribute to the likelihood through a conditional Gutenberg-Richter density.
    """

    def __init__(self, args, device: Optional[torch.device] = None):
        super().__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.reduction = getattr(args, "loss_reduction", "per_event")
        self.num_divide = int(getattr(args, "njdtpp_num_divide", 10))
        if self.num_divide <= 0:
            raise ValueError("njdtpp_num_divide must be a positive integer.")

        self.dim_eta = int(
            getattr(
                args,
                "njdtpp_dim_eta",
                getattr(args, "njdtpp_num_mag_bins", 8),
            )
        )
        if self.dim_eta <= 0:
            raise ValueError("njdtpp_dim_eta must be positive.")

        self.euler_eps = float(getattr(args, "njdtpp_euler_eps", 1e-8))
        if self.euler_eps < 0.0:
            raise ValueError("njdtpp_euler_eps must be non-negative.")
        self.log_intensity_clip = float(getattr(args, "njdtpp_log_intensity_clip", 20.0))
        self.mark_b_floor = float(getattr(args, "njdtpp_mark_b_floor", 1e-3))
        if self.mark_b_floor <= 0.0:
            raise ValueError("njdtpp_mark_b_floor must be positive.")
        self.sample_step = float(getattr(args, "njdtpp_sample_step", 0.1))
        if self.sample_step <= 0.0:
            raise ValueError("njdtpp_sample_step must be positive.")
        self.sample_max_events = int(getattr(args, "njdtpp_sample_max_events", 1024))
        if self.sample_max_events <= 0:
            raise ValueError("njdtpp_sample_max_events must be positive.")
        self.sample_intensity_floor = float(getattr(args, "njdtpp_sample_intensity_floor", 1e-6))
        if self.sample_intensity_floor <= 0.0:
            raise ValueError("njdtpp_sample_intensity_floor must be positive.")
        self.sample_init_noise_scale = float(getattr(args, "njdtpp_sample_init_noise_scale", 0.0))
        if self.sample_init_noise_scale < 0.0:
            raise ValueError("njdtpp_sample_init_noise_scale must be non-negative.")

        mag_min = float(getattr(args, "mag_completeness", -2.0))
        mag_max = float(getattr(args, "mag_max", mag_min + 8.0))
        assert mag_max > mag_min, "mag_max must be greater than mag_completeness (mag_min)."

        self.register_buffer("mag_min", torch.tensor(mag_min, dtype=torch.float32))
        self.register_buffer("mag_max", torch.tensor(mag_max, dtype=torch.float32))

        dim_hidden = int(getattr(args, "njdtpp_dim_hidden", 32))
        num_hidden = int(getattr(args, "njdtpp_num_hidden", 2))
        sigma = float(getattr(args, "njdtpp_sigma", 0.01))
        self.sde = NJDSDE(
            dim_eta=self.dim_eta,
            dim_hidden=dim_hidden,
            num_hidden=num_hidden,
            sigma=sigma,
            activation=nn.Tanh(),
        )
        self.mark_encoder = MLP(
            dim_in=1,
            dim_out=self.dim_eta,
            dim_hidden=dim_hidden,
            num_hidden=num_hidden,
            sigma=sigma,
            activation=nn.Tanh(),
        )
        self.mark_b_head = MLP(
            dim_in=self.dim_eta,
            dim_out=1,
            dim_hidden=dim_hidden,
            num_hidden=max(num_hidden - 1, 0),
            sigma=sigma,
            activation=nn.Tanh(),
        )

        self.eta0 = nn.Parameter(torch.empty(self.dim_eta).normal_(mean=0.0, std=0.1))
        self.to(self.device)

    def _prepare_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        mag = mag.to(device=self.mag_min.device)
        mag_min = self.mag_min.to(dtype=mag.dtype, device=mag.device)
        mag_max = self.mag_max.to(dtype=mag.dtype, device=mag.device)
        if not torch.isfinite(mag).all():
            raise ValueError("NJDTPP received non-finite magnitudes.")
        if ((mag < mag_min) | (mag > mag_max)).any():
            raise ValueError(
                "NJDTPP received magnitudes outside the configured "
                f"[mag_min, mag_max] range: [{float(mag_min.item())}, {float(mag_max.item())}]."
            )
        return mag

    def _normalize_magnitude(self, mag: torch.Tensor) -> torch.Tensor:
        mag_min = self.mag_min.to(dtype=mag.dtype, device=mag.device)
        mag_max = self.mag_max.to(dtype=mag.dtype, device=mag.device)
        denom = (mag_max - mag_min).clamp_min(1e-8)
        return ((mag - mag_min) / denom).unsqueeze(-1)

    def _mark_weights(self, mag: torch.Tensor) -> torch.Tensor:
        weight_logits = self.mark_encoder(self._normalize_magnitude(mag))
        return torch.softmax(weight_logits, dim=-1)

    def _time_log_intensity(self, eta: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(eta, dim=-1).clamp(max=self.log_intensity_clip)

    def _total_intensity(self, eta: torch.Tensor) -> torch.Tensor:
        return torch.exp(self._time_log_intensity(eta))

    def _mark_log_prob(self, eta_left: torch.Tensor, mag: torch.Tensor) -> torch.Tensor:
        mag_dist = self._get_mark_dist(eta_left)
        return mag_dist.log_prob(mag)

    def _get_mark_dist(self, eta_left: torch.Tensor) -> dist.GutenbergRichter:
        b = F.softplus(self.mark_b_head(eta_left).squeeze(-1)) + self.mark_b_floor
        return dist.GutenbergRichter(
            b=b,
            mag_min=float(self.mag_min.item()),
            mag_max=float(self.mag_max.item()),
        )

    def _propagate_interval(
        self,
        eta_initial: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        *,
        compute_integral: bool = True,
        noise_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if eta_initial.numel() == 0:
            integral = torch.zeros(
                eta_initial.shape[0],
                device=eta_initial.device,
                dtype=eta_initial.dtype,
            )
            return eta_initial, integral

        dt_total = (t1 - t0).to(dtype=eta_initial.dtype).clamp_min(0.0)
        eta_final = eta_initial.clone()
        integral = torch.zeros(
            eta_initial.shape[0],
            device=eta_initial.device,
            dtype=eta_initial.dtype,
        )
        active = dt_total > self.euler_eps
        if not active.any():
            return eta_final, integral

        eta_t = eta_initial[active]
        dt = (dt_total[active] / self.num_divide).unsqueeze(-1)
        sqrt_dt = torch.sqrt(dt)
        if compute_integral:
            intensity_prev = self._total_intensity(eta_t)
            integral_active = torch.zeros_like(dt_total[active])

        for _ in range(self.num_divide):
            noise = torch.randn_like(eta_t) * noise_scale if noise_scale > 0.0 else torch.zeros_like(eta_t)
            eta_next = eta_t + self.sde.f(eta_t) * dt + self.sde.g(eta_t) * sqrt_dt * noise
            if compute_integral:
                intensity_next = self._total_intensity(eta_next)
                integral_active = integral_active + 0.5 * (
                    intensity_prev + intensity_next
                ) * dt.squeeze(-1)
                intensity_prev = intensity_next
            eta_t = eta_next

        eta_final[active] = eta_t
        if compute_integral:
            integral[active] = integral_active
        return eta_final, integral

    def _apply_continuous_mark_jump(
        self,
        eta_left: torch.Tensor,
        mag: torch.Tensor,
    ) -> torch.Tensor:
        jump = self.sde.h(eta_left)
        mark_weights = self._mark_weights(mag)
        jump_delta = torch.matmul(jump, mark_weights.unsqueeze(-1)).squeeze(-1)
        return eta_left + jump_delta

    def _prepare_sequence_magnitudes(
        self,
        sequence: src.data.Sequence,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "mag" not in sequence:
            if sequence.num_events == 0:
                return torch.empty(0, device=self.device, dtype=dtype)
            raise ValueError("NJDTPP requires sequence.mag for non-empty sequences.")
        mag = sequence.mag.to(self.device, dtype=dtype)
        return self._prepare_magnitude(mag)

    def _state_at_sequence_end(
        self,
        sequence: src.data.Sequence,
        *,
        noise_scale: float = 0.0,
    ) -> torch.Tensor:
        dtype = self.eta0.dtype
        eta_state = self.eta0.to(device=self.device, dtype=dtype).unsqueeze(0)
        current_time = torch.tensor([sequence.t_start], device=self.device, dtype=dtype)
        arrival = sequence.arrival_times.to(self.device, dtype=dtype)
        mag = self._prepare_sequence_magnitudes(sequence, dtype=dtype)

        for i in range(sequence.num_events):
            event_time = arrival[i].unsqueeze(0)
            eta_state, _ = self._propagate_interval(
                eta_state,
                current_time,
                event_time,
                compute_integral=False,
                noise_scale=noise_scale,
            )
            eta_state = self._apply_continuous_mark_jump(eta_state, mag[i].unsqueeze(0))
            current_time = event_time

        end_time = torch.tensor([sequence.t_end], device=self.device, dtype=dtype)
        eta_state, _ = self._propagate_interval(
            eta_state,
            current_time,
            end_time,
            compute_integral=False,
            noise_scale=noise_scale,
        )
        return eta_state.squeeze(0)

    def _propagate_interval_path(
        self,
        eta_initial: torch.Tensor,
        t0: float,
        t1: float,
        num_grid_points: int,
        *,
        noise_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eta_initial = eta_initial.to(self.device)
        dtype = eta_initial.dtype
        device = eta_initial.device
        start = torch.tensor(float(t0), device=device, dtype=dtype)
        end = torch.tensor(float(t1), device=device, dtype=dtype)
        if end <= start:
            times = start.unsqueeze(0)
            eta_path = eta_initial.unsqueeze(0)
            compensator = torch.zeros(1, device=device, dtype=dtype)
            return times, eta_path, compensator

        num_steps = max(int(num_grid_points), 1)
        times = torch.linspace(start.item(), end.item(), num_steps + 1, device=device, dtype=dtype)
        eta_t = eta_initial.clone()
        eta_path = [eta_t]
        compensator = [torch.zeros((), device=device, dtype=dtype)]
        cumulative = torch.zeros((), device=device, dtype=dtype)
        intensity_prev = self._total_intensity(eta_t.unsqueeze(0)).squeeze(0)

        for i in range(num_steps):
            dt = (times[i + 1] - times[i]).view(1, 1)
            noise = torch.randn(1, self.dim_eta, device=device, dtype=dtype) * noise_scale if noise_scale > 0.0 else torch.zeros(1, self.dim_eta, device=device, dtype=dtype)
            eta_next = eta_t.unsqueeze(0) + self.sde.f(eta_t.unsqueeze(0)) * dt + self.sde.g(eta_t.unsqueeze(0)) * torch.sqrt(dt) * noise
            eta_t = eta_next.squeeze(0)
            intensity_next = self._total_intensity(eta_next).squeeze(0)
            cumulative = cumulative + 0.5 * (intensity_prev + intensity_next) * dt.squeeze()
            eta_path.append(eta_t)
            compensator.append(cumulative.clone())
            intensity_prev = intensity_next

        return times, torch.stack(eta_path, dim=0), torch.stack(compensator, dim=0)

    def _sample_sequence(
        self,
        duration: float,
        *,
        t_start: float,
        initial_eta: torch.Tensor,
    ) -> src.data.Sequence:
        dtype = initial_eta.dtype
        current_time = float(t_start)
        t_end = float(t_start + duration)
        eta_state = initial_eta.to(self.device, dtype=dtype).clone()
        arrival_times: List[float] = []
        magnitudes: List[torch.Tensor] = []

        while current_time < t_end and len(arrival_times) < self.sample_max_events:
            remaining = t_end - current_time
            step = min(self.sample_step, remaining)
            if step <= self.euler_eps:
                break

            lambda_curr = float(
                self._total_intensity(eta_state.unsqueeze(0)).clamp_min(self.sample_intensity_floor).item()
            )
            wait_time = float(torch.distributions.Exponential(rate=eta_state.new_tensor(lambda_curr)).sample().item())

            if wait_time < step:
                event_time = min(current_time + max(wait_time, self.euler_eps), t_end)
                eta_next, _ = self._propagate_interval(
                    eta_state.unsqueeze(0),
                    eta_state.new_tensor([current_time]),
                    eta_state.new_tensor([event_time]),
                    compute_integral=False,
                    noise_scale=1.0,
                )
                eta_state = eta_next.squeeze(0)
                next_mag = self._get_mark_dist(eta_state.unsqueeze(0)).sample().squeeze(0)
                arrival_times.append(event_time)
                magnitudes.append(next_mag.detach().clone())
                eta_state = self._apply_continuous_mark_jump(eta_state.unsqueeze(0), next_mag.unsqueeze(0)).squeeze(0)
                current_time = event_time
                continue

            eta_next, _ = self._propagate_interval(
                eta_state.unsqueeze(0),
                eta_state.new_tensor([current_time]),
                eta_state.new_tensor([current_time + step]),
                compute_integral=False,
                noise_scale=1.0,
            )
            eta_state = eta_next.squeeze(0)
            current_time += step

        boundaries = torch.tensor(
            [t_start, *arrival_times, t_end],
            device=self.device,
            dtype=dtype,
        )
        inter_times = torch.diff(boundaries)
        mag = torch.stack(magnitudes).to(device=self.device, dtype=dtype) if magnitudes else torch.empty(0, device=self.device, dtype=dtype)
        return src.data.Sequence(
            inter_times=inter_times,
            t_start=t_start,
            t_nll_start=t_start,
            mag=mag,
        )

    def _forward_nll_components_per_sequence(
        self,
        batch: src.data.Batch,
    ) -> dict[str, torch.Tensor]:
        if "mag" not in batch:
            raise ValueError("NJDTPP requires batch.mag because it models continuous magnitudes.")

        model_dtype = self.eta0.dtype
        arrival = batch.arrival_times.to(self.device, dtype=model_dtype)
        mag = batch.mag.to(self.device, dtype=model_dtype)
        event_mask = batch.nll_event_mask.bool().to(self.device)
        input_mask = batch.input_mask.bool().to(self.device)
        t_start = batch.t_start.to(self.device, dtype=model_dtype)
        t_end = batch.t_end.to(self.device, dtype=model_dtype)
        t_nll_start = batch.t_nll_start.to(self.device, dtype=model_dtype)
        end_idx = batch.end_idx.to(self.device)

        if mag.shape[:2] != arrival.shape[:2]:
            raise ValueError(
                "NJDTPP expects batch.mag to be padded to the same leading shape as "
                "batch.arrival_times."
            )

        batch_size = arrival.shape[0]
        valid_mag = mag[input_mask]
        if valid_mag.numel() > 0:
            self._prepare_magnitude(valid_mag)

        eta_state = (
            self.eta0.to(device=self.device, dtype=arrival.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
        )
        current_time = t_start.clone()
        time_event_term = torch.zeros(batch_size, device=self.device, dtype=arrival.dtype)
        mark_event_term = torch.zeros(batch_size, device=self.device, dtype=arrival.dtype)
        integral_seq = torch.zeros(batch_size, device=self.device, dtype=arrival.dtype)
        max_events = int(end_idx.max().item()) if batch_size > 0 else 0

        for i in range(max_events):
            valid_idx = torch.where(end_idx > i)[0]
            if valid_idx.numel() == 0:
                continue

            interval_start = current_time[valid_idx]
            event_time = arrival[valid_idx, i]
            split_time = torch.minimum(
                torch.maximum(t_nll_start[valid_idx], interval_start),
                event_time,
            )

            pre_mask = split_time > interval_start
            if pre_mask.any():
                pre_idx = valid_idx[pre_mask]
                eta_state[pre_idx], _ = self._propagate_interval(
                    eta_state[pre_idx],
                    current_time[pre_idx],
                    split_time[pre_mask],
                    compute_integral=False,
                )

            post_mask = event_time > split_time
            if post_mask.any():
                post_idx = valid_idx[post_mask]
                eta_state[post_idx], integral = self._propagate_interval(
                    eta_state[post_idx],
                    split_time[post_mask],
                    event_time[post_mask],
                )
                integral_seq[post_idx] = integral_seq[post_idx] + integral

            eta_left = eta_state[valid_idx]
            time_log_intensity = self._time_log_intensity(eta_left)
            mark_log_prob = self._mark_log_prob(
                eta_left,
                mag[valid_idx, i],
            )
            eta_right = self._apply_continuous_mark_jump(eta_left, mag[valid_idx, i])
            scored_events = event_mask[valid_idx, i]
            if scored_events.any():
                scored_idx = valid_idx[scored_events]
                time_event_term[scored_idx] = (
                    time_event_term[scored_idx] + time_log_intensity[scored_events]
                )
                mark_event_term[scored_idx] = (
                    mark_event_term[scored_idx] + mark_log_prob[scored_events]
                )

            eta_state[valid_idx] = eta_right
            current_time[valid_idx] = event_time

        split_time = torch.minimum(torch.maximum(t_nll_start, current_time), t_end)
        pre_mask = split_time > current_time
        if pre_mask.any():
            eta_state[pre_mask], _ = self._propagate_interval(
                eta_state[pre_mask],
                current_time[pre_mask],
                split_time[pre_mask],
                compute_integral=False,
            )
            current_time[pre_mask] = split_time[pre_mask]

        post_mask = t_end > current_time
        if post_mask.any():
            eta_state[post_mask], integral = self._propagate_interval(
                eta_state[post_mask],
                current_time[post_mask],
                t_end[post_mask],
            )
            integral_seq[post_mask] = integral_seq[post_mask] + integral

        nll_time = integral_seq - time_event_term
        nll_mark = -mark_event_term
        return {
            "time": nll_time,
            "mark": nll_mark,
            "total": nll_time + nll_mark,
        }

    def nll_loss(
        self,
        batch: src.data.Batch,
        *,
        reduction: str | None = None,
        return_dict: bool = False,
        eps: float = 1e-10,
    ) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        reduction = self.reduction if reduction is None else reduction
        out = self._forward_nll_components_per_sequence(batch)
        out = self.reduce_nll_dict(out, batch, reduction=reduction, eps=eps)
        if return_dict:
            return out
        return out["total"]

    @torch.inference_mode()
    def sample(
        self,
        batch_size: int,
        duration: float,
        t_start: float = 0.0,
        past_seq: Optional[src.data.Sequence] = None,
        return_sequences: bool = False,
    ) -> Union[src.data.Batch, List[src.data.Sequence]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if duration <= 0.0:
            raise ValueError("duration must be positive.")

        if past_seq is not None:
            initial_eta = self._state_at_sequence_end(
                past_seq,
                noise_scale=self.sample_init_noise_scale,
            )
            sample_t_start = float(past_seq.t_end)
        else:
            initial_eta = self.eta0.to(self.device)
            sample_t_start = float(t_start)

        sequences = [
            self._sample_sequence(
                duration=duration,
                t_start=sample_t_start,
                initial_eta=initial_eta,
            )
            for _ in range(batch_size)
        ]
        if return_sequences:
            return sequences
        return src.data.Batch.from_list(sequences)

    @torch.inference_mode()
    def evaluate_intensity(
        self,
        sequence: src.data.Sequence,
        num_grid_points: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_grid_points <= 0:
            raise ValueError("num_grid_points must be positive.")

        dtype = self.eta0.dtype
        eta_state = self.eta0.to(self.device, dtype=dtype)
        current_time = float(sequence.t_start)
        arrival = sequence.arrival_times.to(self.device, dtype=dtype)
        mag = self._prepare_sequence_magnitudes(sequence, dtype=dtype)

        grid_chunks: List[torch.Tensor] = []
        intensity_chunks: List[torch.Tensor] = []

        def append_chunk(times: torch.Tensor, values: torch.Tensor, *, drop_first: bool) -> None:
            if drop_first and times.numel() > 1:
                times = times[1:]
                values = values[1:]
            if times.numel() > 0:
                grid_chunks.append(times)
                intensity_chunks.append(values)

        for i in range(sequence.num_events):
            times, eta_path, _ = self._propagate_interval_path(
                eta_state,
                current_time,
                float(arrival[i].item()),
                num_grid_points,
                noise_scale=0.0,
            )
            intensities = self._total_intensity(eta_path)
            append_chunk(times, intensities, drop_first=bool(grid_chunks))
            eta_state = self._apply_continuous_mark_jump(eta_path[-1].unsqueeze(0), mag[i].unsqueeze(0)).squeeze(0)
            jump_time = arrival[i].reshape(1)
            jump_intensity = self._total_intensity(eta_state.unsqueeze(0)).reshape(1)
            grid_chunks.append(jump_time)
            intensity_chunks.append(jump_intensity)
            current_time = float(arrival[i].item())

        times, eta_path, _ = self._propagate_interval_path(
            eta_state,
            current_time,
            float(sequence.t_end),
            num_grid_points,
            noise_scale=0.0,
        )
        intensities = self._total_intensity(eta_path)
        append_chunk(times, intensities, drop_first=bool(grid_chunks))
        return torch.cat(grid_chunks), torch.cat(intensity_chunks)

    @torch.inference_mode()
    def evaluate_compensator(
        self,
        sequence: src.data.Sequence,
        num_grid_points: int = 50,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_grid_points <= 0:
            raise ValueError("num_grid_points must be positive.")

        dtype = self.eta0.dtype
        eta_state = self.eta0.to(self.device, dtype=dtype)
        current_time = float(sequence.t_start)
        arrival = sequence.arrival_times.to(self.device, dtype=dtype)
        mag = self._prepare_sequence_magnitudes(sequence, dtype=dtype)

        grid_chunks: List[torch.Tensor] = []
        compensator_chunks: List[torch.Tensor] = []
        offset = torch.zeros((), device=self.device, dtype=dtype)

        def append_chunk(times: torch.Tensor, values: torch.Tensor, *, drop_first: bool) -> None:
            if drop_first and times.numel() > 1:
                times = times[1:]
                values = values[1:]
            if times.numel() > 0:
                grid_chunks.append(times)
                compensator_chunks.append(values)

        for i in range(sequence.num_events):
            times, eta_path, compensator = self._propagate_interval_path(
                eta_state,
                current_time,
                float(arrival[i].item()),
                num_grid_points,
                noise_scale=0.0,
            )
            append_chunk(times, compensator + offset, drop_first=bool(grid_chunks))
            offset = offset + compensator[-1]
            eta_state = self._apply_continuous_mark_jump(eta_path[-1].unsqueeze(0), mag[i].unsqueeze(0)).squeeze(0)
            current_time = float(arrival[i].item())

        times, _, compensator = self._propagate_interval_path(
            eta_state,
            current_time,
            float(sequence.t_end),
            num_grid_points,
            noise_scale=0.0,
        )
        append_chunk(times, compensator + offset, drop_first=bool(grid_chunks))
        return torch.cat(grid_chunks), torch.cat(compensator_chunks)
