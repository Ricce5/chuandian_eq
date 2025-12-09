import abc
import torch
from src.utils.registrable import Registrable
from src.data.dot_dict import DotDict
import torchcde
import warnings
from src.utils.interp import (
    integrate_uniform_time_series,
    interp_uniform_time_series,
)

class BGModel(torch.nn.Module, abc.ABC, Registrable):
    def __init__(self, device,scale_init=200.0):
        super().__init__()  
        self.device = device
        log_init = torch.log(torch.tensor(scale_init, device=device))
        self.log_scale = torch.nn.Parameter(log_init, requires_grad=True)
        
    @property
    def _scale(self):
        return torch.exp(self.log_scale)  
    
    @abc.abstractmethod
    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    
    def intensity(
        self,
        ts_batch: DotDict,
        t_query: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Compute intensity :math:`\lambda(t)` for given batch and query times.

        Args:
            ts_batch: batch with keys
                - ``time_series``: (B, T, F)
                - ``time_series_times``: (B, T)
                - optionally ``arrival_times``: (B, Nq)
            t_query: optional query times of shape (B, Nq) or (Nq,). If ``None``,
                use ``ts_batch.arrival_times`` when available, otherwise
                ``time_series_times``.
            return_weights: if ``True``, also return the per-timepoint
                weight vectors from the Mamba SSM.

        Returns:
            ``(B, Nq)`` tensor of non-negative intensities, or a tuple
            ``(intensity, weights_over_time)`` when ``return_weights=True``.
        """

        time_series = ts_batch.time_series.to(self.device)  # (B, T, F)
        time_series_times = ts_batch.time_series_times.to(self.device)  # (B, T)
        
        scaled_intensity = self.scaled_intensity(time_series)  # (B, T, 1)
        intensity_traj = scaled_intensity * self._scale  # (B, T, 1)

        # decide query times
        if t_query is None:
            arrival_times = getattr(ts_batch, "arrival_times", None)
            if arrival_times is not None:
                t_query = arrival_times.to(self.device)
            else:
                t_query = time_series_times
        else:
            t_query = t_query.to(self.device)

        # interpolate to requested query times, result (B, Nq, 1)
        intensity = interp_uniform_time_series(
            t=time_series_times,
            x=intensity_traj,
            t_query=t_query,
        )

        intensity = intensity.squeeze(-1).clamp_min(0.0)  # (B, Nq)
        return intensity

    # -------------------------------------------------------------
    # ∫ λ(t) dt
    # -------------------------------------------------------------
    def intensity_integral(self, batch: DotDict) -> torch.Tensor:
        """Compute integral of intensity over ``[t_start, t_end]`` for each item.

        Uses the same definition of intensity trajectory as :meth:`intensity`,
        then integrates it over time.

        Args:
            batch: DotDict with keys ``time_series``, ``time_series_times``,
                ``t_nll_start`` and ``t_end``.

        Returns:
            Tensor of shape ``(B,)`` with integrals.
        """
        time_series = batch.time_series.to(self.device)  # (B, T, F)
        time_series_times = batch.time_series_times.to(self.device)  # (B, T)

        # match intensity() definition: linear projection then positive weights
        scaled_intensity = self.scaled_intensity(time_series)  # (B, T, 1)
        intensity_traj = scaled_intensity * self._scale
        integral = integrate_uniform_time_series(
            t=time_series_times,
            x=intensity_traj,
            t_start=batch.t_nll_start,
            t_end=batch.t_end,
        )  # (B, 1)

        return integral.squeeze(-1)

    
    def nll(self, batch: DotDict) -> torch.Tensor: 
        """
        Compute negative log-likelihood loss for the background model:
        - log λ(t_i) at event times plus integral over observation window.
        Args:
            batch: DotDict used for intensity evaluation (must provide arrival_times and event_mask)
        Returns:
            Tensor of shape (B,) with negative log-likelihood (to be minimized).
        """
        f_intensity = self.intensity(batch)                 # (B, Nq)
        f_intensity_integral = self.intensity_integral(batch)  # (B,)
        f_intensity = f_intensity * batch.nll_event_mask           # (B, Nq)
        log_intensity = torch.clamp_min(f_intensity, 1e-8).log()  # (B, Nq)
        # print(f"log_intensity: {log_intensity},")
        log_like = log_intensity.sum(dim=1) - f_intensity_integral  # (B,)
        return -log_like

        


    def nll_change(self, batch: DotDict, log_h_intensity: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood change for importance sampling:
        - log p_f/p_h on event times minus integral over interval (f intensity).
        Args:
            batch: DotDict used for intensity evaluation (must provide arrival_times and nll_event_mask)
            log_h_intensity: log of proposal intensity h evaluated at event query times, shape (B, Nq)
        Returns:
            Tensor of shape (B,) with negative log-likelihood change (to be minimized).
        """
        f_intensity = self.intensity(batch)                 # (B, Nq)
        f_intensity_integral = self.intensity_integral(batch)  # (B,)
        h_intensity = torch.exp(log_h_intensity)            # (B, Nq)
        # protect against zeros in denominator
        denom = h_intensity.clamp_min(1e-8)
        print(f"f_intensity max:     {f_intensity.max().item()}, h_intensity max: {h_intensity.max().item()}")
        ratio = f_intensity / denom                         # (B, Nq)
        # log change per event-time, only where events are present (mask)
        mask = getattr(batch, "nll_event_mask", None)
        if mask is None:
            raise ValueError("batch must contain 'nll_event_mask' for nll_change computation.")
        log_change = torch.log1p(ratio) * mask              # (B, Nq)
        # sum over query/event times and subtract integral contribution
        log_like_change = log_change.sum(dim=1) - f_intensity_integral  # (B,)
        # return negative log-likelihood change (for minimization)
        return -log_like_change

    def cache_batch(self, time_series, time_series_times):
        assert time_series.shape[0] == time_series_times.shape[0]==1, "Batch size should be 1."
        self.ts_batch_cache = DotDict({
            "time_series": time_series,
            "time_series_times": time_series_times,
        }).to(self.device)
    

    def sample_nhpp(self, B, t0, dt, n_grid=10, return_times_list=False, eps_zero: float = 1e-12):
        """
        Parallel sampling of the first-event waiting time for B NHPPs (relative to t0).
        If the intensity λ(t) is (almost) zero throughout [t0, t0+dt], returns dt — no event.

        Args:
            B  : number of parallel samples
            t0 : start absolute time, can be scalar / length-B vector / tensor
            dt : interval length (t1 - t0), scalar / length-B vector / tensor
            n_grid : number of grid points to estimate λ_max
            return_times_list : if True, also return per-sample list of event times (either [tau] or [])
            eps_zero: threshold for treating λ as zero over the interval
        Returns:
            out: tensor of shape (B,), the i‑th element is the time from t0 to first event,
                or dt[i] if no event in [t0, t0+dt].
            If return_times_list=True, also returns times_list.
        """
        cached = getattr(self, "ts_batch_cache", None)
        assert cached is not None, "Batch data must be cached before sampling."

        device = self.device

        def to_batched_tensor(x, name):
            if torch.is_tensor(x):
                x = x.to(device)
                if x.dim() == 0:
                    x = x.expand(B)
                elif x.numel() == 1:
                    x = x.reshape(1).expand(B)
                else:
                    assert x.shape[0] == B, f"{name} must have length B"
                return x.reshape(B)
            else:
                x_t = torch.tensor(float(x), device=device)
                return torch.full((B,), x_t.item(), device=device)

        with torch.no_grad():
            t0_b = to_batched_tensor(t0, "t0")   # (B,)
            dt_b = to_batched_tensor(dt, "dt")   # (B,)

            # estimate λ_max on a grid
            u = torch.linspace(0.0, 1.0, n_grid, device=device).unsqueeze(0)  # (1, n_grid)
            t_query_rel = dt_b.unsqueeze(1) * u                               # (B, n_grid)
            t_query_abs = t0_b.unsqueeze(1) + t_query_rel                     # (B, n_grid)
            lam = self.intensity(cached, t_query=t_query_abs)                 # (B, n_grid)

            # check if intensity is (almost) zero throughout
            max_lam_per_batch = lam.max(dim=1).values  # (B,)
            if (max_lam_per_batch < eps_zero).all():
                out = dt_b.clone()
                if return_times_list:
                    return [[] for _ in range(B)], out
                return out

            # otherwise, proceed with thinning using λ_max
            lambda_max = max_lam_per_batch.max().item() * 1.2
            lambda_max = max(lambda_max, eps_zero)

            times_list = [[] for _ in range(B)]
            tau = torch.zeros(B, device=device)
            has_event = torch.zeros(B, dtype=torch.bool, device=device)

            while True:
                active = (~has_event) & (tau < dt_b)
                if not active.any():
                    break

                w = torch.zeros(B, device=device)
                num_active = active.sum().item()
                w_active = -torch.log(torch.rand(num_active, device=device)) / lambda_max
                w[active] = w_active
                tau = tau + w

                inside = (~has_event) & (tau < dt_b)
                if not inside.any():
                    break

                t_q_abs = t0_b[inside] + tau[inside]
                lam_t = self.intensity(cached, t_query=t_q_abs.unsqueeze(1)).squeeze(-1)
                accept_prob = (lam_t / lambda_max).clamp(max=1.0)
                u2 = torch.rand(accept_prob.shape[0], device=device)
                accept_local = u2 < accept_prob

                if accept_local.any():
                    inside_idx = torch.nonzero(inside, as_tuple=False).squeeze(1)
                    accepted_idx = inside_idx[accept_local]
                    for idx in accepted_idx.tolist():
                        if not has_event[idx]:
                            times_list[idx].append(tau[idx].item())
                            has_event[idx] = True

            out_vals = []
            for i in range(B):
                if times_list[i]:
                    tau_i = times_list[i][0]
                    out_vals.append(tau_i)
                else:
                    out_vals.append(float(dt_b[i].item()))
            out = torch.tensor(out_vals, device=device)

        if return_times_list:
            return times_list, out
        return out


    
    def sample_nhpp_inverse(self, B, t0, dt):
        cached = getattr(self, "ts_batch_cache", None)
        assert cached is not None, "Batch data must be cached before sampling."
        device = self.device

        def to_batched_tensor(x, name):
            if torch.is_tensor(x):
                x = x.to(device)
                if x.dim() == 0:
                    x = x.expand(B)
                elif x.numel() == 1:
                    x = x.reshape(1).expand(B)
                else:
                    assert x.shape[0] == B, f"{name} must have length B"
                return x.reshape(B)
            else:
                x_t = torch.tensor(float(x), device=device)
                return torch.full((B,), x_t.item(), device=device)

        with torch.no_grad():
            t0_b = to_batched_tensor(t0, "t0")
            dt_b = to_batched_tensor(dt, "dt")

            t0_min = t0_b.min().item()
            t1_max = (t0_b + dt_b).max().item()

            # get cached series (assume batch size 1 was cached)
            ts_times = cached["time_series_times"]  # expected shape (1, T)
            ts_values = cached["time_series"]       # expected shape (1, T, F)

            # flatten the cached time axis
            ts_times_1d = ts_times.reshape(-1)  # (T,)
            # select points inside [t0_min, t1_max]
            mask = (ts_times_1d >= t0_min) & (ts_times_1d <= t1_max)
            if mask.sum().item() == 0:
                # no support points in interval -> no event
                return dt_b.clone()

            x = ts_times_1d[mask].to(device=device)  # (Nq,)
            # prepare t_query for intensity: shape (1, Nq) because cached has batch dim
            t_query = x.unsqueeze(0)  # (1, Nq)

            # compute intensity at these absolute times; intensity returns (B_cached=1, Nq)
            lam = self.intensity(cached, t_query=t_query)  # (1, Nq)
            lam_1d = lam.squeeze(0)                        # (Nq,)

            # build cif inverse/forward on device/dtype of x
            interp = build_cif_inverse(x, lam_1d, num_dense=10000)
            cif_inverse = interp["cif_inverse"]
            cif_forward = interp["cif_forward"]

            # evaluate CIF at each t0 in the batch
            # cif_forward expects times in the same domain as x (absolute times)
            cif_t0 = cif_forward.evaluate(t0_b).squeeze()          # (B,)
            cif_t1 = cif_forward.evaluate(t0_b + dt_b).squeeze()  # (B,)
            cif_max = cif_t1 - cif_t0  # (B,)

            # sample exponential rates and map through inverse CIF
            tau = dt_b.clone()  # default: no event (return dt)
            tau_orig = -torch.log(torch.rand(B, device=device))  # (B,)
            valid_mask = tau_orig < cif_max

            if valid_mask.any():
                cif_query = (tau_orig[valid_mask] + cif_t0[valid_mask]).to(device)
                # cif_inverse.evaluate returns shape (1, M, 1) -> squeeze to (M,)
                tau_abs_valid = cif_inverse.evaluate(cif_query).squeeze()  # absolute times
                # subtract corresponding t0 to get waiting times
                tau_valid = tau_abs_valid - t0_b[valid_mask]
                tau[valid_mask] = tau_valid
                print(f"sample_nhpp_inverse: sampled {valid_mask.sum().item()} events out of {B}.")
            # Ensure sampled taus do not exceed dt due to numerical error: clamp and warn if needed.
            diff = tau - dt_b
            if (diff > 1e-6).any():
                warnings.warn(f"sample_nhpp_inverse: sampled tau exceeds dt by up to {diff.max().item():.3e}; clamping to dt.")
                tau = torch.minimum(tau, dt_b)
        return tau


def build_cif_inverse(x, y, num_dense=10000):
    assert x.ndim == 1 and y.ndim == 1 and x.shape == y.shape
    forward_interp = torchcde.LinearInterpolation(y.unsqueeze(0).unsqueeze(-1), t=x)
    x_dense = torch.linspace(x[0], x[-1], num_dense, device=x.device, dtype=x.dtype)
    y_dense = forward_interp.evaluate(x_dense).squeeze()  # (num_dense,)

    dx = x_dense[1] - x_dense[0]
    cif_dense = torch.cumsum(y_dense, dim=0) * dx
    cif_inverse = torchcde.LinearInterpolation(x_dense.unsqueeze(0).unsqueeze(-1), t=cif_dense)
    cif_forward = torchcde.LinearInterpolation(cif_dense.unsqueeze(0).unsqueeze(-1), t=x_dense)

    return {
        "if": forward_interp,
        "cif_inverse": cif_inverse,
        "cif_forward": cif_forward,
        'cif': cif_dense
    }
