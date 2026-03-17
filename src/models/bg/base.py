import abc
import torch
from src.distributions import clamp_preserve_gradients
from src.utils.registrable import Registrable
from src.data.dot_dict import DotDict
import torchcde
import warnings
from src.utils.interp import (
    integrate_uniform_time_series,
    interp_uniform_time_series,
)

class BGModel(torch.nn.Module, abc.ABC, Registrable):
    def __init__(self, device,scale_init=200.0,no_weight_decay=False):
        super().__init__()  
        self.device = device
        log_init = torch.log(torch.tensor(scale_init, device=device, dtype=torch.float32))
        self.log_scale = torch.nn.Parameter(log_init, requires_grad=True)
        if no_weight_decay:
            self.log_scale._no_weight_decay = True  
        
    @property
    def _scale(self):
        return torch.exp(self.log_scale)  

    def set_scale(self, scale_value: float):
        with torch.no_grad():
            self.log_scale.copy_(torch.log(torch.tensor(scale_value, device=self.device, dtype=torch.float32)))

     # -------------------------------------------------------------
    
    @abc.abstractmethod
    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:
        """
        scaled_intensity 
        Args:
        To make sure the output is zero when the input is zero, linear layers should not have bias terms and the activation functions should satisfy f(0)=0.
        """
        raise NotImplementedError

    def _compute_intensity_traj(self, ts_batch: DotDict):
        """Helper to compute uniform-grid intensity trajectory (B, T, 1) and times (B, T)."""
        time_series = ts_batch.time_series.to(self.device, dtype=torch.float32)  # (B, T, F)
        time_series_times = ts_batch.time_series_times.to(self.device)  # (B, T)
        scaled_intensity = self.scaled_intensity(time_series)  # (B, T, 1)
        intensity_traj = scaled_intensity * self._scale  # (B, T, 1)
        ts_mask = getattr(ts_batch, "time_series_mask", None)
        if ts_mask is not None:
            intensity_traj = intensity_traj * ts_mask.to(self.device).unsqueeze(-1)
        # forward = clamped, backward = identity (keep gradients)
        intensity_traj = intensity_traj + (intensity_traj.clamp_min(0.0) - intensity_traj).detach()
        return time_series_times, intensity_traj

    def intensity_trajectory(
        self,
        ts_batch: DotDict,
    ) -> torch.Tensor:
        """Compute intensity trajectory on the stored uniform grid: (B, T)."""
        _, intensity_traj = self._compute_intensity_traj(ts_batch)
        return intensity_traj.squeeze(-1)  # (B, T)

    def intensity(
        self,
        ts_batch: DotDict,
        t_query: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute intensity lambda(t) for given batch and query times, returning (B, Nq)."""
        time_series_times, intensity_traj = self._compute_intensity_traj(ts_batch)

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
            clamp=True,
        )
        return intensity.squeeze(-1)  # (B, Nq)

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
        # match intensity() forward/backward behavior by reusing the same trajectory construction
        time_series_times, intensity_traj = self._compute_intensity_traj(batch)
        integral = integrate_uniform_time_series(
            t=time_series_times,
            x=intensity_traj,
            t_start=batch.t_nll_start,
            t_end=batch.t_end,
        )  # (B, 1)
        return integral.squeeze(-1).squeeze(-1)  # (B,)
    
    @torch.no_grad()
    def forecast_count(self, t_start: torch.Tensor, t_end: torch.Tensor) -> torch.Tensor:
        assert self.ts_batch_cache is not None, "Batch data must be cached before forecasting." 
        assert self.lambda_cache is not None, "Lambda cache must be available before forecasting."
        integral = integrate_uniform_time_series(
            t=self.ts_batch_cache.time_series_times,
            x=self.lambda_cache.unsqueeze(0).unsqueeze(-1),
            t_start=torch.tensor(t_start, device=self.device),
            t_end=torch.tensor(t_end, device=self.device),
        )  # (B, 1)
        return integral.squeeze()

    
    def nll(self, batch: DotDict, eps: float = 1e-8) -> torch.Tensor: 
        """
        Compute negative log-likelihood loss for the background model:
        - sum log λ(t_i) over events selected by ``nll_event_mask``
        - plus integral over the evaluation window ``[t_nll_start, t_end]``.
        Args:
            batch: DotDict used for intensity evaluation. Must provide
                ``arrival_times``, ``nll_event_mask``, ``t_nll_start`` and ``t_end``.
        Returns:
            Tensor of shape (B,) with negative log-likelihood (to be minimized).
        """
        f_intensity = self.intensity(batch)                 # (B, Nq)
        f_intensity_safe = clamp_preserve_gradients(f_intensity, eps, float("inf"))
        log_intensity = torch.log(f_intensity_safe)         # (B, Nq)
        f_intensity_integral = self.intensity_integral(batch)  # (B,)
        log_intensity = log_intensity * batch.nll_event_mask  # (B, Nq)
        log_like = log_intensity.sum(dim=1) - f_intensity_integral  # (B,)
        return -log_like

        


    def nll_change(self, batch: DotDict, log_h_intensity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        log1p(f_intensity / h_intensity)- f_intensity_integral
        """
        f_intensity = self.intensity(batch)                 # (B, Nq)
        f_intensity_integral = self.intensity_integral(batch)  # (B,)
        h_intensity = torch.exp(log_h_intensity)            # (B, Nq)
        # protect against zeros in denominator
        denom = clamp_preserve_gradients(h_intensity, eps, float("inf"))
        ratio = f_intensity / denom                         # (B, Nq)
        # log change per event-time, only where events are present (mask)
        mask = getattr(batch, "nll_event_mask", None)
        if mask is None:
            raise ValueError("batch must contain 'nll_event_mask' for nll_change computation.")
        log_change = torch.log1p(ratio) * mask              # (B, Nq)
        # sum over query/event times and subtract integral contribution
        log_like_change = log_change.sum(dim=1) - f_intensity_integral  # (B,)
        # return negative log-likelihood change (for minimization)
        # print(f"NLL change: { -log_like_change.mean().item() }")
        return -log_like_change

    def cache_batch(self, time_series, time_series_times,cache_lambda=True):
        assert time_series.shape[0] == time_series_times.shape[0]==1, "Batch size should be 1."
        self.ts_batch_cache = DotDict({
            "time_series": time_series,
            "time_series_times": time_series_times,
        }).to(self.device)
        self.lambda_cache = None if not cache_lambda else self.intensity(self.ts_batch_cache).squeeze(0)
    



    def _to_batched_tensor(self, x, name, B, dtype=None):
        device = self.device
        if torch.is_tensor(x):
            x = x.to(device)
            if dtype is not None:
                x = x.to(dtype)
            if x.dim() == 0:
                return x.expand(B).reshape(B)
            if x.numel() == 1:
                return x.reshape(1).expand(B).reshape(B)
            assert x.shape[0] == B, f"{name} must have length B"
            return x.reshape(B)
        return torch.full((B,), float(x), device=device, dtype=dtype or torch.float32)
    

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


        with torch.no_grad():
            t0_b = self._to_batched_tensor(t0, "t0", B)   # (B,)
            dt_b = self._to_batched_tensor(dt, "dt", B)   # (B,)

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


    @torch.no_grad()
    def sample_nhpp_inverse(
        self,
        B,
        t0,
        dt,
        eps_lam: float = 1e-12,
        eps_disc: float = 0.0,
        use_fp64: bool = False,  
        sample_sequence: bool = False,
        mu : float = 0.0,       
    ):
        """
        Inverse-CDF sampling of first-event waiting times for B NHPPs using a cached
        uniform grid. Returns dt for samples with no event in [t0, t0+dt].
        Args:
            B  : number of parallel samples
            t0 : start absolute time, can be scalar / length-B vector / tensor
            dt : interval length (t1 - t0), scalar / length-B vector / tensor
            eps_lam: threshold for treating λ as zero
            eps_disc: threshold for discriminant in quadratic solver
            use_fp64: if True, use float64 for time/index/integral/CIF computations
            sample_sequence: if True, return full event time sequences, else only first event times
            mu : float, background intensity to add to the cached intensity
        """
        cached = getattr(self, "ts_batch_cache", None)
        assert cached is not None, "Batch data must be cached before sampling."
        device = self.device

        # --- helpers ---

        def build_window_grid(ts_times_full, t0_min, t1_max):
            T = ts_times_full.numel()
            assert T >= 2, "Need at least 2 time points in cached grid."
            dt_grid = (ts_times_full[1] - ts_times_full[0])
            ts0 = ts_times_full[0]
            tsN = ts_times_full[-1]

            i0 = max(int(torch.floor((t0_min - ts0) / dt_grid).item()) - 1, 0)
            i1 = min(int(torch.ceil((t1_max - ts0) / dt_grid).item()) + 1, T - 1)
            ts_times = ts_times_full[i0 : i1 + 1]  # (Tw,)
            return dt_grid, ts0, tsN, ts_times, i0, i1

        def build_cif_from_lam(lam, dt_grid):
            """Build CIF from intensity values on uniform grid."""
            Tw = lam.numel()
            if Tw < 2:
                return None
            cif = torch.zeros_like(lam)
            cif[1:] = torch.cumsum(0.5 * (lam[:-1] + lam[1:]) * dt_grid, dim=0)
            return cif

        def solve_segment_s(delta_L, lam0, lam1, dt_grid_local):
            dlam = (lam1 - lam0) / dt_grid_local
            s_local = torch.zeros_like(delta_L)

            linear_mask = dlam.abs() < 1e-8
            if linear_mask.any():
                s_local[linear_mask] = delta_L[linear_mask] / lam0[linear_mask].clamp_min(eps_lam)

            nonlin_mask = ~linear_mask
            if nonlin_mask.any():
                A = 0.5 * dlam[nonlin_mask]
                Bq = lam0[nonlin_mask]
                C = -delta_L[nonlin_mask]
                disc = (Bq * Bq - 4.0 * A * C).clamp_min(eps_disc)
                s_local[nonlin_mask] = (-Bq + torch.sqrt(disc)) / (2.0 * A)
            return s_local

        def invert_cif_targets(targets, cif, lam, dt_grid_local, ts_rel_local):
            """Invert CIF for a 1D tensor of targets -> relative times.
                targets: any shape
            """
            Tw_local = lam.numel()
            if Tw_local < 2 or targets.numel() == 0:
                return torch.empty_like(targets)

            idx = torch.searchsorted(cif, targets)
            idx = idx.clamp(min=1, max=Tw_local - 1)
            idx0 = idx - 1

            lam0_seg = lam[idx0]
            lam1_seg = lam[idx0 + 1]
            delta_L = targets - cif[idx0]

            s = solve_segment_s(delta_L, lam0_seg, lam1_seg, dt_grid_local)
            t_event_rel = ts_rel_local[idx0] + s
            return t_event_rel

        # --- dtype policy ---
        # Comment in English.
        t_dtype = torch.float64 if use_fp64 else torch.float32

        # --- main flow ---
        t0_b = self._to_batched_tensor(t0, "t0", B, dtype=t_dtype)
        dt_b = self._to_batched_tensor(dt, "dt", B, dtype=t_dtype)
        t1_b = t0_b + dt_b

        # cached uniform time grid (assumes batch dim was 1)
        ts_times_full = cached.time_series_times.squeeze(0).to(device).to(t_dtype)  # (T,)

        t0_min = t0_b.min()
        t1_max = t1_b.max().clamp_max(ts_times_full[-1])

        dt_grid, ts0, tsN, ts_times, i0, i1 = build_window_grid(ts_times_full, t0_min, t1_max)
        assert t0_min >= ts0 and t1_max <= tsN, (
            f"[sample_nhpp_inverse] Query interval [{t0_min.item():.4f}, {t1_max.item():.4f}] "
            f"out of cached range [{ts0.item():.4f}, {tsN.item():.4f}]."
        )

        # intensity on full grid and restrict to window
        lam_full = self.lambda_cache if self.lambda_cache is not None else self.intensity(cached, t_query=ts_times_full.unsqueeze(0)).squeeze(0)  # (T,)
    

        lam = lam_full[i0 : i1 + 1]+ mu  # (Tw,)
        lam = lam.to(t_dtype) if use_fp64 else lam

        Tw = lam.numel()
        if Tw < 2:
            tau_fallback = dt_b.clone().to(self.device)
            if sample_sequence:
                return [[] for _ in range(B)]
            return tau_fallback

        # Comment in English.
        t_shift = ts_times[0]
        ts_rel = ts_times - t_shift        # (Tw,)
        t0_rel = t0_b - t_shift            # (B,)
        t1_rel = t1_b - t_shift            # (B,)

        # Comment in English.
        cif = build_cif_from_lam(lam, dt_grid)
        if cif is None:
            tau_fallback = dt_b.clone().to(self.device)
            if sample_sequence:
                return [[] for _ in range(B)]
            return tau_fallback

        # CIF at relative times in [0, ts_rel[-1]]
        def cif_at_rel(t_rel: torch.Tensor) -> torch.Tensor:
            u = t_rel / dt_grid
            j = torch.floor(u).long().clamp(min=0, max=Tw - 2)
            t_j = ts_rel[j]
            s = (t_rel - t_j).clamp(min=0.0, max=dt_grid)

            lam0_loc = lam[j]
            lam1_loc = lam[j + 1]
            dlam_loc = (lam1_loc - lam0_loc) / dt_grid

            inc = lam0_loc * s + 0.5 * dlam_loc * s * s
            return cif[j] + inc

        # per-sample available cumulative intensity in window
        Lambda0 = cif_at_rel(t0_rel)
        Lambda1 = cif_at_rel(t1_rel)
        Lambda_win = (Lambda1 - Lambda0).clamp_min(0.0)

        if (Lambda_win < eps_lam).all():
            tau_zero = dt_b.clone()
            if sample_sequence:
                return [[] for _ in range(B)]
            return tau_zero

        if not sample_sequence:
            E = -torch.log(torch.rand(B, device=device, dtype=t_dtype))
            tau = dt_b.clone()

            has_event = E < Lambda_win
            if not has_event.any():
                return tau

            target = (Lambda0[has_event] + E[has_event]).clamp_min(0.0)

            # Comment in English.
            cif_end = cif[-1]
            target = target.clamp_max((cif_end - eps_lam).clamp_min(0.0))

            t_event_rel = invert_cif_targets(target, cif, lam, dt_grid, ts_rel)
            tau_event = (t_event_rel - t0_rel[has_event]).clamp_min(0.0)
            tau[has_event] = torch.minimum(tau_event, dt_b[has_event])

            return tau

        # Comment in English.
        tau = dt_b.clone()
        cif_end = cif[-1]

        total_L = Lambda_win.clamp_min(0.0)  # Comment in English.
        n_events = torch.poisson(total_L).long()
        n_events = torch.where(total_L < eps_lam, torch.zeros_like(n_events), n_events)

        max_events = int(n_events.max().item())
        if max_events == 0:
            return [[] for _ in range(B)]

        event_idx = torch.arange(max_events, device=device).unsqueeze(0)
        event_mask = event_idx < n_events.unsqueeze(1)

        U = torch.rand(B, max_events, device=device, dtype=t_dtype)  # Comment in English.
        U = torch.where(event_mask, U, torch.ones_like(U))
        U_sorted, _ = torch.sort(U, dim=1)
        U_sorted = torch.where(event_mask, U_sorted, torch.zeros_like(U_sorted))

        targets = Lambda0.unsqueeze(1) + U_sorted * total_L.unsqueeze(1)
        max_target = (cif_end - eps_lam).clamp_min(0.0)
        targets = targets.clamp_max(max_target)
        targets = torch.where(event_mask, targets, torch.zeros_like(targets))

        flat_targets = targets[event_mask]  # Comment in English.
        if flat_targets.numel() == 0:
            return [[] for _ in range(B)]

        batch_ids = torch.repeat_interleave(torch.arange(B, device=device), n_events)  # Comment in English.
        t_events_rel_flat = invert_cif_targets(flat_targets, cif, lam, dt_grid, ts_rel)

        t0_flat = t0_rel[batch_ids]
        t1_flat = t1_rel[batch_ids]
        within = (t_events_rel_flat >= t0_flat) & (t_events_rel_flat <= t1_flat)
        if not within.any():
            return [[] for _ in range(B)]

        t_events_rel_flat = t_events_rel_flat[within]
        batch_ids = batch_ids[within]

        counts_filtered = torch.bincount(batch_ids, minlength=B)  # Comment in English.
        if counts_filtered.sum().item() == 0:
            return [[] for _ in range(B)]


        counts_cpu = counts_filtered.cpu().tolist()
        segments = torch.split(t_events_rel_flat, counts_cpu)
        times_list = [
            (seg + t_shift).to(torch.float32).cpu().tolist() if seg.numel() > 0 else []
            for seg in segments
            ]
        return times_list
