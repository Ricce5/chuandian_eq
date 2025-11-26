import abc
import torch
from src.utils.interp import interp_uniform_time_series, integrate_uniform_time_series
from src.data.dot_dict import DotDict
from .base import BGModel

@BGModel.register("proportional")
class ProportionalBGModel(BGModel):
    def __init__(self, d_feature, device):
        super().__init__()
        
        # raw weight, unconstrained
        self.raw_weight = torch.nn.Parameter(torch.zeros(1, d_feature))
        self.ts_batch_cache = None
        self.device = device
        if device is not None:
            self.to(device)

    def positive_weight(self):
        """Return positive weights via softplus(raw_weight)."""
        return torch.nn.functional.softplus(self.raw_weight)

    # -------------------------------------------------------------
    # intensity(t)
    # -------------------------------------------------------------
    def intensity(self, ts_batch, t_query=None):
        w = self.positive_weight()  # (1, F)

        feature = interp_uniform_time_series(
            t=ts_batch.time_series_times,      # (B, T)
            x=ts_batch.time_series,            # (B, T, F)
            t_query=t_query if t_query is not None else ts_batch.arrival_times,
        )                                       # (B, Nq, F)
        intensity = torch.nn.functional.linear(feature, w).squeeze(-1)
        return intensity

    # -------------------------------------------------------------
    # ∫ λ(t) dt
    # -------------------------------------------------------------
    def intensity_integral(self, batch):
        w = self.positive_weight()  # (1, F)q
        integral = integrate_uniform_time_series(
            t=batch.time_series_times,    
            x=batch.time_series,          
            t_start=batch.t_nll_start,  
            t_end=batch.t_end,
        )                                # (B, F)
        out = torch.nn.functional.linear(integral, w)  # (B, 1)
        return out.squeeze(-1)

    def nll_change(self, batch, log_h_intensity):
        f_intensity = self.intensity(batch)
        f_intensity_integral = self.intensity_integral(batch)
        h_intensity = torch.exp(log_h_intensity)
        ratio = f_intensity / h_intensity.clamp_min(1e-8)
        log_change = torch.log1p(ratio)*batch.nll_event_mask
        integral_change = f_intensity_integral
        log_like_change= log_change.sum(dim=1) - integral_change
        return -log_like_change
    
    def cache_batch(self, time_series, time_series_times):
        assert time_series.shape[0] == time_series_times.shape[0]==1, "Batch size should be 1."
        self.ts_batch_cache = DotDict({
            "time_series": time_series,
            "time_series_times": time_series_times,
        }).to(self.device)
    
    def sample_nhpp(self, B, t0, dt, n_grid=10, return_times_list=False):
        """
        Parallel sampling of the first-event waiting time for B NHPPs (relative to t0).

        Args:
            B  : number of parallel samples
            t0 : start absolute time, can be a scalar / length-B vector / tensor
            dt : interval length (t1 - t0), can be scalar / length-B vector / tensor
        Returns:
            out: tensor of shape (B,), the i-th element is the time from t0 to the first event;
                 if there is no event in [t0, t0+dt], returns dt[i].
            If return_times_list=True, also returns times_list (each is [tau] or []).
        """
        cached = getattr(self, "ts_batch_cache", None)
        assert cached is not None, "Batch data must be cached before sampling."

        device = self.device

        def to_batched_tensor(x, name):
            # Convert scalar/sequence/tensor x to a tensor of shape (B,) on device
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
                # assume number or list/ndarray
                try:
                    x_t = torch.tensor(x, device=device, dtype=torch.float)
                except Exception:
                    x_t = torch.tensor(float(x), device=device)
                if x_t.dim() == 0 or x_t.numel() == 1:
                    return torch.full((B,), float(x_t.item()), device=device)
                else:
                    assert x_t.shape[0] == B, f"{name} must have length B"
                    return x_t.reshape(B)

        with torch.no_grad():
            # absolute start t0_b and relative interval length dt_b
            t0_b = to_batched_tensor(t0, "t0")   # (B,)
            dt_b = to_batched_tensor(dt, "dt")   # (B,)

            # still use absolute time to compute intensity
            ts_rep = DotDict({
                "time_series": cached.time_series,
                "time_series_times": cached.time_series_times,
            }).to(device)

            # ------- estimate lambda_max (using absolute time) -------
            u = torch.linspace(0.0, 1.0, n_grid, device=device).unsqueeze(0)  # (1, n_grid)
            t_query_rel = dt_b.unsqueeze(1) * u                               # (B, n_grid) relative times
            t_query_abs = t0_b.unsqueeze(1) + t_query_rel                     # (B, n_grid) absolute times
            lam = self.intensity(ts_rep, t_query=t_query_abs)                 # (B, n_grid)
            lambda_max = lam.max().item() * 1.2
            lambda_max = max(lambda_max, 1e-8)

            # ------- state: relative time τ, starting from 0 -------
            times_list = [[] for _ in range(B)]   # stores waiting times tau
            tau = torch.zeros(B, device=device)   # (B,) relative time
            has_event = torch.zeros(B, dtype=torch.bool, device=device)

            while True:
                # samples that have no event yet and tau < dt
                active = (~has_event) & (tau < dt_b)
                if not active.any():
                    break

                # sample waiting times only for active samples
                w = torch.zeros(B, device=device)
                num_active = active.sum()
                w_active = -torch.log(torch.rand(num_active, device=device)) / lambda_max
                w[active] = w_active

                # update on relative time axis
                tau = tau + w

                # samples still inside interval and without event
                inside = (~has_event) & (tau < dt_b)
                if not inside.any():
                    break

                # compute intensity at proposal times (using absolute time)
                t_q_abs = t0_b[inside] + tau[inside]          # absolute times
                t_q = t_q_abs.unsqueeze(1)                    # (N_c, 1)
                lam_t = self.intensity(ts_rep, t_query=t_q).squeeze(-1)  # (N_c,)

                # thinning acceptance
                accept_prob = (lam_t / lambda_max).clamp(max=1.0)
                u2 = torch.rand(accept_prob.shape[0], device=device)
                accept_local = u2 < accept_prob

                if accept_local.any():
                    inside_idx = torch.nonzero(inside, as_tuple=False).squeeze(1)
                    accepted_idx = inside_idx[accept_local]

                    for idx in accepted_idx.tolist():
                        if not has_event[idx]:
                            times_list[idx].append(tau[idx].item())  # store relative time tau
                            has_event[idx] = True

            # ------- construct output: waiting time tau or dt -------
            out_vals = []
            for i in range(B):
                if len(times_list[i]) > 0:
                    # this is relative time, so just check it's within [0, dt]
                    tau_i = times_list[i][0]
                    assert 0.0 <= tau_i <= dt_b[i].item() + 1e-6, \
                        f"Relative first event time {tau_i} not in [0, {dt_b[i].item()}]"
                    out_vals.append(tau_i)
                else:
                    # no events, return the full interval length dt
                    out_vals.append(float(dt_b[i].item()))
            out = torch.tensor(out_vals, device=device)

        if return_times_list:
            return times_list, out
        return out
