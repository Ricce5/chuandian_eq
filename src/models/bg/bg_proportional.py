import torch
from src.utils.interp import interp_uniform_time_series, integrate_uniform_time_series
from src.data.dot_dict import DotDict

class ProportionalBGModel(torch.nn.Module):
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
        assert time_series.shape[0] == time_series_times.shape[0]==1, "Batch size mismatch."
        self.ts_batch_cache = DotDict({
            "time_series": time_series,
            "time_series_times": time_series_times,
        }).to(self.device)
    
    def sample_nhpp(self, B, t0, t1, n_grid=10,return_times_list=False):
        # Parallel thinning-based sampling for B independent processes using the cached time-series.
        cached = getattr(self, "ts_batch_cache", None)
        assert cached is not None, "Batch data must be cached before sampling."

        device = self.device

        with torch.no_grad():
            # estimate an upper bound for intensity using a coarse grid
            t_query = torch.linspace(t0, t1, n_grid, device=device).unsqueeze(0)  # (1, n_grid)
            lam = self.intensity(cached, t_query=t_query)  # (1, n_grid)
            lambda_max = lam.max().item() * 1.2
            lambda_max = max(lambda_max, 1e-8)

            # replicate cached time-series to shape (B, T, F) / (B, T)
            ts_rep = DotDict({
                "time_series": cached.time_series.repeat(B, 1, 1),
                "time_series_times": cached.time_series_times.repeat(B, 1),
            }).to(device)

            # state for each parallel sampler
            times_list = [[] for _ in range(B)]
            t = torch.full((B,), float(t0), device=device)

            # thinning loop: propose exponential waiting times and accept/reject per process
            while True:
                active = t < t1
                if not active.any():
                    break

                # propose waiting times from Exp(lambda_max)
                w = -torch.log(torch.rand(B, device=device)) / lambda_max
                t = t + w

                # only consider proposals inside (t0, t1)
                inside = t < t1
                if not inside.any():
                    break

                # evaluate intensity at proposed times for all B
                t_q = t.unsqueeze(1)  # (B,1)
                lam_t = self.intensity(ts_rep, t_query=t_q).squeeze(-1)  # (B,)

                accept_prob = (lam_t / lambda_max).clamp(max=1.0)
                u = torch.rand(B, device=device)
                accept = (u < accept_prob) & inside

                # append accepted times per sampler
                accepted_idx = torch.nonzero(accept, as_tuple=False)
                if accepted_idx.numel() > 0:
                    for idx in accepted_idx.squeeze(1).tolist():
                        times_list[idx].append(t[idx].item())
            out = [ts[0] if len(ts) > 0 else t1 for ts in times_list]
            out = torch.tensor(out, device=device)  
        if return_times_list:
            return times_list, out
        return out
        
