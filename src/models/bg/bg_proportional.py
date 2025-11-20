import torch
from src.utils.interp import interp_uniform_time_series, integrate_uniform_time_series
class ProportionalBGModel(torch.nn.Module):
    def __init__(self, d_feature):
        super(ProportionalBGModel, self).__init__()
        self.fc = torch.nn.Linear(d_feature, 1, bias=False)

    def intensity(self, batch):
        # Apply Softplus to make sure the weights are positive
        positive_weights = torch.nn.functional.softplus(self.fc.weight)
        self.fc.weight.data = positive_weights

        feature = interp_uniform_time_series(
            t=batch.time_series_times,      # (B, T)
            x=batch.time_series,            # (B, T, F)
            t_query=batch.arrival_times,    # (B, Nq)
        )
        intensity = self.fc(feature).squeeze(-1)
        return intensity

    def intensity_integral(self, batch):
        positive_weights = torch.nn.functional.softplus(self.fc.weight)
        self.fc.weight.data = positive_weights

        integral = integrate_uniform_time_series(
            t=batch.time_series_times,    
            x=batch.time_series,          
            t_start=batch.t_nll_start,  
            t_end=batch.t_end,     
        )
        # Avoid in-place parameter modification; apply Softplus to weights at computation time
        positive_weights = torch.nn.functional.softplus(self.fc.weight)  # (1, F)

        # Use functional linear so gradients flow to the original weights
        out = torch.nn.functional.linear(integral, positive_weights)  # (..., 1)

        # Normalize to shape (B,)
        intensity_integral = out.reshape(out.shape[0])
        return intensity_integral

    def nll_change(self, batch, log_h_intensity):
        f_intensity = self.intensity(batch)
        f_intensity_integral = self.intensity_integral(batch)
        h_intensity = torch.exp(log_h_intensity)
        ratio = f_intensity / h_intensity.clamp_min(1e-8)
        log_change = torch.log1p(ratio)*batch.nll_event_mask
        integral_change = f_intensity_integral
        log_like_change= log_change.sum(dim=1) - integral_change
        # print(log_change.sum(dim=1))
        # print(integral_change)
        # print(batch.t_end - batch.t_nll_start)
        return -log_like_change / (batch.t_end - batch.t_nll_start)