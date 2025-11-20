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
        intensity = self.fc(feature)
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
        intensity_integral = self.fc(integral).squeeze(-1)
        return intensity_integral

    def loglikelihood_change(self, batch, h_intensity):
        f_intensity = self.intensity(batch)
        f_intensity_integral = self.intensity_integral(batch)
        ratio = f_intensity / h_intensity.clamp_min(1e-8)
        log_change = torch.log1p(ratio)
        integral_change = f_intensity_integral
        loglikelihood = log_change.sum(dim=1) - integral_change
        return loglikelihood
