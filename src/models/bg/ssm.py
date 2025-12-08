import torch
from src.utils.interp import interp_uniform_time_series, integrate_uniform_time_series
from src.data.dot_dict import DotDict
from src.models.mamba.scan_wrapper import SelectiveScanWrapper
from .base import BGModel


@BGModel.register("ssm")
class SSMBGModel(BGModel):
    """
    State-space model based background generator.
    Uses a SelectiveScanWrapper to produce per-timepoint features which are
    projected to a scalar intensity via a positive weight vector.
    """

    def __init__(self, d_feature,d_state, device=None, log_scale_init=200.0):
        super().__init__()
        self.ts_batch_cache = None
        # raw (unconstrained) weight of shape (1, F)
        self.raw_weight = torch.nn.Parameter(torch.zeros(1, d_feature))
        # small SSM: d_state fixed here (same as original)
        self.ssm = SelectiveScanWrapper(d_model=d_feature, d_state=d_state, D_positive=True, device=device)
        self.device = device
        if device is not None:
            self.to(device)

    def positive_weight(self) -> torch.Tensor:
        """Return strictly positive weights via softplus(raw_weight), shape (1, F)."""
        return torch.nn.functional.softplus(self.raw_weight)

    # -------------------------------------------------------------
    # intensity(t)
    # -------------------------------------------------------------
    def intensity(self, ts_batch: DotDict, t_query=None, return_ssm_out=False) -> torch.Tensor:
        """
        Compute intensity lambda(t) for given batch and query times.

        Args:
            ts_batch: DotDict with keys:
                - time_series: (B, T, F)
                - time_series_times: (B, T)
                - optionally arrival_times: (B, Nq)
            t_query: optional query times (absolute) of shape (B, Nq) or (Nq,) or None.
                     If None, will use ts_batch.arrival_times if present, otherwise time_series_times.

        Returns:
            Tensor of shape (B, Nq) with nonnegative intensities.
        """
        w = self.positive_weight()  # (1, F)
        ts = ts_batch.time_series  # (B, T, F)

        # compute deltas for scan wrapper (per-timepoint deltas, broadcast to F)
        delta = ts_batch.time_series_times[:, 1:] - ts_batch.time_series_times[:, :-1]
        delta = torch.cat([ts_batch.time_series_times[:, :1] * 0.0, delta], dim=1)
        delta = delta.unsqueeze(-1).expand_as(ts)  # (B, T, F)

        # run SSM and ensure positivity
        ssm_out = self.ssm(ts, delta)  # (B, T, F)
        y = torch.nn.functional.softplus(ssm_out)
        y= y*ts
 
   