import abc
import torch
from src.utils.registrable import Registrable

class BGModel(torch.nn.Module, abc.ABC, Registrable):
    """Abstract base class for background models.

    Subclasses should implement the public API used by the rest of the codebase:
    - `positive_weight()` (if applicable)
    - `intensity(ts_batch, t_query=None)` -> Tensor (B, Nq) or (B,)
    - `intensity_integral(batch)` -> Tensor (B,)
    - `nll_change(batch, log_h_intensity)` -> Tensor (B,)
    - `cache_batch(time_series, time_series_times)`
    - `sample_nhpp(B, t0, t1, ...)`
    """

    @abc.abstractmethod
    def positive_weight(self):
        raise NotImplementedError

    @abc.abstractmethod
    def intensity(self, ts_batch, t_query=None):
        raise NotImplementedError

    @abc.abstractmethod
    def intensity_integral(self, batch):
        raise NotImplementedError

    @abc.abstractmethod
    def nll_change(self, batch, log_h_intensity):
        raise NotImplementedError

    @abc.abstractmethod
    def cache_batch(self, time_series, time_series_times):
        raise NotImplementedError

    @abc.abstractmethod
    def sample_nhpp(self, B, t0, t1, n_grid=10, return_times_list=False):
        raise NotImplementedError
