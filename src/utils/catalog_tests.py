# Adapted from pycsep
# Reference: https://github.com/SCECcode/pycsep/blob/main/csep/models.py
import numpy as np
from typing import Iterable, Callable, Dict, Any, Optional, Tuple
from csep.utils.stats import get_quantiles, cumulative_square_diff
from csep.models import  EvaluationResult     # CatalogNumberTestResult, CatalogMagnitudeTestResult
import matplotlib.pyplot as plt
from csep.utils import plots
from csep.utils.plots import plot_histogram

def _safe_log10(x: np.ndarray) -> np.ndarray:
    """Compute log10(x) with warnings suppressed (adding +1 ensures stability)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(x)


class CatalogNumberTestResult(EvaluationResult):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, ax=None, show=False, plot_args=None):
        plot_args = dict(plot_args or {})
        td = np.asarray(self.test_distribution)

        if td.size == 0:
            if ax is None:
                _, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            if show:
                plt.show()
            return ax

        vmin, vmax = int(np.min(td)), int(np.max(td))
        if vmin == vmax:
            bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)  
        else:
            bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)

        defaults = {
            'percentile': 95,
            'title': 'Number Test',
            'xlabel': 'Event count in catalog',
            'bins': bins
        }
        defaults.update(plot_args)  
        ax = plots.plot_number_test(self, show=show, plot_args=defaults)
        return ax

class CatalogMagnitudeTestResult(EvaluationResult):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, ax=None, show=False, plot_args=None):
        plot_args = dict(plot_args or {})

        defaults = {
            'percentile': 95,
            'title': 'Magnitude Test',
            'bins': 'auto'
        }
        defaults.update(plot_args) 
        ax = plots.plot_magnitude_test(self, show=show, plot_args=defaults)
        return ax



def run_number_test_result(
    event_counts,
    obs_count,
    min_mw=2.0,
    obs_catalog_repr="obs",
    obs_name='observed',
    sim_name='Simulated',
    plot_args=None,
    plot=True
):
    event_counts = np.asarray(list(event_counts)).ravel()
    if event_counts.size == 0:
        result = CatalogNumberTestResult(
            test_distribution=[],
            name='Catalog N-Test',
            observed_statistic=int(obs_count),  
            quantile=(None, None),
            status='not-valid',
            obs_catalog_repr=obs_catalog_repr,
            sim_name=sim_name,
            min_mw=min_mw,
            obs_name=obs_name,
        )
        if plot:
            result.plot(plot_args=plot_args)
        return result

    delta_1, delta_2 = get_quantiles(event_counts, int(obs_count))

    result = CatalogNumberTestResult(
        test_distribution=event_counts.tolist(),
        name='Catalog N-Test',
        observed_statistic=int(obs_count),    
        quantile=(delta_1, delta_2),
        status='normal',
        obs_catalog_repr=obs_catalog_repr,
        sim_name=sim_name,
        min_mw=min_mw,
        obs_name=obs_name,
    )
    if plot:
        result.plot()
    return result

def compute_magnitude_distribution(mag, min_mw, max_mw=10, dmw=0.1):
    """
    Compute the magnitude distribution for a given forecast.

    Args:
        mag (array-like): Magnitudes to compute the distribution for.
        min_mw (float): Minimum magnitude to consider.
        max_mw (float): Maximum magnitude to consider.
        dmw (float): Magnitude bin width.

    Returns:
        tuple: A tuple containing the magnitude bins and the computed distribution.
    """
    from csep.utils.calc import bin1d_vec
    from csep.core import regions

    # Generate magnitude bins
    magnitude_bins = regions.magnitude_bins(min_mw, max_mw, dmw)
    mag = np.asarray(mag)

    # Compute the histogram
    distribution = np.zeros(len(magnitude_bins))
    idx = bin1d_vec(mag, magnitude_bins, tol=None, right_continuous=True)
    np.add.at(distribution, idx, 1)

    return magnitude_bins, distribution


def magnitude_test_from_counts(
    forecast_catalogs: Iterable[Any],
    observed_catalog: Any,
    get_counts: Callable[[Any], np.ndarray] = lambda cat: cat.magnitude_counts(),
    verbose: bool = True,
    return_quantiles: bool = False,
) -> Dict[str, Any]:
    """
    Perform the core computation of the M-test based solely on catalog.magnitude_counts(),
    without requiring expected_rates. This implementation uses a **streaming** approach
    to calculate the average forecast histogram and test_distribution, reducing memory usage.

    Returns:
        {
          "test_distribution": List[float],
          "obs_d_statistic": float or None,
          "scaled_union_histogram": np.ndarray,
          "delta_1": float or None,
          "delta_2": float or None
        }
    """
    obs_hist = np.asarray(get_counts(observed_catalog), dtype=float)
    n_obs = float(np.sum(obs_hist))
    if n_obs == 0:
        if verbose:
            print("Cannot perform magnitude test when observed event count is zero.")
        return {
            "test_distribution": [],
            "obs_d_statistic": None,
            "scaled_union_histogram": np.zeros_like(obs_hist),
            "delta_1": None,
            "delta_2": None,
        }

    sum_hist = None
    n_forecasts = 0
    for i, cat in enumerate(forecast_catalogs):
        counts = np.asarray(get_counts(cat), dtype=float)
        if counts.shape != obs_hist.shape:
            raise ValueError(
                f"magnitude_counts() binning mismatch: forecast bins {counts.shape} vs observed {obs_hist.shape}"
            )
        sum_hist = counts if sum_hist is None else (sum_hist + counts)
        n_forecasts += 1
        if verbose and (i + 1) % 100 == 0:
            print(f"Collected {i+1} forecast histograms")

    if n_forecasts == 0:
        raise ValueError("No forecast catalogs provided.")

    union_hist = sum_hist / n_forecasts
    n_union = float(np.sum(union_hist))
    if n_union == 0:
        if verbose:
            print("Average forecast magnitude histogram sums to zero. Cannot perform M-test.")
        return {
            "test_distribution": [],
            "obs_d_statistic": None,
            "scaled_union_histogram": union_hist,  
            "delta_1": None,
            "delta_2": None,
        }

    # Scale the average forecast to match the observed total count
    scaled_union_hist = union_hist * (n_obs / n_union)

    # ---- Second Iteration: Calculate statistics for each forecast catalog (compared to scaled_union_hist) ----
    # Reiterate over the iterator (if a generator is passed, you can convert it to a list first).
    # For compatibility, convert it to a list here; if the data is very large, consider caching necessary information during the first iteration.
    if not isinstance(forecast_catalogs, (list, tuple)):
        forecast_catalogs = list(forecast_catalogs)

    test_distribution = []
    for j, cat in enumerate(forecast_catalogs):
        counts = np.asarray(get_counts(cat), dtype=float)
        n_events = float(np.sum(counts))
        if n_events == 0:
            continue
        scale = n_obs / n_events
        catalog_hist = counts * scale
        stat = cumulative_square_diff(
            _safe_log10(catalog_hist + 1.0),
            _safe_log10(scaled_union_hist + 1.0)
        )
        test_distribution.append(stat)
        if verbose and (j + 1) % 100 == 0:
            print(f"Processed {j+1} catalogs")

    obs_d_stat = cumulative_square_diff(
        _safe_log10(obs_hist + 1.0),
        _safe_log10(scaled_union_hist + 1.0)
    )

    delta_1 = delta_2 = None
    if return_quantiles:
        delta_1, delta_2 = get_quantiles(test_distribution, obs_d_stat)

    return {
        "test_distribution": test_distribution,
        "obs_d_statistic": float(obs_d_stat),
        "scaled_union_histogram": scaled_union_hist,
        "delta_1": delta_1,
        "delta_2": delta_2,
    }



def run_magnitude_test_result(
    result_dict: Dict[str, Any],
    min_mw: float = 2.0,
    obs_catalog_repr: str = 'obs',
    obs_name: str = 'catalog',
    sim_name: str = 'forecast',
    plot: bool = True
) -> CatalogMagnitudeTestResult:
    test_distribution = result_dict.get('test_distribution', [])
    obs_stat = result_dict.get('obs_d_statistic', None)

    if obs_stat is None or len(test_distribution) == 0:
        result = CatalogMagnitudeTestResult(
            test_distribution=test_distribution,
            name='M-Test',
            observed_statistic=obs_stat,
            quantile=(None, None),
            status='not-valid',
            min_mw=min_mw,
            obs_catalog_repr=obs_catalog_repr,
            obs_name=obs_name,
            sim_name=sim_name
        )
        if plot:
            result.plot()
        return result

    delta_1, delta_2 = get_quantiles(test_distribution, obs_stat)
    result = CatalogMagnitudeTestResult(
        test_distribution=test_distribution,
        name='M-Test',
        observed_statistic=obs_stat,
        quantile=(delta_1, delta_2),
        status='normal',
        min_mw=min_mw,
        obs_catalog_repr=obs_catalog_repr,
        obs_name=obs_name,
        sim_name=sim_name
    )
    if plot:
        result.plot()
    return result





