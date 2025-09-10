import numpy as np
from csep.utils.stats import get_quantiles, cumulative_square_diff
from csep.models import (
    CatalogNumberTestResult,
    CatalogMagnitudeTestResult,
)

def magnitude_test_from_counts(
    forecast_catalogs,
    observed_catalog,
    get_counts=lambda cat: cat.magnitude_counts(),
    verbose=True,
):
    obs_hist = observed_catalog.magnitude_counts()
    obs_hist = np.asarray(obs_hist, dtype=float)
    n_obs = float(np.sum(obs_hist))

    if n_obs == 0:
        if verbose:
            print("Cannot perform magnitude test when observed event count is zero.")
        return {
            "test_distribution": [],
            "obs_d_statistic": None,
            "scaled_union_histogram": np.zeros_like(obs_hist),
        }

    forecast_counts = []
    for i, cat in enumerate(forecast_catalogs):
        counts = np.asarray(get_counts(cat), dtype=float)
        if counts.shape != obs_hist.shape:
            raise ValueError("magnitude_counts() binning mismatch between forecast and observed.")
        forecast_counts.append(counts)
        if verbose and (i + 1) % 100 == 0:
            print(f"Collected {i+1} forecast histograms")

    if len(forecast_counts) == 0:
        raise ValueError("No forecast catalogs provided.")

    F = np.stack(forecast_counts, axis=0)  # (J, nbins)

    union_hist = np.mean(F, axis=0)  # 按目录对每个震级 bin 取平均计数
    n_union = float(np.sum(union_hist))

    if n_union == 0:
        if verbose:
            print("Average forecast magnitude histogram sums to zero. Cannot perform M-test.")
        return {
            "test_distribution": [],
            "obs_d_statistic": None,
            "scaled_union_histogram": union_hist,  # 全零
        }

    scaled_union_hist = union_hist * (n_obs / n_union)
    test_distribution = []
    for j, counts in enumerate(forecast_counts):
        n_events = float(np.sum(counts))
        if n_events == 0:
            continue  # 与原实现一致：跳过 0 事件的目录
        scale = n_obs / n_events
        catalog_hist = counts * scale
        stat = cumulative_square_diff(
            np.log10(catalog_hist + 1.0),
            np.log10(scaled_union_hist + 1.0)
        )
        test_distribution.append(stat)
        if verbose and (j + 1) % 100 == 0:
            print(f"Processed {j+1} catalogs")
    obs_d_stat = cumulative_square_diff(
        np.log10(obs_hist + 1.0),
        np.log10(scaled_union_hist + 1.0)
    )

    # 可选：分位数

    return {
        "test_distribution": test_distribution,
        "obs_d_statistic": float(obs_d_stat),
        "scaled_union_histogram": scaled_union_hist,
    }

def run_magnitude_test_result(
    result_dict,
    Mc,
    obs_catalog_repr='obs',
    obs_name='catalog',
    sim_name='forecast',
    plot=True
):
    delta_1, delta_2 = get_quantiles(
        result_dict['test_distribution'],
        result_dict['obs_d_statistic']
    )
    result = CatalogMagnitudeTestResult(
        test_distribution=result_dict['test_distribution'],
        name='M-Test',
        observed_statistic=result_dict['obs_d_statistic'],
        quantile=(delta_1, delta_2),
        status='normal',
        min_mw=Mc,
        obs_catalog_repr=obs_catalog_repr,
        obs_name=obs_name,
        sim_name=sim_name
    )
    if plot:
        result.plot()
    return result


def run_number_test_result(
    event_counts,
    obs_count,
    obs_catalog_repr="obs",
    sim_name="Poisson (λ=20)",
    min_mw=2.0,
    obs_name="AZDX",
    plot=True
):
    delta_1, delta_2 = get_quantiles(event_counts, obs_count)
    result = CatalogNumberTestResult(
        test_distribution=event_counts,
        name='Catalog N-Test',
        observed_statistic=obs_count,
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