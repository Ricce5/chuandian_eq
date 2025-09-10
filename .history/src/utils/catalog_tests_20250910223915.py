import numpy as np
from typing import Iterable, Callable, Dict, Any, Optional, Tuple
from csep.utils.stats import get_quantiles, cumulative_square_diff
from csep.models import CatalogNumberTestResult, CatalogMagnitudeTestResult


def _safe_log10(x: np.ndarray) -> np.ndarray:
    """log10(x) with warnings silenced (you already add +1, so这只是稳妥起见)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(x)


def magnitude_test_from_counts_optimized(
    forecast_catalogs: Iterable[Any],
    observed_catalog: Any,
    get_counts: Callable[[Any], np.ndarray] = lambda cat: cat.magnitude_counts(),
    verbose: bool = True,
    return_quantiles: bool = False,
) -> Dict[str, Any]:
    """
    仅基于 catalog.magnitude_counts() 完成 M-test 的核心计算（无需 expected_rates）。
    使用**流式**方式计算平均预测直方图与 test_distribution，降低内存占用。

    Returns:
        {
          "test_distribution": List[float],
          "obs_d_statistic": float or None,
          "scaled_union_histogram": np.ndarray,
          "delta_1": float or None,
          "delta_2": float or None
        }
    """
    # ---- 观测直方图 ----
    obs_hist = np.asarray(observed_catalog.magnitude_counts(), dtype=float)
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

    # ---- 第一次遍历：在线求平均预测直方图（Welford-like 简化：逐个累加后再除以数量）----
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
            "scaled_union_histogram": union_hist,  # 全零
            "delta_1": None,
            "delta_2": None,
        }

    # 将平均预测缩放到观测总数
    scaled_union_hist = union_hist * (n_obs / n_union)

    # ---- 第二次遍历：计算每条预测目录的统计量（与 scaled_union_hist 比较）----
    # 重新遍历一次迭代器（如果传入的是只能消费一次的生成器，你可以先 list(...)）
    # 为兼容性，这里转成列表；若非常大可自行改造为在第一次遍历就缓存必要信息。
    if not isinstance(forecast_catalogs, (list, tuple)):
        forecast_catalogs = list(forecast_catalogs)

    test_distribution = []
    for j, cat in enumerate(forecast_catalogs):
        counts = np.asarray(get_counts(cat), dtype=float)
        n_events = float(np.sum(counts))
        if n_events == 0:
            continue  # 与原实现一致：跳过无事件目录
        scale = n_obs / n_events
        catalog_hist = counts * scale
        stat = cumulative_square_diff(
            _safe_log10(catalog_hist + 1.0),
            _safe_log10(scaled_union_hist + 1.0)
        )
        test_distribution.append(stat)
        if verbose and (j + 1) % 100 == 0:
            print(f"Processed {j+1} catalogs")

    # 观测统计量（观测 vs 缩放后的平均预测）
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
    """
    将 magnitude_test_from_counts_* 的返回字典封装为 CatalogMagnitudeTestResult，并可选择绘图。
    """
    test_distribution = result_dict.get('test_distribution', [])
    obs_stat = result_dict.get('obs_d_statistic', None)

    if obs_stat is None or len(test_distribution) == 0:
        # 与 pyCSEP 行为一致：没有有效统计量时标记 not-valid
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


def run_number_test_result(
    event_counts,
    obs_count,
    min_mw=2.0,
    obs_catalog_repr="obs",
    obs_name='catalog',
    sim_name='forecast',
    plot=True
):
    # 确保是 1D 序列（不强制转 float，保留整数更安全）
    event_counts = np.asarray(list(event_counts)).ravel()
    if event_counts.size == 0:
        result = CatalogNumberTestResult(
            test_distribution=[],
            name='Catalog N-Test',
            observed_statistic=int(obs_count),   # 保持为 int
            quantile=(None, None),
            status='not-valid',
            obs_catalog_repr=obs_catalog_repr,
            sim_name=sim_name,
            min_mw=min_mw,
            obs_name=obs_name,
        )
        if plot:
            result.plot()
        return result

    # 计算分位数
    delta_1, delta_2 = get_quantiles(event_counts, int(obs_count))

    result = CatalogNumberTestResult(
        test_distribution=event_counts.tolist(),
        name='Catalog N-Test',
        observed_statistic=int(obs_count),      # 关键：不要用 float
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
