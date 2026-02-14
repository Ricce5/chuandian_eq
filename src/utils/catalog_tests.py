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
        ax = plots.plot_number_test(self, axes=ax, show=show, plot_args=defaults)
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
        ax = plots.plot_magnitude_test(self, axes=ax, show=show, plot_args=defaults)
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


from typing import Any, Callable, Dict, Iterable, List, Optional
import numpy as np

def magnitude_test_from_counts(
    forecast_catalogs: Iterable[Any],
    observed_catalog: Any,
    get_counts: Callable[[Any], np.ndarray] = lambda cat: cat.magnitude_counts(),
    verbose: bool = True,
    return_quantiles: bool = False,
    debug: bool = True,                 # 开关：是否打印调试信息
    debug_every: int = 100,             # 每隔多少个catalog打印一次进度与统计
    print_bins: int = 8,                # 打印前N个bin的值(0表示不打印)
    name_of: Callable[[Any], str] = lambda cat: getattr(cat, "name", repr(cat)),
) -> Dict[str, Any]:
    """
    与原逻辑一致，但增加大量中间值检查与打印，用于定位 test_distribution 出现 NaN 的原因。
    """

    def _stats(arr: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(arr, dtype=float)
        finite = np.isfinite(arr)
        return {
            "shape": arr.shape,
            "sum": float(np.nansum(arr)),
            "min": float(np.nanmin(arr)) if arr.size else np.nan,
            "max": float(np.nanmax(arr)) if arr.size else np.nan,
            "finite_ratio": float(np.mean(finite)) if arr.size else 0.0,
            "nan_count": int(np.isnan(arr).sum()),
            "inf_count": int(np.isinf(arr).sum()),
            "neg_count": int((arr < 0).sum()),
            "zero_count": int((arr == 0).sum()),
        }

    def _print_arr_head(tag: str, arr: np.ndarray):
        if print_bins and arr.size:
            head = np.asarray(arr).ravel()[:print_bins]
            print(f"[DEBUG] {tag} head({print_bins}) =", head)

    def _check_and_print(tag: str, arr: np.ndarray):
        if not debug:
            return
        s = _stats(arr)
        print(f"[DEBUG] {tag} stats:", s)
        _print_arr_head(tag, arr)

    # --- Observed ---
    obs_hist = np.asarray(get_counts(observed_catalog), dtype=float)
    if debug:
        print("[DEBUG] observed_catalog =", name_of(observed_catalog))
        _check_and_print("obs_hist", obs_hist)

    n_obs = float(np.sum(obs_hist))
    if not np.isfinite(n_obs) or n_obs < 0:
        print(f"[DEBUG][WARN] n_obs abnormal: n_obs={n_obs}")
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

    # --- First pass: sum forecasts ---
    # 注意：这里直接遍历 forecast_catalogs 会“消耗”生成器
    if not isinstance(forecast_catalogs, (list, tuple)):
        forecast_catalogs = list(forecast_catalogs)

    sum_hist = None
    n_forecasts = 0

    for i, cat in enumerate(forecast_catalogs):
        counts = np.asarray(get_counts(cat), dtype=float)

        if counts.shape != obs_hist.shape:
            raise ValueError(
                f"magnitude_counts() binning mismatch: forecast bins {counts.shape} vs observed {obs_hist.shape}"
            )

        if debug and (i < 3):  # 前3个先详细打印一下
            print(f"\n[DEBUG] forecast[{i}] =", name_of(cat))
            _check_and_print(f"forecast[{i}].counts", counts)

        # 关键：检查 counts 是否已经包含 NaN/Inf/负值
        if debug:
            if np.isnan(counts).any() or np.isinf(counts).any() or (counts < 0).any():
                print(f"[DEBUG][WARN] forecast[{i}] counts has NaN/Inf/negatives!")

        sum_hist = counts if sum_hist is None else (sum_hist + counts)
        n_forecasts += 1

        if verbose and (i + 1) % debug_every == 0:
            print(f"Collected {i+1} forecast histograms")
        if debug and (i + 1) % debug_every == 0:
            _check_and_print("sum_hist (running)", sum_hist)

    if n_forecasts == 0:
        raise ValueError("No forecast catalogs provided.")

    union_hist = sum_hist / n_forecasts
    if debug:
        print("\n[DEBUG] After first pass:")
        _check_and_print("union_hist (mean forecast)", union_hist)

    n_union = float(np.sum(union_hist))
    if debug and (not np.isfinite(n_union) or n_union <= 0):
        print(f"[DEBUG][WARN] n_union abnormal: n_union={n_union}")

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

    # --- Scale union to observed total ---
    scale_union = n_obs / n_union
    scaled_union_hist = union_hist * scale_union
    if debug:
        print(f"\n[DEBUG] scale_union = n_obs/n_union = {n_obs}/{n_union} = {scale_union}")
        _check_and_print("scaled_union_hist", scaled_union_hist)

    # 检查：scaled_union_hist 是否出现 NaN/Inf
    if debug and (np.isnan(scaled_union_hist).any() or np.isinf(scaled_union_hist).any()):
        print("[DEBUG][WARN] scaled_union_hist has NaN/Inf. Likely n_union is 0 or union_hist had NaN/Inf.")

    # --- Second pass: compute test_distribution ---
    test_distribution: List[float] = []

    for j, cat in enumerate(forecast_catalogs):
        counts = np.asarray(get_counts(cat), dtype=float)
        n_events = float(np.sum(counts))

        if not np.isfinite(n_events) or n_events < 0:
            if debug:
                print(f"[DEBUG][WARN] forecast[{j}] n_events abnormal: {n_events} ({name_of(cat)})")

        if n_events == 0:
            if debug and j < 5:
                print(f"[DEBUG] forecast[{j}] skipped (n_events=0)")
            continue

        scale = n_obs / n_events
        catalog_hist = counts * scale

        if debug and j < 3:
            print(f"\n[DEBUG] Second pass forecast[{j}] = {name_of(cat)}")
            print(f"[DEBUG] n_events={n_events}, scale={scale}")
            _check_and_print(f"catalog_hist[{j}]", catalog_hist)

        # 重点：检查 log10 输入是否可能异常（负数会炸）
        # 你这里加了 +1.0，所以只要 catalog_hist / scaled_union_hist 不含 < -1 的值就不会出现 log10(<=0)
        if debug:
            if (catalog_hist + 1.0 <= 0).any():
                print(f"[DEBUG][WARN] catalog_hist[{j}] has values <= -1 -> log10 input <=0!")
            if (scaled_union_hist + 1.0 <= 0).any():
                print(f"[DEBUG][WARN] scaled_union_hist has values <= -1 -> log10 input <=0!")

        a = _safe_log10(catalog_hist + 1.0)
        b = _safe_log10(scaled_union_hist + 1.0)
        # print(f"catalog_hist_sim {catalog_hist.sum()}, scaled_union_hist sum {scaled_union_hist.sum()}")

        if debug and j < 3:
            _check_and_print(f"logA[{j}] = _safe_log10(catalog_hist+1)", a)
            _check_and_print("logB = _safe_log10(scaled_union_hist+1)", b)

        stat = cumulative_square_diff(a, b)

        if debug:
            if (not np.isfinite(stat)) or np.isnan(stat):
                print(f"[DEBUG][WARN] stat is NaN/Inf at forecast[{j}] ({name_of(cat)}) -> stat={stat}")
                # 额外：打印更具体的异常定位
                print("[DEBUG] a finite_ratio:", np.mean(np.isfinite(a)))
                print("[DEBUG] b finite_ratio:", np.mean(np.isfinite(b)))
                # 若 cumulative_square_diff 内部可能用到了除法/归一化，这里可进一步打印 a,b 的 min/max
                print("[DEBUG] a min/max:", np.nanmin(a), np.nanmax(a))
                print("[DEBUG] b min/max:", np.nanmin(b), np.nanmax(b))

        test_distribution.append(float(stat))

        if verbose and (j + 1) % debug_every == 0:
            print(f"Processed {j+1} catalogs")

    # --- observed statistic ---
    a_obs = _safe_log10(obs_hist + 1.0)
    b_union = _safe_log10(scaled_union_hist + 1.0)
    # print(f"obs_hist_sim {obs_hist.sum()}, scaled_union_hist sum {scaled_union_hist.sum()}")
    if debug:
        print("\n[DEBUG] Observed statistic inputs:")
        _check_and_print("logObs = _safe_log10(obs_hist+1)", a_obs)
        _check_and_print("logUnion = _safe_log10(scaled_union_hist+1)", b_union)

    obs_d_stat = cumulative_square_diff(a_obs, b_union)

    if debug:
        print("[DEBUG] obs_d_stat =", obs_d_stat)
        if not np.isfinite(obs_d_stat):
            print("[DEBUG][WARN] obs_d_stat is NaN/Inf")

        # test_distribution 概览
        td = np.asarray(test_distribution, dtype=float)
        print("[DEBUG] test_distribution size =", len(test_distribution))
        if len(test_distribution) > 0:
            print("[DEBUG] test_distribution nan_count =", int(np.isnan(td).sum()))
            print("[DEBUG] test_distribution inf_count =", int(np.isinf(td).sum()))
            print("[DEBUG] test_distribution min/max =", np.nanmin(td), np.nanmax(td))

    delta_1 = delta_2 = None
    if return_quantiles:
        delta_1, delta_2 = get_quantiles(test_distribution, obs_d_stat)

    return {
        "test_distribution": test_distribution,
        "obs_d_statistic": float(obs_d_stat) if np.isfinite(obs_d_stat) else float("nan"),
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
    plot: bool = True,
    plot_args: Optional[Dict[str, Any]] = None
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
            result.plot(**plot_args if plot_args else {})
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
        result.plot(**plot_args if plot_args else {})
    return result




