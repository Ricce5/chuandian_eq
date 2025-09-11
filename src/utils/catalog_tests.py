import numpy as np
from typing import Iterable, Callable, Dict, Any, Optional, Tuple
from csep.utils.stats import get_quantiles, cumulative_square_diff
from csep.models import  EvaluationResult     # CatalogNumberTestResult, CatalogMagnitudeTestResult
import matplotlib.pyplot as plt
from csep.utils import plots
from csep.utils.plots import plot_histogram

def _safe_log10(x: np.ndarray) -> np.ndarray:
    """log10(x) with warnings silenced (you already add +1, so这只是稳妥起见)."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.log10(x)




class CatalogMagnitudeTestResult(EvaluationResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, ax=None, show=False, plot_args=None):
        plot_args = dict(plot_args or {})

        # 默认绘图参数
        defaults = {
            'percentile': 95,
            'title': 'Magnitude Test',
            'bins': 'auto'
        }
        defaults.update(plot_args)  # 用户参数覆盖默认

        # 选择目标轴
        if ax is None:
            ax = plt.gca()

        # 把 ax 传给底层绘图函数（注意这里也要改 plot_magnitude_test 支持 ax 参数）
        ax = plot_magnitude_test(self, ax=ax, show=show, plot_args=defaults)
        return ax

class CatalogNumberTestResult(EvaluationResult):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def plot(self, ax=None, show=False, plot_args=None):
        plot_args = dict(plot_args or {})
        td = np.asarray(self.test_distribution)

        # 防御：空分布直接返回空轴
        if td.size == 0:
            if ax is None:
                _, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_axis_off()
            if show:
                plt.show()
            return ax

        # 计算更合理的整数直方图 bins（柱中心落在整数处）
        vmin, vmax = int(np.min(td)), int(np.max(td))
        # 让边界在 .5 处，这样整数计数的柱子对齐
        if vmin == vmax:
            bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)  # 至少一个柱
        else:
            bins = np.arange(vmin - 0.5, vmax + 1.5, 1.0)

        # 默认绘图参数
        defaults = {
            'percentile': 95,
            'title': 'Number Test',
            'xlabel': 'Event count in catalog',
            'bins': bins
        }
        defaults.update(plot_args)  # 用户参数覆盖默认

        # 选择目标轴
        if ax is None:
            ax = plt.gca()

        # 把 ax 传给底层绘图函数（需要 plots.plot_number_test 支持 ax=...）
        ax = plot_number_test(self, ax=ax, show=show, plot_args=defaults)
        return ax


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
    obs_name='oberved',
    sim_name='Simulated',
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


def plot_number_test(evaluation_result, axes=None, show=True, plot_args=None):
    """
    N-test 直方图绘制；兼容传入子图 axes。
    """
    # ==== 链式绘制判定 ====
    chained = axes is not None

    # ==== 读取参数 ====
    plot_args = dict(plot_args or {})
    title = plot_args.get('title', 'Number Test')
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', 'Event count of catalogs')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    ylabel = plot_args.get('ylabel', 'Number of catalogs')
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    text_fontsize = plot_args.get('text_fontsize', 14)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    filename = plot_args.get('filename', None)
    bins = plot_args.get('bins', 'auto')
    xy = plot_args.get('xy', (0.5, 0.3))
    legend = plot_args.get('legend', True)
    legend_loc = plot_args.get('legend_loc', 'best')

    # 颜色/外观（给默认值，可被覆盖）
    hist_color = plot_args.get('hist_color', 'C0')
    hist_alpha = plot_args.get('hist_alpha', 0.8)
    hist_edgecolor = plot_args.get('hist_edgecolor', 'none')  # 避免“全黑”
    obs_line_color = plot_args.get('obs_line_color', 'C0')
    ci_color = plot_args.get('ci_color', 'C0')  # 若 plot_histogram 会画 CI，可用

    # ==== 固定的标签（供图例使用）====
    # 你的 plot_histogram 若读取 obs_label/sim_label，这里统一提供
    fixed_plot_args = {
        'obs_label': evaluation_result.obs_name,
        'sim_label': evaluation_result.sim_name,
        # 也把外观参数传下去（如果 plot_histogram 支持，就能统一样式）
        'hist_color': hist_color,
        'hist_alpha': hist_alpha,
        'hist_edgecolor': hist_edgecolor,
        'obs_line_color': obs_line_color,
        'ci_color': ci_color,
        'legend': legend,           # 传下去以便底层添加 label
    }
    plot_args.update(fixed_plot_args)

    # ==== 选择/创建 Axes ====
    if chained:
        ax = axes
    else:
        figsize = plot_args.get('figsize', (6.4, 4.8))
        fig, ax = plt.subplots(figsize=figsize)

    # ==== 绘图 ====
    ax_ret = plot_histogram(
        evaluation_result.test_distribution,
        evaluation_result.observed_statistic,
        catalog=evaluation_result.obs_catalog_repr,
        plot_args=plot_args,
        bins=bins,
        axes=ax,                 # 显式传入 ax
        percentile=percentile
    )
    # 有些实现会返回新 ax（例如内部新建过）；优先用返回的
    if ax_ret is not None:
        ax = ax_ret

    # ==== 注释 ====
    if not chained:
        try:
            ax.annotate(
                '$\\delta_1 = P(X \\ge x) = {:.2f}$\n$\\delta_2 = P(X \\le x) = {:.2f}$\n$\\omega = {:d}$'
                .format(*evaluation_result.quantile, evaluation_result.observed_statistic),
                xycoords='axes fraction',
                xy=xy,
                fontsize=text_fontsize
            )
        except Exception:
            ax.annotate(
                '$\\gamma = P(X \\le x) = {:.2f}$\n$\\omega = {:.2f}$'
                .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                xycoords='axes fraction',
                xy=xy,
                fontsize=text_fontsize
            )

    # ==== 轴标题 ====
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    # ==== 图例 ====
    if legend:
        # 只有在有带 label 的艺术家时才会出现图例
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc)

    # ==== 布局 & 保存 & 显示 ====
    if tight_layout and ax.figure is not None:
        ax.figure.tight_layout()

    if filename is not None and ax.figure is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    if show and not chained:
        pyplot.show()

    return ax



def plot_number_test(evaluation_result, axes=None, ax=None, show=True, plot_args=None):
    """
    N-test 直方图绘制；兼容 ax / axes，两者任选其一。
    - 若未提供轴，则内部创建新 figure/axes。
    - 链式绘制（传入 ax/axes）时不自动 show()，便于在子图中组合绘制。
    """
    # --- 兼容参数名 ---
    if ax is not None and axes is None:
        axes = ax
    chained = axes is not None

    # --- 参数读取 ---
    plot_args = dict(plot_args or {})
    title = plot_args.get('title', 'Number Test')
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', 'Event count of catalogs')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    ylabel = plot_args.get('ylabel', 'Number of catalogs')
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    text_fontsize = plot_args.get('text_fontsize', 14)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    filename = plot_args.get('filename', None)
    bins = plot_args.get('bins', 'auto')
    xy = plot_args.get('xy', (0.5, 0.3))
    legend = plot_args.get('legend', True)
    legend_loc = plot_args.get('legend_loc', 'best')

    # 外观（可被 plot_args 覆盖；需 plot_histogram 使用这些键）
    plot_args.setdefault('hist_color', 'C0')
    plot_args.setdefault('hist_alpha', 0.8)
    plot_args.setdefault('hist_edgecolor', 'none')
    plot_args.setdefault('obs_line_color', 'C0')
    plot_args.setdefault('ci_color', 'C0')
    plot_args.setdefault('legend', legend)

    # 固定标签（供图例使用；需 plot_histogram 内部读取）
    plot_args.update({
        'obs_label': evaluation_result.obs_name,
        'sim_label': evaluation_result.sim_name,
    })

    # --- 选择/创建 Axes ---
    if axes is None:
        figsize = plot_args.get('figsize', (6.4, 4.8))
        fig, axes = plt.subplots(figsize=figsize)
    ax = axes

    # --- 绘制 ---
    ax_ret = plot_histogram(
        evaluation_result.test_distribution,
        evaluation_result.observed_statistic,
        catalog=evaluation_result.obs_catalog_repr,
        plot_args=plot_args,
        bins=bins,
        axes=ax,
        percentile=percentile
    )
    if ax_ret is not None:
        ax = ax_ret

    # --- 注释 ---
    if not chained:
        try:
            ax.annotate(
                '$\\delta_1 = P(X \\ge x) = {:.2f}$\n$\\delta_2 = P(X \\le x) = {:.2f}$\n$\\omega = {:d}$'
                .format(*evaluation_result.quantile, evaluation_result.observed_statistic),
                xycoords='axes fraction', xy=xy, fontsize=text_fontsize
            )
        except Exception:
            ax.annotate(
                '$\\gamma = P(X \\le x) = {:.2f}$\n$\\omega = {:.2f}$'
                .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                xycoords='axes fraction', xy=xy, fontsize=text_fontsize
            )

    # --- 轴标题 ---
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    # --- 图例 ---
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc)

    # --- 布局/保存/显示 ---
    if tight_layout and ax.figure is not None:
        ax.figure.tight_layout()

    if filename is not None and ax.figure is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    if show and not chained:
        pyplot.show()

    return ax


def plot_magnitude_test(evaluation_result, axes=None, ax=None, show=True, plot_args=None):
    """
    M-test 直方图绘制；兼容 ax / axes，两者任选其一。
    - 若未提供轴，则内部创建新 figure/axes。
    - 链式绘制（传入 ax/axes）时不自动 show()，便于在子图中组合绘制。
    """
    # --- 兼容参数名 ---
    if ax is not None and axes is None:
        axes = ax
    chained = axes is not None

    # --- 参数读取 ---
    plot_args = dict(plot_args or {})
    title = plot_args.get('title', 'Magnitude Test')
    title_fontsize = plot_args.get('title_fontsize', None)
    xlabel = plot_args.get('xlabel', 'D* Statistic')
    xlabel_fontsize = plot_args.get('xlabel_fontsize', None)
    ylabel = plot_args.get('ylabel', 'Number of catalogs')
    ylabel_fontsize = plot_args.get('ylabel_fontsize', None)
    tight_layout = plot_args.get('tight_layout', True)
    percentile = plot_args.get('percentile', 95)
    text_fontsize = plot_args.get('text_fontsize', 14)
    filename = plot_args.get('filename', None)
    bins = plot_args.get('bins', 'auto')
    xy = plot_args.get('xy', (0.55, 0.6))
    legend = plot_args.get('legend', True)
    legend_loc = plot_args.get('legend_loc', 'best')

    # 外观（可被 plot_args 覆盖；需 plot_histogram 使用这些键）
    plot_args.setdefault('hist_color', 'C1')
    plot_args.setdefault('hist_alpha', 0.8)
    plot_args.setdefault('hist_edgecolor', 'none')
    plot_args.setdefault('obs_line_color', 'C1')
    plot_args.setdefault('ci_color', 'C1')
    plot_args.setdefault('legend', legend)

    # 固定标签（供图例使用）
    plot_args.update({
        'obs_label': evaluation_result.obs_name,
        'sim_label': evaluation_result.sim_name,
    })

    # --- 选择/创建 Axes ---
    if axes is None:
        figsize = plot_args.get('figsize', (6.4, 4.8))
        fig, axes = plt.subplots(figsize=figsize)
    ax = axes

    # --- 绘制 ---
    ax_ret = plot_histogram(
        evaluation_result.test_distribution,
        evaluation_result.observed_statistic,
        catalog=evaluation_result.obs_catalog_repr,
        plot_args=plot_args,
        bins=bins,
        axes=ax,
        percentile=percentile
    )
    if ax_ret is not None:
        ax = ax_ret

    # --- 注释 ---
    if not chained:
        try:
            ax.annotate(
                '$\\gamma = P(X \\ge x) = {:.2f}$\\n$\\omega = {:.2f}$'
                .format(evaluation_result.quantile, evaluation_result.observed_statistic),
                xycoords='axes fraction', xy=xy, fontsize=text_fontsize
            )
        except TypeError:
            ax.annotate(
                '$\\gamma = P(X \\ge x) = {:.2f}$\\n$\\omega = {:.2f}$'
                .format(evaluation_result.quantile[0], evaluation_result.observed_statistic),
                xycoords='axes fraction', xy=xy, fontsize=text_fontsize
            )

    # --- 轴标题 ---
    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=xlabel_fontsize)
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize)

    # --- 图例 ---
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc=legend_loc)

    # --- 布局/保存/显示 ---
    if tight_layout and ax.figure is not None:
        ax.figure.tight_layout()

    if filename is not None and ax.figure is not None:
        ax.figure.savefig(filename + '.pdf')
        ax.figure.savefig(filename + '.png', dpi=300)

    if show and not chained:
        pyplot.show()

    return ax
