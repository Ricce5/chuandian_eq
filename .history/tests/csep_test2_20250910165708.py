# -*- coding: utf-8 -*-
# 仅时间+震级的目录型评估（N / M / PL）；不需要空间信息。
# 如果你已有自己的观测与模拟目录，直接把 “构造示例数据” 那段替换为你的加载逻辑即可。

import os, json, math, numpy as np
from datetime import datetime, timedelta

import csep
from csep.core import catalog_evaluations as cat_eval
from csep.core.forecasts import CatalogForecast
# CSEPCatalog 的位置在不同版本略有差异，下面两条任选其一（保留一条能 import 成功的）
try:
    from csep.core.catalogs import CSEPCatalog
# ---------- 工具 ----------
def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")

def simulate_catalog_time_only(start: datetime, end: datetime,
                               rate_per_day=2.0, b_value=1.0, mmin=3.0, mmax=6.0):
    """简化：泊松过程生成事件时刻，Gutenberg–Richter 生成震级（截断）"""
    days = (end - start).days
    n = np.random.poisson(rate_per_day * days)
    ts = np.random.uniform(0, (end - start).total_seconds(), size=n)
    times = np.sort([start + timedelta(seconds=float(s)) for s in ts])

    # 震级：F(m) = (1 - 10^{-b(m-mmin)}) / (1 - 10^{-b(mmax-mmin)})
    u = np.random.rand(n)
    denom = 1.0 - 10**(-b_value * (mmax - mmin))
    mags = mmin - (1.0 / b_value) * np.log10(1.0 - u * denom)
    mags = np.clip(mags, mmin, mmax)

    events = [{"time": iso(t), "magnitude": float(m)} for t, m in zip(times, mags)]
    return events

def catalog_from_events(events, start_time, end_time, min_mag):
    """把 {time, magnitude} 列表转成 CSEPCatalog"""
    # CSEPCatalog.from_dict 在较新版本可用；若无该方法，可用构造器创建
    if hasattr(CSEPCatalog, "from_dict"):
        return CSEPCatalog.from_dict({
            "start_time": iso(start_time),
            "end_time": iso(end_time),
            "min_magnitude": float(min_mag),
            "events": events
        })
    else:
        # 手工构造：仅提供必须字段（时间、震级）；经纬度可缺省
        times = np.array([datetime.fromisoformat(e["time"]) for e in events], dtype="datetime64[ns]")
        mags  = np.array([e["magnitude"] for e in events], dtype=float)
        cat = CSEPCatalog()
        cat._data = {
            "origin_time": times,
            "magnitude": mags,
        }
        cat.start_time = start_time
        cat.end_time = end_time
        cat.min_magnitude = float(min_mag)
        cat.region = None
        return cat

# ---------- 1) 构造示例数据（你有真实数据就把这段换成读取你的文件） ----------
np.random.seed(123)
START = datetime(2019, 7, 4)
END   = datetime(2019, 8, 3)   # 30 天
MMIN  = 3.0

# 观测目录（稍微与模拟不同的平均率）
obs_events = simulate_catalog_time_only(START, END, rate_per_day=2.3, b_value=1.0, mmin=MMIN, mmax=5.5)
obs = catalog_from_events(obs_events, START, END, MMIN)

# 模拟目录集合（J 条）
J = 1000
sim_catalogs = []
for j in range(J):
    # 给模拟的日率加随机性：对数正态围绕 2.0
    rate = float(np.random.lognormal(mean=np.log(2.0), sigma=0.25))
    evs = simulate_catalog_time_only(START, END, rate_per_day=rate, b_value=1.0, mmin=MMIN, mmax=5.5)
    sim_catalogs.append(catalog_from_events(evs, START, END, MMIN))

# 打包成 Catalog-based forecast
forecast = CatalogForecast.from_catalogs(
    catalogs=sim_catalogs,
    start_time=START,
    end_time=END,
    name="time_only_catalog_forecast",
    region=None,               # 没有空间信息就留 None；或自建 1x1 区域也行
    min_magnitude=MMIN
)

# ---------- 2) 目录型一致性评估（N / M / PL；无空间） ----------
# N-test：观测总数在模拟总数分布中的分位
n_res  = cat_eval.number_test(forecast, obs)

# M-test：若有 magnitude（我们有），比较震级频率分布
m_res  = cat_eval.magnitude_test(forecast, obs)

# PL-test：用模拟目录估计期望率，计算伪似然（时间+震级的综合）
pl_res = cat_eval.pseudolikelihood_test(forecast, obs)

# ---------- 3) 输出与可视化 ----------
# 文本输出（核心统计量）
def head(res):
    return {k: getattr(res, k, None) for k in ["observed_statistic", "quantile", "p_value", "test_distribution_name", "name"]}

print("\n=== N-test ===")
print(head(n_res))
print("\n=== M-test ===")
print(head(m_res))
print("\n=== PL-test ===")
print(head(pl_res))

# 保存 JSON（便于留痕 / 论文附录）
csep.write_json(n_res,  "n_test.json")
csep.write_json(m_res,  "m_test.json")
csep.write_json(pl_res, "pl_test.json")

# 可选：画 N-test 图（需要 GUI 或把图存盘）
try:
    from csep.utils import plots
    ax = plots.plot_number_test(n_res)
    import matplotlib.pyplot as plt
    plt.savefig("n_test_plot.png", bbox_inches="tight")
    plt.close()
    print("\nSaved figure: n_test_plot.png")
except Exception as e:
    print("\n[Warn] plot skipped:", e)

print("\nDone.")
