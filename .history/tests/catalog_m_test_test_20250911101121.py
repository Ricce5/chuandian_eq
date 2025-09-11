# %%
import numpy as np
from src.utils.catalog_tests import magnitude_test_from_counts
# --- 构造假的 Catalog 类 ---
class FakeCatalog:
    def __init__(self, counts):
        self._counts = np.array(counts)

    def magnitude_counts(self):
        return self._counts

# --- 构造观测数据 (obs) ---
observed_catalog = FakeCatalog([5, 10, 15, 8, 2])   # 模拟观测的震级分布

# --- 构造预测数据 (forecast) ---
forecast_catalogs = [
    FakeCatalog([4, 9, 16, 7, 3]),
    FakeCatalog([6, 11, 14, 9, 1]),
    FakeCatalog([5, 10, 15, 8, 2])  # 刚好等于观测的
]

# --- 调用测试函数 ---
result_dict = magnitude_test_from_counts(
    forecast_catalogs,
    observed_catalog,
    verbose=True,
    return_quantiles=True
)

# --- 输出结果 ---
print("Test distribution:", result_dict["test_distribution"])
print("Observed d-statistic:", result_dict["obs_d_statistic"])
print("Scaled union histogram:", result_dict["scaled_union_histogram"])
print("Quantiles (delta_1, delta_2):", result_dict["delta_1"], result_dict["delta_2"])

# %%
