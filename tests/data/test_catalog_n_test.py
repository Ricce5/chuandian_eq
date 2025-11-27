# %%
import numpy as np
import matplotlib.pyplot as plt

event_counts = np.random.poisson(lam=1000, size=5000)

obs_count = 19
delta_1 = np.mean(event_counts >= obs_count)  # P(N >= N_obs)
delta_2 = np.mean(event_counts <= obs_count)  # P(N <= N_obs)
print(f"Observed value: {obs_count}")
print(f"δ1 = P(N >= {obs_count}) = {delta_1:.2f}")
print(f"δ2 = P(N <= {obs_count}) = {delta_2:.2f}")

# 可视化
plt.hist(event_counts, bins=30, color="skyblue", edgecolor="black")
plt.axvline(obs_count, color="red", linestyle="--", label=f"Observed = {obs_count}")
plt.title("Number Test")
plt.xlabel("Event count of catalogs")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# %%
from csep.models import (
    CatalogNumberTestResult,
)
from csep.utils.stats import get_quantiles, cumulative_square_diff, MLL_score
delta_1, delta_2 = get_quantiles(event_counts, obs_count)
# prepare result
number_test_result = CatalogNumberTestResult(test_distribution=event_counts,
                                    name='Catalog N-Test',
                                    observed_statistic=obs_count,
                                    quantile=(delta_1, delta_2),
                                    status='normal',
                                    obs_catalog_repr="SCEDC",
                                    sim_name="Poisson (λ=20)",
                                    min_mw=2.0,
                                     obs_name="AZDX",)
# %%
ax = number_test_result.plot()

# %%
