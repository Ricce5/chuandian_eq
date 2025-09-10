# %%
import numpy as np
import matplotlib.pyplot as plt

# 假设模型给了 5000 个模拟目录
# 每个目录的地震数 ~ 泊松(均值=20)
event_counts = np.random.poisson(lam=20, size=5000)

# 假设观测目录里观测到 19 个地震
obs_count = 19

# 计算 delta_1 和 delta_2
delta_1 = np.mean(event_counts >= obs_count)  # P(N >= N_obs)
delta_2 = np.mean(event_counts <= obs_count)  # P(N <= N_obs)

print(f"观测值: {obs_count}")
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
