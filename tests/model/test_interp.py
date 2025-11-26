# %%
import torch
from src.utils.interp import interp_uniform_time_series,integrate_uniform_time_series
# %%
# ====== 构造简单数据 ======
B = 2       # batch size
T = 5       # 时间点个数
F = 1       # 特征维度

# 时间轴：每个 batch 一样：[0, 1, 2, 3, 4]
t_single = torch.arange(T, dtype=torch.float32)      # (T,)
t = t_single.unsqueeze(0).repeat(B, 1)               # (B, T)

# 特征：x(t) = t
x = t.unsqueeze(-1)                                  # (B, T, F)，F=1

# 查询时间：在区间 [0, 4] 内随便取一些点
t_query = torch.tensor([
    [0.0, 0.5, 1.2, 2.7, 3.9],
    [0.3, 0.8, 1.5, 2.2, 4.0],
], dtype=torch.float32)                              # (B, Nq)

# ====== 调用插值函数 ======
xq = interp_uniform_time_series(t, x, t_query, clamp=True)

print("t:")
print(t)
print("x (x=t):")
print(x.squeeze(-1))
print("t_query:")
print(t_query)
print("xq (插值结果):")
print(xq.squeeze(-1))  # 去掉最后一维 F 看得更清楚
# %%
# 还是复用上面的 t 和 x (x=t)

# 设每个 batch 有两个区间要积分
t_start = torch.tensor([
    [0.0, 1.0],   # batch 0: 积分 [0,2], [1,3]
    [0.5, 2.0],   # batch 1: 积分 [0.5, 3.5], [2,4]
], dtype=torch.float32)  # (B, N)

t_end = torch.tensor([
    [2.0, 3.0],
    [3.5, 4.0],
], dtype=torch.float32)   # (B, N)

I = integrate_uniform_time_series(t, x, t_start, t_end, clamp=True)  # (B, N, F)

print("t:")
print(t)
print("x (x=t):")
print(x.squeeze(-1))

print("t_start:")
print(t_start)
print("t_end:")
print(t_end)

print("数值积分结果 I:")
print(I.squeeze(-1))  # (B, N)

# ====== 计算解析解做对比 ======
def analytical_integral(a, b):
    return 0.5 * (b ** 2 - a ** 2)

I_true = analytical_integral(t_start, t_end)
print("解析积分结果 I_true:")
print(I_true)
print("误差 I - I_true:")
print(I.squeeze(-1) - I_true)


# %%
