import torch

def interp_uniform_time_series(t, x, t_query, clamp=False):
    """
    t:       (B, T)      等间隔时间轴
    x:       (B, T, F)   特征
    t_query: (B, Nq)     查询时间
    return:  (B, Nq, F)  插值后的值
    """
    B, T = t.shape
    F = x.shape[-1]

    # (B, 1)
    t0 = t[:, 0:1]
    dt = (t[:, 1] - t[:, 0]).view(B, 1)  #

    if clamp:
        t_min = t[:, 0:1]
        t_max = t[:, -1:]
        tq = t_query.clamp(t_min, t_max)
    else:
        tq = t_query

    u = (tq - t0) / dt  # (B, Nq)
    i = torch.floor(u).long()           
    i = i.clamp(0, T - 2)              
    s = (u - i).unsqueeze(-1)      

    idx_i = i.unsqueeze(-1).expand(-1, -1, F)       # (B, Nq, F)
    idx_ip1 = (i + 1).unsqueeze(-1).expand(-1, -1, F)

    x_i   = torch.gather(x, 1, idx_i)   # (B, Nq, F)
    x_ip1 = torch.gather(x, 1, idx_ip1)

    xq = x_i + (x_ip1 - x_i) * s       
    return xq
import torch

def integrate_uniform_time_series(t, x, t_start, t_end, clamp=True):
    """
    在等间隔时间轴上，对线性插值后的 x(t) 在 [t_start, t_end] 上积分。

    Args:
        t:       (B, T)      等间隔时间轴（每个 batch 内等间隔）
        x:       (B, T, F)   对应特征
        t_start: (B,) or (B, N)  积分下限
        t_end:   (B,) or (B, N)  积分上限
        clamp:   是否把 t_start/t_end 限制到 [t_min, t_max]

    Returns:
        I:       (B, N, F)   每个区间的积分值；如果某些地方 t_end < t_start，则结果为负（符合积分方向）
    """
    assert t.dim() == 2 and x.dim() == 3
    B, T = t.shape
    F = x.shape[-1]
    assert x.shape[0] == B and x.shape[1] == T

    # 统一成 (B, N)
    if t_start.dim() == 1:
        t_start = t_start[:, None]      # (B, 1)
    if t_end.dim() == 1:
        t_end = t_end[:, None]          # (B, 1)
    assert t_start.shape == t_end.shape
    assert t_start.shape[0] == B
    _, N = t_start.shape

    device = t.device
    dtype = x.dtype

    # 每个 batch 的 dt, t0
    dt = (t[:, 1] - t[:, 0]).view(B, 1, 1)  # (B, 1, 1)
    t0 = t[:, 0].view(B, 1)                 # (B, 1)

    # 可选：把积分上下限 clamp 到时间范围内
    if clamp:
        t_min = t[:, 0:1]   # (B, 1)
        t_max = t[:, -1:]   # (B, 1)
        t_start = t_start.clamp(t_min, t_max)
        t_end   = t_end.clamp(t_min, t_max)

    # 1. 先算每个 segment 的面积，再累加得到 F(t_k)
    x0 = x[:, :-1, :]           # (B, T-1, F)
    x1 = x[:, 1:,  :]           # (B, T-1, F)
    seg_area = dt * (x0 + x1) / 2.0  # (B, T-1, F)

    # A_nodes[b, k] = ∫_{t0}^{t_k} x(t) dt
    A_nodes = torch.zeros((B, T, F), device=device, dtype=dtype)
    A_nodes[:, 1:, :] = torch.cumsum(seg_area, dim=1)

    # 2. 定义一个连续 F(t_query) 的函数（按线性插值解析积分）
    def F_continuous(tq):   # tq: (B, N)
        # 连续索引 u
        u = (tq - t0) / dt.squeeze(-1)   # (B, N)，这里 dt.squeeze(-1)->(B,1)

        # 数值上稍微往里 clamp 一点，避免刚好落到 T-1 时的浮点边界问题
        u = u.clamp(0.0, T - 1 - 1e-6)

        idx = torch.floor(u).long()      # (B, N)，所在 segment 左端点索引
        idx = idx.clamp(max=T - 2)       # 保证 idx+1 不越界

        s = (u - idx).unsqueeze(-1)      # (B, N, 1)，归一化位置 [0,1)

        # gather x_k, x_{k+1}, A_k
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, F)   # (B, N, F)
        x_k   = torch.gather(x,       1, idx_exp)       # (B, N, F)
        x_k1  = torch.gather(x,       1, idx_exp + 1)
        A_k   = torch.gather(A_nodes, 1, idx_exp)

        # F(t) = A_k + dt * ( x_k * s + 0.5 * (x_{k+1}-x_k) * s^2 )
        return A_k + dt * (x_k * s + 0.5 * (x_k1 - x_k) * (s ** 2))

    # 3. 积分 = F(t_end) - F(t_start)
    F_start = F_continuous(t_start)   # (B, N, F)
    F_end   = F_continuous(t_end)     # (B, N, F)

    I = F_end - F_start               # (B, N, F)
    return I
