# %%
import src
from src.data.preparation import prepare_data_tpp
from config.config_loader import load_args_from_yaml 
args= load_args_from_yaml("../config/mixer_tpp.yaml")
base_dir = f"../data/{args.dataset}"
# %%
seq, train_loader, val_loader, test_loader, catalog_ds = prepare_data_tpp(base_dir=base_dir, args=args,)
# %%
import math
from typing import Dict, Optional
import torch

LN10 = math.log(10.0)

def update_bayes_gr_over_sequence(
    seq,                      # 你的 Sequence 对象（DotDict）
    Mc: float = 2.5,          # 完备震级
    delta: float = 0.97,      # 折扣（越小越快适应时变）
    a0: float = 1e-3,         # Gamma 先验 shape
    s0: float = 1e-3,         # Gamma 先验 rate
    mag_key: str = "mag",     # 震级属性名（在 seq 中的键）
    write_back: bool = True,  # 是否把结果写回到 seq
) -> Dict[str, torch.Tensor]:
    """
    Event-by-event Bayesian updating for GR b-value with time-variation (discounted conjugate Gamma on alpha=b*ln10).

    Returns:
        dict with keys:
          - a_t, s_t: 每个事件后（折扣+更新后）的 Gamma 参数
          - b_mean, b_sd: 每个事件后的 b 后验均值与标准差
          - b_lo, b_hi: 约 95% 置信带（正态近似）
          - used_mask: 是否 (mag >= Mc) 被用于本次更新（0/1）
    Side effect:
        若 write_back=True，会把这些张量写回到 seq 中作为同名字段。
    """
    if mag_key not in seq:
        raise KeyError(f"Sequence 缺少震级属性 '{mag_key}'。已有键：{list(seq.keys())}")

    mags: torch.Tensor = seq[mag_key].to(dtype=torch.get_default_dtype())
    device = mags.device
    n = mags.shape[0]

    # 结果容器
    a_t = torch.empty(n, dtype=mags.dtype, device=device)
    s_t = torch.empty(n, dtype=mags.dtype, device=device)
    b_mean = torch.empty(n, dtype=mags.dtype, device=device)
    b_sd = torch.empty(n, dtype=mags.dtype, device=device)
    b_lo = torch.empty(n, dtype=mags.dtype, device=device)
    b_hi = torch.empty(n, dtype=mags.dtype, device=device)
    used_mask = torch.zeros(n, dtype=torch.int64, device=device)

    # 标量状态（放到同一 device）
    a = torch.tensor(a0, dtype=mags.dtype, device=device)
    s = torch.tensor(s0, dtype=mags.dtype, device=device)

    z975 = torch.tensor(1.959963984540054, dtype=mags.dtype, device=device)

    for i in range(n):
        # 事件间折扣（时变性）
        a = a * delta
        s = s * delta

        m_i = mags[i]
        if m_i >= Mc:
            x_i = m_i - Mc
            a = a + 1.0
            s = s + x_i
            used_mask[i] = 1  # 本事件用于更新

        # 逐事件后验摘要（映射到 b）
        bm = a / (s * LN10)
        bstd = torch.sqrt(a) / (s * LN10)
        blo = torch.clamp(bm - z975 * bstd, min=0.0)
        bhi = bm + z975 * bstd

        a_t[i] = a
        s_t[i] = s
        b_mean[i] = bm
        b_sd[i] = bstd
        b_lo[i] = blo
        b_hi[i] = bhi

    out = dict(
        a_t=a_t, s_t=s_t, b_mean=b_mean, b_sd=b_sd, b_lo=b_lo, b_hi=b_hi, used_mask=used_mask
    )

    if write_back:
        # 把结果直接挂到 seq（DotDict）上，维度与 num_events 对齐
        for k, v in out.items():
            seq[k] = v

    return out

# %%
