# tests/attn_pl_biased_test.py
import torch
import torch.nn as nn
import pytest

from src.models.extractors.attn_time_biased import TimeAwareAttnPool

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_batch(B=4, L=16, D=32, t_as_3d=False):
    x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)
    # 构造带有 padding 的 mask：前半有效，后半无效
    mask = torch.ones(B, L, device=DEVICE, dtype=torch.bool)
    mask[:, L // 2 :] = False

    # t in [0,1]
    t = torch.rand(B, L, device=DEVICE)
    if t_as_3d:
        t = t.unsqueeze(-1)  # [B,L,1]

    return x, mask, t

def build_model(D=32, H=64, bias_type="linear"):
    model = TimeAwareAttnPool(
        d_model=D,
        d_hidden=H,
        t_dim=16,
        bias_type=bias_type,
        alpha0=10.0,
        device=DEVICE,
    )
    return model

def test_forward_backward_linear():
    """基本的前向 + 反向是否可运行且梯度存在"""
    B, L, D, H = 4, 16, 32, 64
    x, mask, t = make_batch(B, L, D, t_as_3d=False)
    model = build_model(D, H, bias_type="linear")

    pooled, alpha = model(x, mask, {"event_time": t}, return_score=True)

    # 形状检查
    assert pooled.shape == (B, D)
    assert alpha.shape == (B, L)

    # mask 后被屏蔽位置的注意力应为 0
    assert torch.all(alpha[:, L // 2 :] == 0)

    # 反传
    loss = (pooled ** 2).mean()
    loss.backward()

    # 关键参数/输入应有梯度
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} 没有梯度"

def test_forward_backward_log_bias():
    """bias_type='log' 也应正常工作"""
    B, L, D, H = 2, 10, 16, 32
    x, mask, t = make_batch(B, L, D, t_as_3d=False)
    model = build_model(D, H, bias_type="log")

    pooled = model(x, mask, {"event_time": t}, return_score=False)
    assert pooled.shape == (B, D)

    loss = pooled.abs().mean()
    loss.backward()

    # 简单的有限数检查
    assert torch.isfinite(pooled).all()

def test_event_time_shape_handling():
    """event_time 同时支持 [B,L] 与 [B,L,1] 两种输入形状"""
    B, L, D, H = 3, 12, 24, 48
    x1, mask1, t2d = make_batch(B, L, D, t_as_3d=False)
    x2, mask2, t3d = make_batch(B, L, D, t_as_3d=True)
    model = build_model(D, H, bias_type="linear")

    pooled2d, alpha2d = model(x1, mask1, {"event_time": t2d}, return_score=True)
    pooled3d, alpha3d = model(x2, mask2, {"event_time": t3d}, return_score=True)

    assert pooled2d.shape == pooled3d.shape == (B, D)
    assert alpha2d.shape == alpha3d.shape == (B, L)
    # 基本数值合理性
    assert torch.allclose(alpha2d.sum(dim=1), torch.ones(B, device=DEVICE), atol=1e-5)
    assert torch.allclose(alpha3d.sum(dim=1), torch.ones(B, device=DEVICE), atol=1e-5)

if __name__ == "__main__":
    # 允许直接运行此文件进行快速验证
    pytest.main([__file__, "-q"])
