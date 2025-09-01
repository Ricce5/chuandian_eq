# tests/attn_pl_biased_test.py
import torch
import pytest

from src.models.extractors.attn_time_biased import TimeAwareAttnPool

torch.manual_seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_batch(B=4, L=16, D=32, t_as_3d=False):
    """构造一批次数据：x、mask（后半段为 padding）、t∈[0,1]"""
    x = torch.randn(B, L, D, device=DEVICE, requires_grad=True)

    mask = torch.ones(B, L, device=DEVICE, dtype=torch.bool)
    mask[:, L // 2 :] = False  # 后半段无效

    t = torch.rand(B, L, device=DEVICE)
    if t_as_3d:
        t = t.unsqueeze(-1)  # [B,L,1]

    return x, mask, t


def build_model(D=32, H=64, bias_type="linear"):
    return TimeAwareAttnPool(
        d_model=D,
        d_hidden=H,
        t_dim=16,
        bias_type=bias_type,
        alpha0=10.0,
        device=DEVICE,
    )


def _common_shape_and_mask_checks(pooled, alpha, B, L, D, mask_split_idx):
    """通用形状与 mask 断言"""
    assert pooled.shape == (B, D)
    assert alpha.shape == (B, L)
    # mask 后被屏蔽位置的注意力应为 0
    assert torch.all(alpha[:, mask_split_idx:] == 0)
    # 每个样本注意力和为 1（数值稳定性考虑一个小容差）
    s = alpha.sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5)


def test_forward_backward_linear():
    """bias_type='linear' 下的前向 + 反向与梯度检查"""
    B, L, D, H = 4, 16, 32, 64
    x, mask, t = make_batch(B, L, D, t_as_3d=False)
    model = build_model(D, H, bias_type="linear")

    pooled, alpha = model(x, mask, {"event_time": t}, return_score=True)

    _common_shape_and_mask_checks(pooled, alpha, B, L, D, L // 2)

    # 反传
    loss = (pooled ** 2).mean()
    loss.backward()

    # 输入与参数应有梯度（线性分支下 log_alpha 不参与前向，不要求它有梯度）
    assert x.grad is not None and torch.isfinite(x.grad).all()

    got_any_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and name != "log_alpha":
            assert p.grad is not None, f"{name} 没有梯度（linear 分支）"
            got_any_grad = True
    assert got_any_grad, "linear 分支下应至少有一个参数得到梯度"


def test_forward_backward_log_bias():
    """bias_type='log' 分支应能前后向，并且 log_alpha 参与梯度"""
    B, L, D, H = 2, 10, 16, 32
    x, mask, t = make_batch(B, L, D, t_as_3d=False)
    model = build_model(D, H, bias_type="log")

    pooled = model(x, mask, {"event_time": t}, return_score=False)
    assert pooled.shape == (B, D)

    loss = pooled.abs().mean()
    loss.backward()

    # 数值有限性
    assert torch.isfinite(pooled).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    grads = dict(model.named_parameters())
    # log 分支：log_alpha 参与 _time_bias -> 必须有梯度
    assert grads["log_alpha"].grad is not None, "log 分支下 log_alpha 应该有梯度"


def test_event_time_shape_handling():
    """event_time 同时支持 [B,L] 与 [B,L,1] 两种形状"""
    B, L, D, H = 3, 12, 24, 48
    x2d, mask2d, t2d = make_batch(B, L, D, t_as_3d=False)
    x3d, mask3d, t3d = make_batch(B, L, D, t_as_3d=True)
    model = build_model(D, H, bias_type="linear")

    pooled2d, alpha2d = model(x2d, mask2d, {"event_time": t2d}, return_score=True)
    pooled3d, alpha3d = model(x3d, mask3d, {"event_time": t3d}, return_score=True)

    _common_shape_and_mask_checks(pooled2d, alpha2d, B, L, D, L // 2)
    _common_shape_and_mask_checks(pooled3d, alpha3d, B, L, D, L // 2)


if __name__ == "__main__":
    # 允许直接运行该测试文件： python tests/attn_pl_biased_test.py
    # pytest.main([__file__, "-q"])
    pytest.main([__file__])

