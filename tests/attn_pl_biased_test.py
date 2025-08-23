import torch
import torch.nn as nn
import math
from src.models.extractors.attn_time_biased import TimeAwareAttnPool

# ======= 测试 1：基本前向/反向 =======
def test_forward_backward():
    torch.manual_seed(0)
    B, L, D, H = 2, 5, 32, 16
    x = torch.randn(B, L, D, requires_grad=True)

    # 构造递增的时间戳 [0..1]，并做一个 padding 掩码
    t = torch.linspace(0, 1, L).unsqueeze(0).repeat(B, 1)  # [B,L]
    mask = torch.ones(B, L, dtype=torch.bool)
    mask[0, -1] = False  # 第0个样本最后一个位置是padding
    t = t * mask          # 被mask的位置“时间=0”，与实现一致

    model = TimeAwareAttnPool(d_model=D, d_hidden=H, bias_type="linear")
    pooled, alpha = model(x, mask, t)

    print("pooled shape:", pooled.shape)   # 期望 [B, D]
    print("alpha shape:", alpha.shape)     # 期望 [B, L]
    print("alpha row sums:", alpha.sum(dim=1))  # 被mask的位权重为0

    # 反向传播看看梯度是否顺畅
    loss = pooled.pow(2).mean()
    loss.backward()
    print("x.grad nan? ", torch.isnan(x.grad).any().item())

# ======= 测试 2：只看“时间偏置”能否把注意力拉向最近时刻 =======
def test_time_bias_only_linear():
    torch.manual_seed(0)
    B, L, D, H = 1, 6, 8, 8
    x = torch.zeros(B, L, D)  # 用0简化内容影响

    # 时间均匀递增，最后两个 token 有效，其余有效，演示“最近时刻=最大t”
    t = torch.linspace(0, 1, L).unsqueeze(0)  # [1,L]
    mask = torch.ones(B, L, dtype=torch.bool)

    model = TimeAwareAttnPool(d_model=D, d_hidden=H, bias_type="linear")

    # 让内容打分“归零”：W、v、t_mlp全置0，只有时间偏置起作用
    with torch.no_grad():
        for m in [model.W, model.v]:
            m.weight.zero_()
            if m.bias is not None:
                m.bias.zero_()
        for layer in model.t_mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
        model.g.copy_(torch.tensor(5.0))  # 放大时间偏置强度

    pooled, alpha = model(x, mask, t)
    print("alpha (linear bias):", torch.round(alpha[0]*100)/100)

    # 检查是否单调递增（越“新”越大）
    is_monotonic = bool(torch.all(alpha[0, 1:] >= alpha[0, :-1]))
    print("monotonic non-decreasing?", is_monotonic)

# ======= 测试 3：对比 linear vs log 偏置 =======
def test_compare_linear_log():
    torch.manual_seed(0)
    B, L, D, H = 1, 6, 8, 8
    x = torch.zeros(B, L, D)
    t = torch.linspace(0, 1, L).unsqueeze(0)
    mask = torch.ones(B, L, dtype=torch.bool)

    # 先做 log
    model_log = TimeAwareAttnPool(d_model=D, d_hidden=H, bias_type="log", alpha0=10.0)
    with torch.no_grad():
        for m in [model_log.W, model_log.v]:
            m.weight.zero_(); m.bias.zero_()
        for layer in model_log.t_mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_(); layer.bias.zero_()
        model_log.g.copy_(torch.tensor(5.0))

    _, alpha_log = model_log(x, mask, t)

    # 再做 linear（保持同样设置）
    model_lin = TimeAwareAttnPool(d_model=D, d_hidden=H, bias_type="linear")
    with torch.no_grad():
        for m in [model_lin.W, model_lin.v]:
            m.weight.zero_(); m.bias.zero_()
        for layer in model_lin.t_mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.zero_(); layer.bias.zero_()
        model_lin.g.copy_(torch.tensor(5.0))

    _, alpha_lin = model_lin(x, mask, t)

    print("alpha_log:", torch.round(alpha_log[0]*100)/100)
    print("alpha_lin:", torch.round(alpha_lin[0]*100)/100)
    print("last token weight (log, linear):", float(alpha_log[0,-1]), float(alpha_lin[0,-1]))

if __name__ == "__main__":
    print("== Test 1: forward/backward ==")
    test_forward_backward()
    print("\n== Test 2: time bias only (linear) ==")
    test_time_bias_only_linear()
    print("\n== Test 3: compare linear vs log ==")
    test_compare_linear_log()
