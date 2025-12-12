import torch
import torch.nn.functional as F
from causal_conv1d import causal_conv1d_fn

# 随机生成输入
batch, dim, seqlen = 4, 16, 20
width = 3  # 卷积核大小

# 输入张量
x = torch.randn(batch, dim, seqlen, dtype=torch.float32).cuda()

# 权重 (dim, kernel_width)
weight = torch.randn(dim, width, dtype=torch.float32).cuda()

# 偏置
bias = torch.randn(dim, dtype=torch.float32).cuda()

# GPU 执行 causal_conv1d_fn``
out_causal = causal_conv1d_fn(x, weight, bias=bias)

# 用 F.conv1d 构造等价实现
w_unsq = weight.unsqueeze(1)  # (dim, 1, width)
out_conv1d = F.conv1d(
    x, w_unsq, bias=bias,
    padding=width - 1,        # 填充使输出 >= seqlen
    groups=dim
)[..., :seqlen]              # 只取前 seqlen 长度

# 比较两者是否一致
print("Max abs diff:", (out_causal - out_conv1d).abs().max().item())
print("Allclose:", torch.allclose(out_causal, out_conv1d, atol=1e-6))
