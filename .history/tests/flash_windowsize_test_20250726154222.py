import torch
from math import sqrt
import src
from src.models.transformer.attentions import FlashAttentionWrapper
from src.utils.utils import set_seed
set_seed(42)
B, L, H, D = 2, 8, 4, 32  # batch, seq_len, num_heads, dim_per_head
device = "cuda" if torch.cuda.is_available() else "cpu"

# 构造测试数据
q = torch.randn(B, L, H, D, device=device)
k = torch.randn(B, L, H, D, device=device)
v = torch.randn(B, L, H, D, device=device)
non_pad_mask = torch.ones(B, L, device=device)

# 创建 FlashAttentionWrapper 实例
attention = FlashAttentionWrapper(attn_dropout=0.1, output_attention=False, scale=None, precision="fp16").to(device)

# 执行 forward 进行测试
out, _ = attention(q, k, v, non_pad_mask=non_pad_mask)

# 输出 shape 验证
print("Output shape:", out.shape)  # 应为 [B, L, H, D]
