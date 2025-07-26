import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func

# 模拟一个批次中的两个句子：3 个 token 和 2 个 token
q = torch.randn(5, 2, 32, device='cuda', dtype=torch.float16, requires_grad=True)  # total_q = 5
k = torch.randn(5, 1, 32, device='cuda', dtype=torch.float16)
v = torch.randn(5, 1, 32, device='cuda', dtype=torch.float16)

# 每个序列的累计偏移索引（batch size = 2）
cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device='cuda')

# 设置 max_seqlen
max_seqlen_q = 3
max_seqlen_k = 3

# 设置滑动窗口 attention
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens, cu_seqlens,
    max_seqlen_q, max_seqlen_k,
    dropout_p=0.0,
    causal=False,
    window_size=(1, 1),  # 每个 query 只能看到前后 1 个 key
    return_attn_probs=True
)

# 查看输出
print("输出 shape:", output[0].shape)  # 应该是 (5, 2, 32)
print("Softmax logsumexp:", output[1].shape)  # optional
print("Attention probs shape:", output[2].shape)  # optional
