import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
import matplotlib.pyplot as plt

# ---------- 模拟输入参数 ----------
# 假设有两个样本，分别有 3 和 2 个 token，头数 = 1，head dim = 32
batch_size = 2
nheads = 1
headdim = 32

# token 数 = 3 + 2 = 5
total_q = 5
total_k = 5

# q, k, v 的 shape: (total_seq_len, nheads, headdim)
q = torch.randn(total_q, nheads, headdim, device='cuda', dtype=torch.float16, requires_grad=True)
k = torch.randn(total_k, nheads, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(total_k, nheads, headdim, device='cuda', dtype=torch.float16)

# 每个序列的起始位置索引（累加）
cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device='cuda')

# ---------- 参数设置 ----------
max_seqlen_q = 3
max_seqlen_k = 3

# 启用 sliding window（例如：每个 query 只能看左右 1 个）
window_size = (1, 1)

# ---------- 调用函数 ----------
out, softmax_lse, attn_probs = flash_attn_varlen_func(
    q=q,
    k=k,
    v=v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    dropout_p=0.0,
    causal=False,
    window_size=window_size,
    return_attn_probs=True
)

# ---------- 查看输出 ----------
print("输出 shape:", out.shape)
print("注意力矩阵 shape:", attn_probs.shape)

# 打印第一个 batch 第一个 head 的注意力权重（query vs key）
print("\nAttention matrix (batch 0, head 0):")
print(attn_probs[0, 0])  # shape: (3, 3) for first sample

# 获取每个 query 最关注的 key
topk_idx = attn_probs[0, 0].argmax(dim=-1)
print("\n每个 query 最关注的 key 索引：", topk_idx.tolist())

# ---------- 可视化 ----------
plt.imshow(attn_probs[0, 0].detach().cpu().numpy(), cmap='viridis')
plt.title("Attention Heatmap (Batch 0, Head 0)")
plt.xlabel("Key Index")
plt.ylabel("Query Index")
plt.colorbar()
plt.show()
