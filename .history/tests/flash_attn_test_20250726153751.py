import torch
from flash_attn.flash_attn_interface import flash_attn_varlen_func
import matplotlib.pyplot as plt

# ---------- 模拟输入 ----------
batch_size = 2
nheads_q = 2
nheads_k = 1
headdim = 32

# token 数 = 3 + 2 = 5
total_q = 5
total_k = 5

# 满足要求：q 的 head 数 是 k/v 的整数倍
q = torch.randn(total_q, nheads_q, headdim, device='cuda', dtype=torch.float16, requires_grad=True)
k = torch.randn(total_k, nheads_k, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(total_k, nheads_k, headdim, device='cuda', dtype=torch.float16)

cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32, device='cuda')
max_seqlen_q = 3
max_seqlen_k = 3

# ---------- 调用 flash attn ----------
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
    window_size=(1, 1),  # 可调
    return_attn_probs=True
)

print("输出 shape:", out.shape)
print("注意力矩阵 shape:", attn_probs.shape)

# ---------- 打印注意力矩阵 ----------
# shape: (batch_size, nheads_q, seqlen_q, seqlen_k)
# 打印第 0 个 batch、head 0 的 attention map
print("\nAttention matrix (batch 0, head 0):")
print(attn_probs[0, 0])  # shape: (3, 3)

# 每个 query 最关注的 key
topk_idx = attn_probs[0, 0].argmax(dim=-1)
print("\n每个 query 最关注的 key 索引：", topk_idx.tolist())

# ---------- 可视化 ----------
plt.imshow(attn_probs[0, 0].detach().cpu().numpy(), cmap='viridis')
plt.title("Attention Heatmap (Batch 0, Head 0)")
plt.xlabel("Key Index")
plt.ylabel("Query Index")
plt.colorbar()
plt.show()
