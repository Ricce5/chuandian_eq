# tests/test_rotary_time.py
# %%
import torch
from flash_attn.layers.rotary import RotaryEmbedding as RotaryEmbeddingOrig
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime
from src.utils.utils import set_seed
set_seed(42)

device = "cuda"
dtype = torch.float16

dim = 8
seq_len = 5

rope_orig = RotaryEmbeddingOrig(dim).to(device)
rope_time = RotaryEmbeddingTime(dim).to(device)

# 模拟整数时间索引（0,1,2,...） => 原版行为
times = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)

# 模拟 qkv
qkv = torch.randn(1, seq_len, 3, 1, dim, device=device, dtype=dtype)

# 两个实现
out_orig = rope_orig(qkv.clone())
out_time = rope_time(qkv.clone(), times=times)

# 计算差异
diff = (out_orig - out_time).abs().max().item()
print("Max difference:", diff)

# 验证是否接近
assert diff < 1e-3, "行为与原版不一致"
print("✅ RotaryEmbeddingTime 与原版在整数索引下表现一致")

# %%
print(rope_time._cos_k_cached, rope_time._sin_k_cached,rope_time._cos_cached, rope_time._sin_cached)
# %%
import torch
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime

def test_rotary_embedding_time_strict():
    device = "cuda"
    dtype = torch.float16

    B, L, H, D = 1, 5, 2, 8  # batch, seq_len, num_heads, head_dim
    num_heads = H

    # 模拟 qkv 输入 [B, L, 3, H, D]
    qkv_full = torch.randn(B, L, 3, H, D, device=device, dtype=dtype)
    times = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)

    # ===================== 1️⃣ 离散模式一次性计算 =====================
    rope_full = RotaryEmbeddingTime(D, interleaved=False).to(device)
    out_full = rope_full(qkv_full.clone(), num_heads_q=num_heads)
    print("out_full:", out_full.shape)

    # # ===================== 2️⃣ 离散模式按步计算（模拟 KV Cache） =====================
    print(f"qkv_full:{torch.sum(qkv_full)}")
    rope_cache = RotaryEmbeddingTime(D, interleaved=False).to(device)
    rope_cache._update_cos_sin_cache_with_times(times, device=device, dtype=dtype)
    outs = []
    for i in range(L):
        qkv_step = qkv_full.clone()[:, i:i+1]  # 单步
        out_step = rope_cache(qkv_step, num_heads_q=num_heads, seqlen_offset=i)
        outs.append(out_step)
    out_cache = torch.cat(outs, dim=1)
    print("out_cache:", out_cache.shape)

    max_diff_cache = (out_full - out_cache).abs().max().item()
    print(f"离散模式 vs KV Cache max_diff: {max_diff_cache:.6f}")

    # ===================== 3️⃣ 时间模式 ====================
    print(f"qkv_full:{torch.sum(qkv_full)}")
    # 构造与离散模式一致的时间序列

    rope_time = RotaryEmbeddingTime(D, interleaved=False).to(device)
    print(rope_time._cos_k_cached, rope_time._sin_k_cached, rope_time._cos_cached, rope_time._sin_cached)
    out_time = rope_time(qkv_full.clone(), num_heads_q=num_heads, times=times)
    print("out_time:", out_time.shape)

    max_diff_time = (out_full - out_time).abs().max().item()
    print(f"离散模式 vs 时间模式 max_diff: {max_diff_time:.6f}")
    print(f"out_full: {torch.sum(out_full)}, out_time: {torch.sum(out_time)},")
if __name__ == "__main__":
    test_rotary_embedding_time_strict()

# %%
