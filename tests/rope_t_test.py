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
# %%
import torch
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime

def test_rotary_embedding_time_strict():
    device = "cuda"
    dtype = torch.float16

    B, L, H, D = 1, 5, 2, 8  # batch, seq_len, num_heads, head_dim
    num_heads = H

    qkv_full = torch.randn(B, L, 3, H, D, device=device, dtype=dtype)
    times = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1, L)

    # 1️⃣ 标准模式（一次性）
    rope_full = RotaryEmbeddingOrig(D, interleaved=False).to(device)
    out_full = rope_full(qkv_full.clone(), num_heads_q=num_heads)
    print("out_full:", out_full.shape)

    # 2️⃣ 按步 KV Cache 模拟
    rope_cache = RotaryEmbeddingTime(D, interleaved=False).to(device)
    outs = []
    for i in range(L):
        qkv_step = qkv_full[:, i:i+1]
        time = times[:, i:i+1]  # (1, 1)
        out_step = rope_cache(qkv_step.clone(), num_heads_q=num_heads, times=time)
        outs.append(out_step)
    out_cache = torch.cat(outs, dim=1)
    print("out_cache:", out_cache.shape)

    # 3️⃣ 连续时间模式
    rope_time = RotaryEmbeddingTime(D, interleaved=False).to(device)
    out_time = rope_time(qkv_full.clone(), num_heads_q=num_heads, times=times)
    print("out_time:", out_time.shape)

    # 📊 误差比较
    max_diff_cache = (out_full - out_cache).abs().max().item()
    max_diff_time = (out_full - out_time).abs().max().item()
    print(f"Diff vs KV cache: {max_diff_cache:.6f}")
    print(f"Diff vs time mode: {max_diff_time:.6f}")

    assert max_diff_cache < 1e-2, "与 KV Cache 模式不一致"
    assert max_diff_time < 1e-2, "与时间模式不一致"
    print("✅ RotaryEmbeddingTime 在三种推理场景中表现一致")

if __name__ == "__main__":
    test_rotary_embedding_time_strict()

# %%
