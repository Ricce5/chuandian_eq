import torch
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime

def test_rotary_embedding_time_strict():
    device = "cuda"
    dtype = torch.float16

    B, L, H, D = 1, 5, 2, 8  # batch, seq_len, num_heads, head_dim
    num_heads = H

    print("🧪 Test config: B={}, L={}, H={}, D={}".format(B, L, H, D))

    # 模拟 qkv 输入 [B, L, 3, H, D]
    qkv_full = torch.randn(B, L, 3, H, D, device=device, dtype=dtype)
    times = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)

    # ===================== 1️⃣ 离散模式一次性计算 =====================
    rope_full = RotaryEmbeddingTime(D, interleaved=False).to(device)
    out_full = rope_full(qkv_full.clone(), num_heads_q=num_heads)
    print("✅ out_full:", out_full.shape)

    # ===================== 2️⃣ 时间模式一次性计算 =====================
    rope_time = RotaryEmbeddingTime(D, interleaved=False).to(device)
    out_time = rope_time(qkv_full.clone(), num_heads_q=num_heads, times=times)
    print("✅ out_time:", out_time.shape)

    max_diff_time = (out_full - out_time).abs().max().item()
    print(f"🔍 离散 vs 时间 max_diff: {max_diff_time:.6f}")
    assert max_diff_time < 1e-3, "时间模式行为与离散不一致"

    # ===================== 3️⃣ 模拟 KV-Cache 按步计算 =====================
    rope_cache = RotaryEmbeddingTime(D, interleaved=False).to(device)
    # 提前缓存完整时间序列，避免每步重算、覆盖
    rope_cache._update_cos_sin_cache_with_times(times, device=device, dtype=dtype)

    outs = []
    for i in range(L):
        qkv_step = qkv_full[:, i:i+1]  # 单步
        out_step = rope_cache(qkv_step, num_heads_q=num_heads, seqlen_offset=i)
        outs.append(out_step)

    out_cache = torch.cat(outs, dim=1)
    print("✅ out_cache:", out_cache.shape)

    max_diff_cache = (out_full - out_cache).abs().max().item()
    print(f"🔍 离散 vs KV Cache max_diff: {max_diff_cache:.6f}")
    assert max_diff_cache < 1e-3, "KV Cache行为与离散不一致"

    # ========== 结果展示 ==========
    print("🎉 所有模式一致，测试通过！")
    print(f"sum(out_full): {out_full.sum().item():.6f}")
    print(f"sum(out_time): {out_time.sum().item():.6f}")
    print(f"sum(out_cache): {out_cache.sum().item():.6f}")

if __name__ == "__main__":
    test_rotary_embedding_time_strict()
