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

rope_orig = RotaryEmbeddingOrig(dim, scale_base=512).to(device)
rope_time = RotaryEmbeddingTime(dim, scale_base=512).to(device)

times = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)

qkv = torch.randn(1, seq_len, 3, 1, dim, device=device, dtype=dtype)

out_orig = rope_orig(qkv.clone())
out_time = rope_time(qkv.clone(), times=times)

diff = (out_orig - out_time).abs().max().item()
print("Max difference:", diff)

# assert diff < 1e-3, "Behavior does not match the original implementation"
# print("✅ RotaryEmbeddingTime matches the original implementation with integer indices")

# %%
import torch
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime

def test_rotary_embedding_time_strict():
    device = "cuda"
    dtype = torch.float16

    B, L, H, D = 1, 500, 2, 8  # batch, seq_len, num_heads, head_dim
    num_heads = H

    qkv_full = torch.randn(B, L, 3, H, D, device=device, dtype=dtype)
    times = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)  # (1, L)

    # 1️⃣ Standard mode (one-shot)
    rope_full = RotaryEmbeddingOrig(D, interleaved=False, scale_base=1024).to(device)
    out_full = rope_full(qkv_full.clone(), num_heads_q=num_heads)
    print("out_full:", out_full.shape)

    # 2️⃣ Step-by-step KV Cache simulation
    rope_cache = RotaryEmbeddingTime(D, interleaved=False, scale_base=1024, time_center=L//2).to(device)
    outs = []
    for i in range(L):
        qkv_step = qkv_full[:, i:i+1]
        time = times[:, i:i+1]  # (1, 1)
        out_step = rope_cache(qkv_step.clone(), num_heads_q=num_heads, times=time)
        outs.append(out_step)
    out_cache = torch.cat(outs, dim=1)
    print("out_cache:", out_cache.shape)

    # 3️⃣ Continuous time mode
    rope_time = RotaryEmbeddingTime(D, interleaved=False, scale_base=1024, time_center=L//2).to(device)
    out_time = rope_time(qkv_full.clone(), num_heads_q=num_heads, times=times)
    print("out_time:", out_time.shape)

    # 📊 Error comparison
    max_diff_cache = (out_full - out_cache).abs().max().item()
    max_diff_time = (out_full - out_time).abs().max().item()
    print(f"Diff vs KV cache: {max_diff_cache:.6f}")
    print(f"Diff vs time mode: {max_diff_time:.6f}")

    assert max_diff_cache < 1e-2, "Inconsistent with KV Cache mode"
    assert max_diff_time < 1e-2, "Inconsistent with time mode"
    print("✅ RotaryEmbeddingTime performs consistently across three inference scenarios")

if __name__ == "__main__":
    test_rotary_embedding_time_strict()

# %%
import torch
from flash_attn.layers.rotary import RotaryEmbedding as RotaryEmbeddingOrig
from src.models.mha.rotary_embedding_time import RotaryEmbeddingTime
from src.utils.utils import set_seed
set_seed(42)
rope_time = RotaryEmbeddingTime(dim, scale_base=512).to(device)
base = torch.arange(seq_len, device=device).float()
noise = torch.randn(2, seq_len, device=device) * 0.1  # Add some small noise
times = (base.unsqueeze(0) + noise).clamp(min=0.0)
# Simulate qkv
qkv = torch.randn(2, seq_len, 3, 1, dim, device=device, dtype=dtype)
# %%
out_time = rope_time(qkv.clone(), times=times)
# %%
