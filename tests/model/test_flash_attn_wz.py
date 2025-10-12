import torch
from math import sqrt
import matplotlib.pyplot as plt

from src.models.transformer.attentions import FlashAttentionWrapper
from src.utils.utils import set_seed

def test_flash_attention_with_window_sizes():
    set_seed(42)
    B, L, H, D = 2, 12800, 4, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(B, L, H, D, device=device)
    k = torch.randn(B, L, H, D, device=device)
    v = torch.randn(B, L, H, D, device=device)
    mask = torch.ones(B, L, device=device)

    attention = FlashAttentionWrapper(attn_dropout=0.0, precision="fp16").to(device)

    window_sizes = [(-1, -1), (6400, 6400), (6400, 0), (3200, 3200), (3200, 1), (800, 800), (12800, 0), (0, 12800)]
    outputs = {}
    mem_usages = {}

    print("📊 FlashAttention Test Results:")
    for ws in window_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        out, _ = attention(q, k, v, non_pad_mask=mask, window_size=ws, causal=True)
        outputs[ws] = out

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # Convert to MB
        mem_usages[ws] = peak_memory

        print(f"[window_size={ws}] → mean={out.mean().item():.4f}, std={out.std().item():.4f}, peak_mem={peak_memory:.2f} MB")

    # Error comparison
    print("\n📈 Mean Absolute Difference with Global Attention:")
    ref = outputs[(-1, -1)]
    for ws, out in outputs.items():
        if ws == (-1, -1):
            continue
        diff = (ref - out).abs().mean()
        print(f"window_size={ws} → mean abs diff = {diff.item():.6f}")


if __name__ == "__main__":
    test_flash_attention_with_window_sizes()
