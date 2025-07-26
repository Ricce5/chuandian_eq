import torch
from math import sqrt
import src
from src.models.transformer.attentions import FlashAttentionWrapper
from src.utils.utils import set_seed
set_seed(42)
def test_flash_attention_with_window_sizes():
    B, L, H, D = 2, 128, 4, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(B, L, H, D, device=device)
    k = torch.randn(B, L, H, D, device=device)
    v = torch.randn(B, L, H, D, device=device)
    mask = torch.ones(B, L, device=device)

    attention = FlashAttentionWrapper(attn_dropout=0.0, precision="fp16").to(device)

    window_sizes = [(-1, -1), (64, 64), (32, 32), (8, 8)]
    outputs = {}

    for ws in window_sizes:
        out, _ = attention(q, k, v, non_pad_mask=mask, window_size=ws)
        outputs[ws] = out
        print(f"[window_size={ws}] → mean={out.mean().item():.4f}, std={out.std().item():.4f}")

    # 比较差异
    print("\n差异比较（与全局 attention 的平均绝对差）:")
    ref = outputs[(-1, -1)]
    for ws, out in outputs.items():
        if ws == (-1, -1):
            continue
        diff = (ref - out).abs().mean()
        print(f"window_size={ws} → mean abs diff = {diff.item():.6f}")


if __name__ == "__main__":
    test_flash_attention_with_window_sizes()