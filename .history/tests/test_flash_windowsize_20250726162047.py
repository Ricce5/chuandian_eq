import torch
from math import sqrt
import matplotlib.pyplot as plt

from src.models.transformer.attentions import FlashAttentionWrapper
from src.utils.utils import set_seed

def test_flash_attention_with_window_sizes():
    set_seed(42)
    B, L, H, D = 2, 128, 4, 64
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q = torch.randn(B, L, H, D, device=device)
    k = torch.randn(B, L, H, D, device=device)
    v = torch.randn(B, L, H, D, device=device)
    mask = torch.ones(B, L, device=device)

    attention = FlashAttentionWrapper(attn_dropout=0.0, precision="fp16").to(device)

    window_sizes = [(-1, -1), (64, 64), (64, 0), (32, 32), (32, 1), (8, 8), (128, 0), (0, 128)]
    outputs = {}
    mem_usages = {}

    print("📊 FlashAttention 测试结果：")
    for ws in window_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        out, _ = attention(q, k, v, non_pad_mask=mask, window_size=ws, causal=True)
        outputs[ws] = out

        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # 转为 MB
        mem_usages[ws] = peak_memory

        print(f"[window_size={ws}] → mean={out.mean().item():.4f}, std={out.std().item():.4f}, peak_mem={peak_memory:.2f} MB")

    # 误差对比
    print("\n📈 与全局 attention 的平均绝对差:")
    ref = outputs[(-1, -1)]
    for ws, out in outputs.items():
        if ws == (-1, -1):
            continue
        diff = (ref - out).abs().mean()
        print(f"window_size={ws} → mean abs diff = {diff.item():.6f}")

    # 显存图可视化
    plt.figure(figsize=(10, 5))
    labels = [str(ws) for ws in mem_usages.keys()]
    values = list(mem_usages.values())
    bars = plt.bar(labels, values)
    plt.title("🧠 FlashAttention 显存使用峰值（MB）")
    plt.ylabel("Peak Memory (MB)")
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()

    # 在柱状图顶部显示具体值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.1f}", ha='center', va='bottom', fontsize=8)

    plt.show()


if __name__ == "__main__":
    test_flash_attention_with_window_sizes()
