import torch
import src
from src.models.transformer.encoders import Encoder_Conv  # 替换为实际 import 路径

def test_Encoder_Conv():
    # === 配置参数 ===
    batch_size = 4
    seq_len = 10
    d_model = 32
    d_inner = 64
    n_layers = 2
    n_head = 4
    d_k = d_v = 8
    dim = 16         # 输入的事件属性维度
    emb_n_layer = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 构建模型 ===
    model = Encoder_Conv(
        d_model=d_model,
        d_inner=d_inner,
        n_layers=n_layers,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=0.1,
        device=device,
        attn_type="flash",  # 或者你支持的其它类型
        dim=dim,
        emb_n_layer=emb_n_layer
    ).to(device)

    # === 构建输入数据 ===
    event_mark = torch.randn(batch_size, seq_len, dim).to(device)  # 假设连续属性事件
    event_time = torch.linspace(0, 10, steps=seq_len).repeat(batch_size, 1).to(device)  # shape: (B, T)

    # mask: 1 表示有效，0 表示 padding
    non_pad_mask = torch.ones(batch_size, seq_len, 1).to(device)  # 全部有效，形状 (B, T, 1)

    # attn_mask（可选）
    attn_mask = None

    # === 执行前向传播 ===
    with torch.no_grad():
        output = model(
            event_mark=event_mark,
            event_time=event_time,
            non_pad_mask=non_pad_mask,
            attn_mask=attn_mask,
            caches=None
        )
    
    print("✅ Forward pass successful.")
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, seq_len, d_model)
    assert not torch.isnan(output).any(), "❌ Output contains NaNs"
    print("🎉 Encoder_Conv passed the basic test.")

if __name__ == "__main__":
    test_Encoder_Conv()
