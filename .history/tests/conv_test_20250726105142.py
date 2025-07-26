import torch
import src
from src.models.layers.ckconv import LocalConv  # 假设上述代码保存在 your_module.py

# 配置模型
model = LocalConv(
    d_model=4,          # 输入特征维度
    siren_hid=16,       # SIREN 隐藏层宽度
    siren_hid_num=2,    # SIREN 层数
    num_channel=2,      # 每个位置的 kernel 输出通道数
    horizon=[2, 4],     # 两层卷积感受野
    omega=30            # SIREN 频率超参数
)

# 构造输入数据
batch_size, seq_len, d_model = 2, 5, 4
embed_seq = torch.randn(batch_size, seq_len, d_model)  # 嵌入序列

# 时间戳不规则，比如毫秒单位
time_seq = torch.tensor([
    [0.0, 0.1, 0.4, 0.8, 1.5],
    [0.0, 0.2, 0.3, 1.0, 2.0]
])

# 掩码 (0 表示可用，1 表示 pad 或无效)
mask = torch.zeros(batch_size, seq_len).bool()  # 全部有效

# 运行模型
output = model(embed_seq, time_seq, mask)
print("Output shape:", output.shape)
print("Output:", output)
