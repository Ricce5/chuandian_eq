import torch
from mamba_ssm import Mamba2
model = Mamba2(
    d_model=512,  # 隐藏层维度
    d_state=128,  # 状态空间维度
    d_conv=4,  # 卷积层大小
    ngroups=1,  # 分组数
    expand=2,  # 扩展因子
    headdim=64,  # 头维度
    rmsnorm=True,  # 是否使用 RMSNorm
    chunk_size=256,  # 分块大小
    device='cuda'  # 使用 CUDA
)
import torch

# 给定的序列长度
seq_lengths = [5, 10, 6, 8, 3, 7, 9, 5]
batch_size = len(seq_lengths)
seqlen = max(seq_lengths)  # 批次中的最大序列长度

# 计算 cu_seqlens：累积序列长度
cu_seqlens = torch.cumsum(torch.tensor([0] + seq_lengths[:-1]), dim=0).to('cuda')

# 创建一个新的 seq_idx，它的形状应该是 (batch_size, seqlen)
seq_idx = torch.zeros((batch_size, seqlen), dtype=torch.int32).to('cuda')

# 填充 seq_idx，每个序列中的元素都为该序列在批次中的索引
for i, length in enumerate(seq_lengths):
    seq_idx[i, :length] = i

# 打印 cu_seqlens 和 seq_idx 以验证
print("cu_seqlens:", cu_seqlens)
print("seq_idx:", seq_idx)

# 进行推理
u = torch.randn(batch_size, seqlen, 512).to('cuda')  # 示例输入张量，最大序列长度
inference_params = None  # 推理参数占位符

model.eval()
with torch.no_grad():
    output = model(u, seqlen=None, seq_idx=seq_idx, cu_seqlens=cu_seqlens, inference_params=inference_params)

print(output.shape)

