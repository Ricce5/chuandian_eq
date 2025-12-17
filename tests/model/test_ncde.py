import torch
from src.models.ncde.slcde import LinearCDEBlock
# ===== 1) 设备 =====
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 2) 造假数据 =====
batch_size = 4
seq_len = 80000
input_dim = 64          # 关键：要和 hidden_dim 一致
hidden_dim = 64

X = torch.randn(batch_size, seq_len, input_dim, device=device)

# ===== 3) 实例化 LinearCDEBlock =====
from torch import nn

block = LinearCDEBlock(
    input_dim=input_dim,
    hidden_dim=hidden_dim,
    init_std=1.0,
    sparsity=0.2,         # 只对非 diagonal 且非 diagonal_dense 的 vf_A 有效
    dropout_rate=0.1,
    block_size=1,
    use_glu=True,         # 可改 False
    diagonal=False,       # 可改 True（对角形式）
    diagonal_dense=False, # 可改 True（对角+末尾小dense块）
    fwht=False,           # True 时 hidden_dim 必须是 2 的幂；且你 hadamard_matrix 里写死 cuda
    rank=0,
).to(device)

# ===== 4) 前向测试 =====
block.eval()
with torch.no_grad():
    Y = block(X)

print("X shape:", X.shape)  # (B, T, input_dim)
print("Y shape:", Y.shape)  # (B, T, input_dim)

# ===== 5)（可选）反向 + 稀疏梯度mask示例 =====
block.train()
Y = block(X)
loss = Y.mean()
loss.backward()

# 保持被mask掉的权重梯度为0（如果你训练时想维持稀疏结构，需要每次 backward 后调用）
block.mask_grads()

# 然后再 optimizer.step()
