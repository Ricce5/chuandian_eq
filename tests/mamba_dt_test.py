import torch
from src.models.mamba.mamba_time import Mamba_dt  # 替换为你的 Mamba_dt 实现路径
from mamba_ssm.utils.generation import InferenceParams

# 设置随机种子
torch.manual_seed(42)

# 配置参数
d_model = 64
batch_size = 2
seq_len = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
model = Mamba_dt(
    d_model=d_model,
    d_state=16,
    d_conv=3,
    expand=2,
    dt_rank="auto",
    layer_idx=1,  # 一定要指定 layer_idx
).to(device)
model.eval()

# 构造输入数据
input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
dt_input = torch.rand(batch_size, seq_len, device=device) * 0.1 + 0.01  # (B, L)，模拟 Δt

# 一次性前向传播
with torch.no_grad():
    output_full = model(input_tensor, dt_input=dt_input)  # 传入 dt_input

# 设置 InferenceParams 缓存
inference_params = InferenceParams(
    max_seqlen=seq_len,
    max_batch_size=batch_size,
    key_value_memory_dict={
        model.layer_idx: model.allocate_inference_cache(batch_size=batch_size, max_seqlen=seq_len)
    },
)
model._get_states_from_cache(inference_params, batch_size)

# 增量推理（模拟 autoregressive decoding）
with torch.no_grad():
    outputs = []
    for t in range(seq_len):
        token = input_tensor[:, t:t+1, :]       # (B, 1, D)
        dt_step = dt_input[:, t:t+1]                # (B, 1)
        out = model(token, inference_params=inference_params, dt_input=dt_step)
        inference_params.seqlen_offset += 1
        outputs.append(out)
    output_stepwise = torch.cat(outputs, dim=1)  # (B, L, D)

# 比较差异
diff = (output_full - output_stepwise).abs().max()
print("Max difference between full and stepwise forward:", diff.item())
assert diff < 1e-2, "Mismatch between full and stepwise forward"
print("✅ Success: forward() and step() outputs are consistent!")
