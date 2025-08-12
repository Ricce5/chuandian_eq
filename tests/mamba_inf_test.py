# %%
import torch
from mamba_ssm import Mamba,Mamba2  
# from src.models.mamba.mamba2_time import Mamba2  
# 设置随机种子以保证可复现
torch.manual_seed(42)

# 初始化模型
d_model = 64
batch_size = 2
seq_len = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Mamba2(
    d_model=d_model,
    d_state=16,
    d_conv=3,
    expand=2,
    # dt_rank="auto",
    # use_fast_path=False
    use_mem_eff_path=False
).to(device)
model.eval()  # 禁用 dropout 等行为

# 构造测试输入
input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)

# %%
# 一次性前向传播
with torch.no_grad():
    output_full = model(input_tensor)

# %%
from mamba_ssm.utils.generation import InferenceParams
model.layer_idx = 1 
inference_params = InferenceParams(
    max_seqlen=seq_len,
    max_batch_size=batch_size,
    key_value_memory_dict={1: model.allocate_inference_cache(batch_size=batch_size, max_seqlen=seq_len)},
)
model._get_states_from_cache(inference_params, batch_size)
# %%
# 增量 step() 推理
with torch.no_grad():
    outputs = []
    for t in range(seq_len):
        token = input_tensor[:, t:t+1, :]  # shape: (B, 1, D)
        out = model.forward(token, inference_params=inference_params)
        inference_params.seqlen_offset += 1
        outputs.append(out)

    output_stepwise = torch.cat(outputs, dim=1)  # shape: (B, L, D)

# %%
# 比较两种方式是否一致
diff = (output_full - output_stepwise).abs().max()
print("Max difference between forward() and step():", diff.item())

# 判断是否基本一致（浮点误差容忍 1e-5）
assert diff < 1e-2, "Mismatch between step() and forward()"

print("✅ forward() and step() outputs are consistent!")
print("Output shape:", output_stepwise.shape)
# %%
