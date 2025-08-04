# %%
import torch
from config.config_loader import load_args_from_yaml
args = load_args_from_yaml('./config/mixer_tpp.yaml')
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# %%
print(args)
# %
# %%
config = args.mixer_model_config
# %%
from src.models.mamba.mixer_seq import MixerModel 

model = MixerModel(**config, device=device, dtype=dtype).to(device)
# %%
print(config)
# %%
input_dim = config.get("input_dim") 
d_model = config.get("d_model") 
batch_size = 2
seq_len = 16
input_features = torch.randn(batch_size, seq_len, input_dim, device=device)

with torch.no_grad():
    output = model(input_features)  # shape: (B, L, d_model)

print("Output shape:", output.shape)
assert output.shape == (batch_size, seq_len, d_model), "输出维度不正确"

# %%
# 测试 inference cache 分配
cache = model.allocate_inference_cache(batch_size=batch_size, max_seqlen=seq_len)
print("Inference cache keys:", cache.keys())

# %%
# 测试单步输入（模拟 autoregressive 推理）
from mamba_ssm.utils.generation import InferenceParams

inference_params = InferenceParams(
    max_seqlen=seq_len,
    max_batch_size=batch_size,
    key_value_memory_dict=model.allocate_inference_cache(batch_size, seq_len),
)
generated = []
for i in range(seq_len):
    current_input = torch.randn(batch_size, 1, input_dim, device=device).float().contiguous()
    model = model.float()
    inference_params.seqlen_offset = i

    with torch.no_grad():
        output = model(current_input, inference_params=inference_params)

    generated.append(output[:, -1, :])  # 取最后一个 token 的输出

final_output = torch.stack(generated, dim=1)  # shape: (batch, seq_len, d_model)
print("Final autoregressive output shape:", final_output.shape)

# %%
# 重新设置输入，确保一致
input_features = torch.randn(batch_size, seq_len, input_dim, device=device)

# 一次性输出
with torch.no_grad():
    full_output = model(input_features)

# 增量输出
inference_params = InferenceParams(
    max_seqlen=seq_len,
    max_batch_size=batch_size,
    key_value_memory_dict=model.allocate_inference_cache(batch_size, seq_len),
)

generated = []
for i in range(seq_len):
    current_input = input_features[:, i:i+1, :]  # 第 i 步
    inference_params.seqlen_offset = i

    with torch.no_grad():
        output = model(current_input.contiguous(), inference_params=inference_params)

    generated.append(output[:, -1, :])  # 当前 token 的输出

incremental_output = torch.stack(generated, dim=1)

# 比较一致性
max_diff = torch.max(torch.abs(full_output - incremental_output)).item()
print("Max difference between full and incremental output:", max_diff)

assert torch.allclose(full_output, incremental_output, atol=1e-3), "增量与一次性输出不一致！"
