# %%
import torch
from src.models.mamba.mixer_seq import MixerModel  # 替换为你的 MixerModel 实现路径

# 配置参数
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

d_model = 32
n_layer = 1
d_intermediate = 0
input_dim = 64  # 模拟embedding输出，不一定要等于d_model

# 实例化 MixerModel
model = MixerModel(
    d_model=d_model,
    n_layer=n_layer,
    d_intermediate=d_intermediate,
    input_dim=input_dim,
    ssm_cfg={
        "layer": "Mamba2",           # 必须存在，用于选择使用 Mamba2
        "d_state": 32,              # 状态维度
        "d_conv": 4,                 # 卷积核大小
        "conv_init": None,          # 卷积初始化方式
        "expand": 2,                # 扩展倍数（控制中间通道）
        "headdim": 8,              # 每头的维度
        "d_ssm": None,              # 如果设置，只对部分维度应用SSM
        "ngroups": 1,               # 分组数（通常用于group norm/conv）
        "A_init_range": (1, 16),    # 初始化范围
        "D_has_hdim": False,        # D项是否使用 headdim
        "rmsnorm": True,            # 是否内部启用 RMSNorm
        "norm_before_gate": False,  # 是否在 Gated MLP 之前归一化
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init_floor": 1e-4,
        "dt_limit": (0.0, float("inf")),
        "bias": False,
        "conv_bias": True,
        "chunk_size": 256,          # 用于内存优化的分块大小
        "use_mem_eff_path": True,   # 是否启用 fused kernel 路径
        "sequence_parallel": True,  # 是否启用 sequence parallel
        # "process_group": None,    # 可选，用于分布式训练
    },
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    device=device,
    dtype=dtype,
).to(device)
model.eval()

# %%
# 测试前向推理
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
