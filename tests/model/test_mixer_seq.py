# %%
import torch
from src.models.mamba.mixer_seq import MixerModel  

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

d_model = 32
n_layer = 1
d_intermediate = 0
input_dim = 64  # Simulating embedding output, not necessarily equal to d_model

# Instantiate MixerModel
model = MixerModel(
    d_model=d_model,
    n_layer=n_layer,
    d_intermediate=d_intermediate,
    input_dim=input_dim,
    ssm_cfg={
        "layer": "Mamba2",           # Must exist, used to select Mamba2
        "d_state": 32,              # State dimension
        "d_conv": 4,                # Convolution kernel size
        "conv_init": None,          # Convolution initialization method
        "expand": 2,                # Expansion factor (controls intermediate channels)
        "headdim": 8,               # Dimension per head
        "d_ssm": None,              # If set, applies SSM to partial dimensions only
        "ngroups": 1,               # Number of groups (usually for group norm/conv)
        "A_init_range": (1, 16),    # Initialization range
        "D_has_hdim": False,        # Whether D term uses headdim
        "rmsnorm": True,            # Whether to enable RMSNorm internally
        "norm_before_gate": False,  # Whether to normalize before Gated MLP
        "dt_min": 0.001,
        "dt_max": 0.1,
        "dt_init_floor": 1e-4,
        "dt_limit": (0.0, float("inf")),
        "bias": False,
        "conv_bias": True,
        "chunk_size": 256,          # Chunk size for memory optimization
        "use_mem_eff_path": True,   # Whether to enable fused kernel path
        "sequence_parallel": True,  # Whether to enable sequence parallel
        # "process_group": None,    # Optional, used for distributed training
    },
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    device=device,
    dtype=dtype,
).to(device)
model.eval()

# %%
# Test forward inference
batch_size = 2
seq_len = 16000
input_features = torch.randn(batch_size, seq_len, input_dim, device=device)

with torch.no_grad():
    output = model(input_features)  # shape: (B, L, d_model)

print("Output shape:", output.shape)
assert output.shape == (batch_size, seq_len, d_model), "Output dimensions are incorrect"

# %%
# Test inference cache allocation
cache = model.allocate_inference_cache(batch_size=batch_size, max_seqlen=seq_len)
print("Inference cache keys:", cache.keys())

# %%
# Test single-step input (simulate autoregressive inference)
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

    generated.append(output[:, -1, :])  # Take the output of the last token

final_output = torch.stack(generated, dim=1)  # shape: (batch, seq_len, d_model)
print("Final autoregressive output shape:", final_output.shape)

# %%
# Reset input to ensure consistency
input_features = torch.randn(batch_size, seq_len, input_dim, device=device)

# Full output at once
with torch.no_grad():
    full_output = model(input_features)

# Incremental output
inference_params = InferenceParams(
    max_seqlen=seq_len,
    max_batch_size=batch_size,
    key_value_memory_dict=model.allocate_inference_cache(batch_size, seq_len),
)

generated = []
for i in range(seq_len):
    current_input = input_features[:, i:i+1, :]  
    inference_params.seqlen_offset = i

    with torch.no_grad():
        output = model(current_input.contiguous(), inference_params=inference_params)

    generated.append(output[:, -1, :])  

incremental_output = torch.stack(generated, dim=1)


max_diff = torch.max(torch.abs(full_output - incremental_output)).item()
print("Max difference between full and incremental output:", max_diff)

assert torch.allclose(full_output, incremental_output, atol=1e-3), "Incremental output does not match full output!"

