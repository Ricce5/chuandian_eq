import torch

from src.models.mha.mha_time import MHATime  
from mamba_ssm.modules.mha import MHA
from mamba_ssm.utils.generation import InferenceParams
B, S, D, H = 2, 600, 64, 4  # batch size, sequence length, embedding dimension, number of heads
T = torch.arange(S).unsqueeze(0).expand(B, -1).float()  # time indices
from src.utils.utils import set_seed
set_seed(42)

# Create the model and convert it to float16
model = MHATime(
    embed_dim=D,
    num_heads=H,
    rotary_emb_dim=D // H // 2,
    layer_idx=0,
    causal=True,
    rotary_emb_scale_base=512,
).to(dtype=torch.float16, device="cuda")

x = torch.randn(B, S, D, dtype=torch.float16, device="cuda")
T = T.to(dtype=torch.float16, device="cuda")  # <== Time indices also need the same type
inference_params1 = InferenceParams(
    max_seqlen=S,
    max_batch_size=B,
    key_value_memory_dict={
        model.layer_idx: model.allocate_inference_cache(batch_size=B, max_seqlen=S)
    },)

out_full = model(x, times=T, inference_params=inference_params1)

inference_params2 = InferenceParams(
    max_seqlen=S,
    max_batch_size=B,
    key_value_memory_dict={
        model.layer_idx: model.allocate_inference_cache(batch_size=B, max_seqlen=S)
    },)
print(out_full.shape)  # Should be (B, S, D)

outputs = []
for t in range(S):
    token = x[:, t:t+1, :]  # Token at the current time step
    dt_step = T[:, t:t+1]   # Time index at the current time step
    out = model(token, times=dt_step, inference_params=inference_params2)
    inference_params2.seqlen_offset += 1  # Update sequence offset
    outputs.append(out)

# Concatenate the outputs from stepwise inference
output_stepwise = torch.cat(outputs, dim=1)

# Compare the difference between full forward pass and stepwise forward pass
diff = (out_full - output_stepwise).abs().max()  # Maximum difference
print(f"full {out_full[0,:,0]}")
print(f"step {output_stepwise[0,:,0]}  ")
print("Maximum difference between full forward pass and stepwise forward pass:", diff.item())
assert diff < 1e-4, "Results of full forward pass and stepwise forward pass are inconsistent"
print("✅ Success: Results of full forward pass and stepwise forward pass are consistent!")
