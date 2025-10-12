import torch

# from src.models.mha.mha import MHA 
from mamba_ssm.modules.mha import MHA
from mamba_ssm.utils.generation import InferenceParams
B, S, D, H = 2, 6, 64, 4  # batch, seqlen, embed_dim, num_heads
T = torch.arange(S).unsqueeze(0).expand(B, -1).float()  # time indices
from src.utils.utils import set_seed
set_seed(42)


# convert to float16
model = MHA(
    embed_dim=D,
    num_heads=H,
    # rotary_emb_dim=D // H // 2,
    layer_idx=0,
    causal=True,
).to(dtype=torch.float16, device="cuda")


x = torch.randn(B, S, D, dtype=torch.float16, device="cuda")

out_full = model(x, inference_params=None)

inference_params = InferenceParams(
    max_seqlen=S,
    max_batch_size=B,
    key_value_memory_dict={
        model.layer_idx: model.allocate_inference_cache(batch_size=B, max_seqlen=S)
    },
)
print(out_full.shape) 
outputs = []
for t in range(S):
    token = x[:, t:t+1, :] 
    out = model(token,  inference_params=inference_params)
    inference_params.seqlen_offset += 1
    outputs.append(out)

output_stepwise = torch.cat(outputs, dim=1)

diff = (out_full - output_stepwise).abs().max()  
print(f"full {out_full[0,:,0]}")
print(f"step {output_stepwise[0,:,0]}  ")
print("Maximum difference between full forward pass and stepwise forward pass:", diff.item())
assert diff < 1e-2, "Mismatch: Full forward pass and stepwise forward pass results are inconsistent"
print("✅ Success: Full forward pass and stepwise forward pass results are consistent!")

