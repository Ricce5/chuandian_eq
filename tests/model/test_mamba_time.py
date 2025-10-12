import torch
from src.models.mamba.mamba_time import MambaTime  # Replace with the path to your Mamba_dt implementation
from mamba_ssm.utils.generation import InferenceParams

# Set random seed
torch.manual_seed(42)

# Configuration parameters
d_model = 64
batch_size = 4
seq_len = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = MambaTime(
    d_model=d_model,
    d_state=16,
    d_conv=3,
    expand=2,
    dt_rank="auto",
    layer_idx=1,  # Make sure to specify layer_idx
    use_conv=False,
).to(device)
model.eval()

# Construct input data
input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
inter_times = torch.rand(batch_size, seq_len, device=device) * 0.1 + 0.01  # (B, L), simulate Δt

# One-shot forward pass
with torch.no_grad():
    output_full = model(input_tensor, inter_times=inter_times)  # Pass inter_times

# Set InferenceParams cache
inference_params = InferenceParams(
    max_seqlen=seq_len,
    max_batch_size=batch_size,
    key_value_memory_dict={
        model.layer_idx: model.allocate_inference_cache(batch_size=batch_size, max_seqlen=seq_len)
    },
)
model._get_states_from_cache(inference_params, batch_size)

# Incremental inference (simulate autoregressive decoding)
with torch.no_grad():
    outputs = []
    for t in range(seq_len):
        token = input_tensor[:, t:t+1, :]       # (B, 1, D)
        dt_step = inter_times[:, t:t+1]        # (B, 1)
        out = model(token, inference_params=inference_params, inter_times=dt_step)
        inference_params.seqlen_offset += 1
        outputs.append(out)
    output_stepwise = torch.cat(outputs, dim=1)  # (B, L, D)

# Compare differences
diff = (output_full - output_stepwise).abs().max()
print("Max difference between full and stepwise forward:", diff.item())
assert diff < 1e-2, "Mismatch between full and stepwise forward"
print("✅ Success: forward() and step() outputs are consistent!")
