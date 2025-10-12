import torch
from src.models.mamba.mamba2_time import Mamba2Time  # 替换为你的 Mamba_dt 实现路径
from mamba_ssm.utils.generation import InferenceParams
torch.manual_seed(42)


d_model = 64
batch_size = 2
seq_len = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Mamba2Time(
    d_model=d_model,
    d_state=16,
    d_conv=3,
    expand=2,
    layer_idx=1,  # 一定要指定 layer_idx
).to(device)
model.eval()

input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)
inter_times = torch.rand(batch_size, seq_len, device=device) * 0.1 + 0.01  # (B, L)，模拟 Δt

with torch.no_grad():
    output_full = model(input_tensor, inter_times=inter_times)  # 传入 inter_times

# set up inference parameters
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
        dt_step = inter_times[:, t:t+1]                # (B, 1)
        out = model(token, inference_params=inference_params, inter_times=dt_step)
        inference_params.seqlen_offset += 1
        outputs.append(out)
    output_stepwise = torch.cat(outputs, dim=1)  # (B, L, D)

diff = (output_full - output_stepwise).abs().max()
print("Max difference between full and stepwise forward:", diff.item())
assert diff < 1e-2, "Mismatch between full and stepwise forward"
print("✅ Success: forward() and step() outputs are consistent!")
