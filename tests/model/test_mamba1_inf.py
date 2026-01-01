# %%
import torch
from mamba_ssm import Mamba  
torch.manual_seed(42)

d_model = 64
batch_size = 2
seq_len = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Mamba(
    d_model=d_model,
    d_state=16,
    d_conv=3,
    expand=2,
    dt_rank="auto",
    use_fast_path=False
).to(device)
model.eval()  

input_tensor = torch.randn(batch_size, seq_len, d_model, device=device)

# %%
with torch.no_grad():
    output_full = model(input_tensor)

# %%
# Incremental step() inference
with torch.no_grad():
    conv_state, ssm_state = model.allocate_inference_cache(batch_size=batch_size, max_seqlen=seq_len)
    outputs = []
    for t in range(seq_len):
        token = input_tensor[:, t:t+1, :] # shape: (B, 1, D)
        out, conv_state, ssm_state = model.step(token, conv_state, ssm_state)

        print(f"{torch.sum(conv_state)}")
        outputs.append(out)

    output_stepwise = torch.cat(outputs, dim=1)  # shape: (B, L, D)

# %%
diff = (output_full - output_stepwise).abs().max()
print("Max difference between forward() and step():", diff.item())
assert diff < 1e-5, "Mismatch between step() and forward()"

print("✅ forward() and step() outputs are consistent!")
print("Output shape:", output_stepwise.shape)
# %%
model(input_tensor*0).sum()
# %%
model(input_tensor).sum()
# %
# %%
input_tensor_positive = torch.abs(input_tensor)
# %%
model(input_tensor_positive)
# %%
out, conv_state_new, ssm_state_new = model.step(token, conv_state, ssm_state)
# Check if the state is updated correctly
print( torch.equal(conv_state, conv_state_new))
print( torch.equal(ssm_state, ssm_state_new))
# %%
# %%
