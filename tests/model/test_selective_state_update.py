import torch
from mamba_ssm.ops.triton.selective_state_update import selective_state_update, selective_state_update_ref

# Set parameters
batch = 2
nheads = 2
dim = 4
dstate = 8
ngroups = 1  # Ensure nheads % ngroups == 0

# Construct minimal tensors, all of type float32
state = torch.randn(batch, nheads, dim, dstate, device='cuda')
x = torch.randn(batch, nheads, dim, device='cuda')
dt = torch.rand(batch, nheads, dim, device='cuda')  # Ensure non-negative
dt_bias = torch.zeros(nheads, dim, device='cuda')  # Ensure non-negative
A = torch.randn(nheads, dim, dstate, device='cuda')
B = torch.randn(batch, ngroups, dstate, device='cuda')
C = torch.randn(batch, ngroups, dstate, device='cuda')

# Optional parameter set to None
D = torch.randn(nheads, dim, device='cuda') if ngroups > 0 else None
z = None

# Do not use dt_softplus
dt_softplus = False

# Simulate function call
try:
    out = selective_state_update(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
    )
    print("✅ Test passed, no errors. Output shape:", out.shape)
    out_ref = selective_state_update_ref(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
    )
    diff = (out - out_ref).abs().max()
    print("Maximum difference:", diff.item())
    assert diff < 1e-5, "Output is inconsistent with the reference implementation!"
except Exception as e:
    print("❌ Error:", e)
