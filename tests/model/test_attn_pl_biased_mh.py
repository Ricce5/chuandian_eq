import torch
import torch.nn as nn
from src.models.extractors.attn_time_biased_mh import TimeAwareAttnPoolMH

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

B, L, D = 2, 6, 8           # Batch size, sequence length, feature dimension
H, Dh = 4, 16               # Number of heads, hidden dimension
x = torch.randn(B, L, D, device=device)  # Input tensor

valid_len = torch.tensor([6, 4], device=device)  # Valid lengths for each sample
mask = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1) < valid_len.unsqueeze(1)  # [B, L], mask for valid positions

t = torch.linspace(0.2, 1.0, L, device=device).unsqueeze(0).repeat(B, 1)  # [B, L], event time
t[1, 4:] = 0.0  # Zero out invalid positions for the second sample

# ====== 1) Linear bias + concat output ======
pool_lin = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                               n_heads=H, agg="concat", device=device)
pooled_lin, alpha_lin = pool_lin(x, mask, extra_inputs={"event_time": t}, return_score=True)
print("linear/concat -> pooled:", pooled_lin.shape, " alpha:", alpha_lin.shape)
# Expected: pooled = [B, H*D] = [2, 32]; alpha = [B, H, L] = [2, 4, 6]

# Backpropagation test
loss = pooled_lin.mean()
loss.backward()
print("backward ok (linear)")

# ====== 2) Log bias + mean output ======
pool_log = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="log",
                               alpha0=10.0, n_heads=H, agg="mean", device=device)
pooled_log, alpha_log = pool_log(x, mask, extra_inputs={"event_time": t}, return_score=True)
print("log/mean -> pooled:", pooled_log.shape, " alpha:", alpha_log.shape)
# Expected: pooled = [B, D] = [2, 8]; alpha = [2, 4, 6]

# Backpropagation test
loss = (pooled_log**2).sum()
loss.backward()
print("backward ok (log)")

# ====== 3) Check if mask is effective: alpha for invalid positions should be close to 0 ======
with torch.no_grad():
    print("alpha (sample=1):\n", alpha_lin[1])  # [H, L]
    print("mask (sample=1):\n", mask[1].int())  # [L]

# ====== 4) New feature test: fuse last_token ======
from copy import deepcopy

# Construct a differentiable last_token (simulate the last token representation from an encoder)
last_token = torch.randn(B, D, device=device, requires_grad=True)

# 4.1 Residual addition + variance matching s (no LN for easier numerical verification)
pool_add = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                               n_heads=H, agg="mean",
                               fuse_mode="add", use_ln=False, use_var_scale=True,
                               device=device)
pooled_add, alpha_add = pool_add(x, mask, extra_inputs={"event_time": t},
                                 return_score=True, last_token=last_token)
print("fuse=add(use_var_scale=True) ->", pooled_add.shape)  # Expected [B, D]

# Numerical verification: manually compute s and check fused = lt + s*pt
with torch.no_grad():
    # Obtain pooled without fusion (reuse internal logic for consistency)
    pool_none = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                    n_heads=H, agg="mean",
                                    fuse_mode="none", device=device)
    # Copy weights to ensure consistent output
    pool_none.load_state_dict({k: v for k, v in pool_add.state_dict().items()
                               if "ln_" not in k and "fuse_" not in k and "gate_" not in k}, strict=False)
    pooled_plain = pool_none(x, mask, extra_inputs={"event_time": t}, return_score=False)  # [B, D]

    lt = last_token.detach()
    pt = pooled_plain.detach()
    var_lt = lt.var(dim=-1, unbiased=False, keepdim=True) + 1e-8
    var_pt = pt.var(dim=-1, unbiased=False, keepdim=True) + 1e-8
    s = torch.sqrt(var_lt / var_pt)  # [B, 1]
    fused_manual = lt + s * pt
    diff = (pooled_add.detach() - fused_manual).abs().max().item()
    print(f"[check add+scale] max|diff| = {diff:.6g}")

# Backpropagation test: check if gradients flow back to last_token
(pooled_add.sum()).backward(retain_graph=True)
print("grad on last_token (add):", last_token.grad.norm().item())
last_token.grad.zero_()

# 4.2 Concat fusion + linear projection back to d_model
pool_concat = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                  n_heads=H, agg="mean",
                                  fuse_mode="concat", repr_dim=D, use_ln=True,
                                  device=device)
pooled_concat, _ = pool_concat(x, mask, extra_inputs={"event_time": t},
                               return_score=True, last_token=last_token)
print("fuse=concat ->", pooled_concat.shape)  # Expected [B, D]
loss = pooled_concat.mean()
loss.backward()
print("backward ok (concat) | grad last_token:", last_token.grad.norm().item())
last_token.grad.zero_()

# 4.3 Gate-based fusion (sample-level adaptive)
pool_gate = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                n_heads=H, agg="mean",
                                fuse_mode="gate", use_ln=True,
                                device=device)
pooled_gate, _ = pool_gate(x, mask, extra_inputs={"event_time": t},
                           return_score=True, last_token=last_token)
print("fuse=gate ->", pooled_gate.shape)  # Expected [B, D]
loss = (pooled_gate**2).sum()
loss.backward()
print("backward ok (gate) | grad last_token:", last_token.grad.norm().item())
last_token.grad.zero_()

# 4.4 Fallback when last_token is not provided (should use the last valid token per sample)
pool_add_fb = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                  n_heads=H, agg="mean",
                                  fuse_mode="add", use_ln=True, use_var_scale=False,
                                  device=device)
pooled_fb = pool_add_fb(x, mask, extra_inputs={"event_time": t}, return_score=False)  # No last_token provided
print("fuse=add (fallback last_token) ->", pooled_fb.shape)  # Expected [B, D]

# 4.5 Compatibility with agg='concat': recommended to use fuse_mode='concat'/'gate' (not 'add' due to dimension mismatch)
pool_cat_heads = TimeAwareAttnPoolMH(d_model=D, d_hidden=Dh, bias_type="linear",
                                     n_heads=H, agg="concat",
                                     fuse_mode="concat", repr_dim=D, device=device)
pooled_cat_heads = pool_cat_heads(x, mask, extra_inputs={"event_time": t},
                                  return_score=False, last_token=last_token)
print("agg=concat + fuse=concat ->", pooled_cat_heads.shape)  # Expected [B, D]
