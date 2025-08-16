# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class MambaTime(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        layer_idx=None,
        device=None,
        dtype=None,
        eps=1e-5,
        use_conv: bool = True,
    ):

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_conv = use_conv
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if self.use_conv:
            self.conv1d = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            self.conv1d = None

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_input_proj = nn.Linear(1, self.d_inner, bias=False,**factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        
        nn.init.uniform_(self.dt_input_proj.weight, a=0.01, b=0.1)

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.eps = eps

    def _encode_external_dt(self, dt_input: Tensor, batch: int, seqlen: int, dtype, device) -> Tensor:
        """
        Encodes external dt input into shape (B, d_inner, L) using bounded learnable weights.
        """
        if dt_input.shape != (batch, seqlen):
            raise ValueError(f"Expected dt_input shape ({batch}, {seqlen}), got {dt_input.shape}")

        dt = dt_input.to(dtype=dtype, device=device).clamp_min(self.eps)  # Ensure positive dt
        weight = self._bounded_weight_tanh(self.dt_min, self.dt_max)      # (1, d_inner)
        dt = torch.matmul(dt.unsqueeze(-1), weight)                       # (B, L, d_inner)
        dt = dt.permute(0, 2, 1).contiguous()                             # (B, d_inner, L)
        return dt


    def _bounded_weight_tanh(self, min_val: float = 0.01, max_val: float = 1 ) -> Tensor:
        """
        Returns a bounded positive projection weight tensor in range [min_val, max_val].
        """
        raw_weight = self.dt_input_proj.weight.view(1, -1)  # (1, d_inner)
        bounded_weight = min_val + (max_val - min_val) * 0.5 * (torch.tanh(raw_weight) + 1)
        return bounded_weight  # (1, d_inner)



    def forward(self, hidden_states, inference_params=None, inter_times: Optional[Tensor] = None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inter_times is not None:
                assert inter_times.shape == (batch, seqlen), \
                    f"Expected inter_times shape ({batch}, {seqlen}), got {inter_times.shape}"
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state, inter_times)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        x, z = xz.chunk(2, dim=1)
        # Compute short convolution
        if self.use_conv:
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )
        else: 
            x = self.act(x)

        # We're careful here about the layout, to avoid extra transposes.
        # We want dt to have d as the slowest moving dimension
        # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # ...

        if  inter_times is not None:
            dt = self._encode_external_dt(inter_times, batch, seqlen, x.dtype, x.device)
            delta_bias = None
            delta_softplus = False  # Don't apply softplus again
        else:
            # Default case: dt learned from projection
            dt = self.dt_proj.weight @ dt.t()  # (d_inner, B*L)
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            delta_bias = self.dt_proj.bias.float()
            delta_softplus = True

        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        # print(f"x shape: {x.shape}")
        # print(f"dt shape: {dt.shape}")
        # print(f"A shape: {A.shape}")
        # print(f"B shape: {B.shape}")
        # print(f"C shape: {C.shape}")
        # print(f"D shape: {self.D.shape}")
        # print(f"z shape: {z.shape}")
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state, dt_input: Optional[Tensor] = None):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B, 2D)
        x, z = xz.chunk(2, dim=-1)  # (B, D)

        # Conv step
        if self.use_conv:
            if causal_conv1d_update is None:
                conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # (B, D, W)
                conv_state[:, :, -1] = x
                x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
                if self.conv1d.bias is not None:
                    x = x + self.conv1d.bias
                x = self.act(x).to(dtype=dtype)
            else:
                x = causal_conv1d_update(
                    x,
                    conv_state,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
        else:
            x = self.act(x).to(dtype=dtype)

        x_db = self.x_proj(x)  # (B, dt_rank + 2 * d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        if dt_input is not None:
            dt = self._encode_external_dt(dt_input, hidden_states.shape[0], 1, x.dtype, x.device).squeeze(-1)
            delta_softplus = False
        else:
            dt = F.linear(dt, self.dt_proj.weight)  # -> (B, d_inner)
            dt = dt + self.dt_proj.bias.to(dtype=dt.dtype)
            dt = F.softplus(dt)
            delta_softplus = True


        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))  # (B, d_inner, d_state)
            dB = torch.einsum("bd,bn->bdn", dt, B)             # (B, d_inner, d_state)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B, D)
        else:
            y = selective_state_update(
                ssm_state,
                x,
                dt,
                A,
                B,
                C,
                self.D,
                z=z,
                dt_bias=None if not delta_softplus else self.dt_proj.bias,
                dt_softplus=delta_softplus,
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state


    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        if self.use_conv:
            conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
            conv_state = torch.zeros(
                batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
            )
        else:
            conv_state = None

        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            if self.use_conv:
                conv_state = torch.zeros(
                    batch_size,
                    self.d_model * self.expand,
                    self.d_conv,
                    device=self.conv1d.weight.device,
                    dtype=self.conv1d.weight.dtype,
                )
            else:
                conv_state = None

            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                if conv_state is not None:
                    conv_state.zero_() 
                ssm_state.zero_()
        return conv_state, ssm_state
