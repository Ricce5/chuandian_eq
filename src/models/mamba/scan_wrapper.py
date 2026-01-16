import torch
from einops import rearrange
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
import torch.nn.functional as F

class BoundedSelectiveScanWrapper(nn.Module):
    def __init__(self, d_model, d_state, device, B_range=None, C_range=None, output_range=None, **kwargs):
        """
        d_model: Feature dimension of the model
        d_state: Dimension of the state space
        device: Current device, either CPU or CUDA
        B_range: Range for parameter B (min_val, max_val), optional
        C_range: Range for parameter C (min_val, max_val), optional
        output_range: Range for the output (min_val, max_val), optional
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.device = device
        self.B_range = B_range if B_range is not None else (0, 1)  # Default range [0, 1]
        self.C_range = C_range if C_range is not None else (1, 2)  # Default range [-1, 1]
        self.output_range = output_range 

        # A: (d_model, d_state)
        self.A = torch.zeros(d_model, d_state, device=device)
        # B, C are learnable parameters, initialize them and apply range constraints
        self.B = nn.Parameter(self._init_bounded_tensor(d_model, d_state, *self.B_range))  # Initialize B within range
        if self.C_range is None:
            self.C = torch.ones(d_model, d_state, device=device)
        else:
            self.C = nn.Parameter(self._init_bounded_tensor(d_model, d_state, *self.C_range))
        self.D = torch.zeros(d_model, device=device) 
        self.delta_bias = None  # Trainable bias, can be enabled if needed

    def forward(self, x, delta):
        """
        x: Input tensor, shape (batch, length, d_model)
        delta: Time step tensor, shape (batch, length, d_model)
        """
        # Rearrange input tensor from (B, L, D) to (B, D, L) to match selective_scan_fn requirements
        u = rearrange(x, 'b l d -> b d l')
        delta_rearranged = rearrange(delta, 'b l d -> b d l')

        y_out = selective_scan_fn(
            u=self._bounded_tanh(u), 
            delta=delta_rearranged, 
            A=self.A, 
            B=self._bounded_tanh(self.B),  # Apply range constraints to B
            C=self._bounded_tanh(self.C),  # Apply range constraints to C
            D=self.D, 
            delta_bias=self.delta_bias
        )
        
        # Rearrange output from (B, D, L) back to (B, L, D)
        y_out = rearrange(y_out, 'b d l -> b l d')    

        if self.output_range is not None:
            y_out = self._bounded_tanh(y_out, *self.output_range)

        return y_out

    def _bounded_tanh(self, input, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """
        Use a tanh activation function with range constraints to limit output within [min_val, max_val].
        """
        output = min_val + (max_val - min_val) * 0.5 * (torch.tanh(input) + 1)
        return output

    def _init_bounded_tensor(self, d_model, d_state, min_val, max_val):
        """
        Initialize a tensor with specified range constraints and return it.
        """
        tensor = torch.randn(d_model, d_state)  # Initialize with standard normal distribution
        # Limit tensor values within [min_val, max_val]
        return min_val + (max_val - min_val) * 0.5 * (torch.tanh(tensor) + 1)
    

class BoundedDiscreteSSM(nn.Module):
    def __init__(self, device, B_range=None, output_range=None, output_init=None):
        super(BoundedDiscreteSSM, self).__init__()
        self.device = device  # Ensure that device is correctly initialized
        self.B_range = B_range if B_range is not None else (0, 1)
        self.output_range = output_range if output_range is not None else (0.5, 2)
        self.kappa = self._init_bounded_tensor(*self.B_range)  # Use self.device here
        if output_init is None:
            output_init = torch.tensor(0.8, device=self.device)  # Default initial output
        else:
            output_init = torch.tensor(output_init, device=self.device)
        out_min, out_max = self.output_range
        self.state_init = self._inverse_bounded_tanh(output=output_init, min_val=out_min, max_val=out_max)  # Use self.device here

    def forward(self, x, delta_t=None, return_last_state=False):
        B, L, D = x.shape
        state = torch.zeros(B, L, D, device=x.device)
        if delta_t is not None and delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)  # Make it [B, L, 1]
            delta_t = delta_t.expand(-1, -1, D)  # Expand to [B, L, D]
        if delta_t is not None:
            sequence = self.kappa * torch.tanh(x) * delta_t  # [B, L, D]
        else:
            sequence = self.kappa * torch.tanh(x) 
        state = torch.cumsum(sequence, dim=1)  
        if self.state_init is not None:
            state += self.state_init
        out = self._bounded_tanh(state, *self.output_range)
        if return_last_state:
            return out, state
        else:
            return out

    def _bounded_tanh(self, input, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """
        Use a tanh activation function with range constraints to limit output within [min_val, max_val].
        """
        output = min_val + (max_val - min_val) * 0.5 * (torch.tanh(input) + 1)
        return output

    def _inverse_bounded_tanh(self, output, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """
        Compute the original input corresponding to the given output.
        """
        clamped_output = torch.clamp(output, min=min_val + 1e-6, max=max_val - 1e-6)

        term = (clamped_output - min_val) / (max_val - min_val) * 2 - 1
        input = torch.atanh(term)
        return input
    
    def _init_bounded_tensor(self, min_val, max_val):
        """
        Initialize a tensor with specified range constraints and return it.
        """
        tensor = torch.randn(1, device=self.device)  # Use self.device here
        return min_val + (max_val - min_val) * 0.5 * (torch.tanh(tensor) + 1)
    

class SelectiveScanWrapper(nn.Module):
    def __init__(self, d_model, d_state, device, B_positive: bool = False, C_positive: 
                 bool = False, D_positive: bool = False, use_D: bool = True,
                 A_max: float = 0,   
                   **kwargs):
        """
        d_model: Feature dimension of the model
        d_state: Dimension of the state space
        device: Current device, either CPU or CUDA
        B_positive/C_positive/D_positive: If True, enforce parameter > 0 via a softplus transform.
        use_D: If False, do not create or pass D to selective_scan_fn.
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.device = device
        self.A_max = A_max

        # A: (d_model, d_state)
        self.A = nn.Parameter(torch.zeros(d_model, d_state, device=device))

        # helper to create either a raw param (to be transformed) or a direct param / tensor
        def _maybe_create(name, shape, positive=False, init_zeros=False, default_zero=False, scale=1e-2):
            if positive:
                if init_zeros:
                    init = torch.zeros(shape, device=self.device)
                else:
                    init = scale * torch.randn(shape, device=self.device)
                setattr(self, f"raw_{name}", nn.Parameter(init))
            else:
                if default_zero:
                    setattr(self, name, torch.zeros(shape, device=self.device))
                else:
                    init = scale * torch.randn(shape, device=self.device)
                    setattr(self, name, nn.Parameter(init))

        # B and C: either direct trainable params or raw params to be transformed to positive
        _maybe_create("B", (d_model, d_state), positive=B_positive)
        _maybe_create("C", (d_model, d_state), positive=C_positive)

        # D: optionally created
        self.use_D = use_D
        if self.use_D:
            _maybe_create("D", d_model, positive=D_positive, init_zeros=True, default_zero=not D_positive)

        # delta_bias parameter (optional)
        self.delta_bias = None

        # store positivity flags for forward
        self.B_positive = B_positive
        self.C_positive = C_positive
        self.D_positive = D_positive

    def _positive_transform(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x)

    @property
    def _stable_A(self) -> torch.Tensor:
        A_clamp = self.A.clamp(max=self.A_max)
        return self.A + (A_clamp - self.A).detach()

    def _resolve(self, name):
        raw_attr = f"raw_{name}"
        if hasattr(self, raw_attr):
            return self._positive_transform(getattr(self, raw_attr))
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            return None

    def forward(self, x, delta, ssm_state: torch.Tensor = None, return_state: bool = False):
        """
        x: Input tensor with shape (batch, length, d_model)
        delta: Time step tensor with shape (batch, length, d_model)
        """
        A = self._stable_A
        B = self._resolve("B")
        C = self._resolve("C")
        D = self._resolve("D") if self.use_D else None

        u = rearrange(x, 'b l d -> b d l')
        delta_rearranged = rearrange(delta, 'b l d -> b d l')

        scan_kwargs = dict(
            u=u,
            delta=delta_rearranged,
            A=A,
            B=B,
            C=C,
            delta_bias=self.delta_bias,
        )
        if self.use_D:
            scan_kwargs['D'] = D

        y_out = selective_scan_fn(**scan_kwargs)

        state_out = None
        if return_state or (ssm_state is not None):
            # Recompute final state via incremental step path to ensure consistency with step()
            state_cache = ssm_state
            if state_cache is None:
                state_cache = self.allocate_inference_cache(batch_size=x.shape[0], dtype=x.dtype, device=x.device)
            for t in range(x.shape[1]):
                x_t = x[:, t:t+1, :]
                dt_t = delta[:, t:t+1, :]
                _, state_cache = self.step(x_t, dt_t, state_cache) # (B, D, 1, N)
            state_out = state_cache.squeeze(2) # (B, D, N)

        y_out = rearrange(y_out, 'b d l -> b l d')
        if return_state:
            return y_out, state_out 
        return y_out

    def allocate_inference_cache(self, batch_size: int, dtype=None, device=None):
        """Allocate SSM state cache for incremental decoding."""
        device = self.A.device if device is None else device
        dtype = self.A.dtype if dtype is None else dtype
        return torch.zeros(batch_size, self.d_model, 1, self.d_state, device=device, dtype=dtype)

    def step(self, x, delta, ssm_state: torch.Tensor):
        """Single-step update that mirrors selective_scan_fn for incremental inference."""
        batch, seqlen, d_model = x.shape
        assert seqlen == 1, "step() expects a single-step input (seqlen=1)"
        assert d_model == self.d_model, f"Expected d_model={self.d_model}, got {d_model}"
        if ssm_state.dim() == 3:
            ssm_state = ssm_state.unsqueeze(2)
        if ssm_state.shape[0] == 1 and batch > 1:
            ssm_state = ssm_state.expand(batch, -1, -1, -1).contiguous()
        assert ssm_state.shape == (batch, self.d_model, 1, self.d_state), "ssm_state has incompatible shape"

        A = self._stable_A.to(dtype=ssm_state.dtype).unsqueeze(1)  # (D, 1, N)
        B_param = self._resolve("B").to(dtype=ssm_state.dtype)
        C_param = self._resolve("C").to(dtype=ssm_state.dtype)
        B = B_param.unsqueeze(0).expand(batch, -1, -1)  # (B, D, N)
        C = C_param.unsqueeze(0).expand(batch, -1, -1)  # (B, D, N)
        D_param = None
        if self.use_D:
            D_param = self._resolve("D").to(dtype=ssm_state.dtype).unsqueeze(-1)  # (D, 1)

        x_step = rearrange(x, 'b l d -> b d l')  # (B, D, 1)
        delta_step = rearrange(delta, 'b l d -> b d l')  # (B, D, 1)

        if self.delta_bias is None:
            dt_bias = torch.zeros(self.d_model, 1, device=delta_step.device, dtype=delta_step.dtype)
        else:
            dt_bias = self.delta_bias.to(device=delta_step.device, dtype=delta_step.dtype)

        y = selective_state_update(
            state=ssm_state,
            x=x_step,
            dt=delta_step,
            A=A,
            B=B,
            C=C,
            D=D_param,
            z=None,
            dt_bias=dt_bias,
            dt_softplus=False,
        )

        y_out = rearrange(y, 'b d l -> b l d')  # (B, 1, D)
        return y_out, ssm_state
