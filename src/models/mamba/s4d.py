"""Diagonal S4 (S4D) layer with FFT-kernel convolution.

This module follows the diagonal state-space parameterization from S4D and the
minimal implementation pattern in the official state-spaces/s4 repository, but
keeps the public tensor convention aligned with this project: `(batch, length,
d_model)` by default.

References:
    - Gu et al., "On the Parameterization and Initialization of Diagonal State
      Space Models", arXiv:2206.11893.
    - Gupta et al., "Diagonal State Spaces are as Effective as Structured State
      Spaces", arXiv:2203.14343.
    - https://github.com/state-spaces/s4
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.float64:
        return torch.complex128
    return torch.complex64


def _real_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype == torch.complex128:
        return torch.float64
    return torch.float32


def _fft_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _parameter_dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype in (torch.float64, torch.complex128):
        return torch.float64
    return torch.float32


def _make_activation(name: Optional[str]) -> nn.Module:
    if name is None or name.lower() in {"identity", "none"}:
        return nn.Identity()
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class S4DKernel(nn.Module):
    """Generate a depthwise S4D convolution kernel from diagonal SSM parameters.

    The continuous diagonal SSM uses one complex diagonal matrix per channel:

        A = -exp(log_A_real) + i * A_imag

    With zero-order-hold discretization, the length-L convolution kernel is
    computed as a Vandermonde product and then used by `S4D` through FFT
    convolution.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        A_real_init: float = 0.5,
        C_init_scale: float = 1.0,
        learnable_A_imag: bool = True,
        learnable_dt: bool = True,
        kernel_chunk_size: Optional[int] = None,
        lr: Optional[float] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_state < 2 or d_state % 2 != 0:
            raise ValueError("d_state must be an even integer >= 2 for complex-conjugate S4D pairs")
        if not 0 < dt_min < dt_max:
            raise ValueError("Expected 0 < dt_min < dt_max")
        if A_real_init <= 0:
            raise ValueError("A_real_init must be positive")

        factory_kwargs = {"device": device, "dtype": _parameter_dtype(dtype)}
        self.d_model = d_model
        self.d_state = d_state
        self.n_complex = d_state // 2
        self.kernel_chunk_size = kernel_chunk_size

        log_dt = torch.empty(d_model, **factory_kwargs).uniform_(math.log(dt_min), math.log(dt_max))
        C = torch.randn(d_model, self.n_complex, 2, **factory_kwargs)
        C = C * (C_init_scale / math.sqrt(self.n_complex))
        log_A_real = torch.full(
            (d_model, self.n_complex),
            math.log(A_real_init),
            **factory_kwargs,
        )
        A_imag = math.pi * torch.arange(self.n_complex, **factory_kwargs)
        A_imag = A_imag.unsqueeze(0).expand(d_model, -1).contiguous()

        self._register_tensor("log_dt", log_dt, trainable=learnable_dt, lr=lr)
        self.C = nn.Parameter(C)
        self._register_tensor("log_A_real", log_A_real, trainable=True, lr=lr)
        self._register_tensor("A_imag", A_imag, trainable=learnable_A_imag, lr=lr)

    def _register_tensor(
        self,
        name: str,
        tensor: Tensor,
        trainable: bool,
        lr: Optional[float],
    ) -> None:
        if trainable:
            parameter = nn.Parameter(tensor)
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            parameter._optim = optim
            parameter._no_weight_decay = True
            self.register_parameter(name, parameter)
        else:
            self.register_buffer(name, tensor)

    def _materialize_parameters(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        param_dtype = self.log_A_real.dtype
        complex_dtype = _complex_dtype(param_dtype)

        dt = torch.exp(self.log_dt).to(dtype=param_dtype)
        C = torch.view_as_complex(self.C.contiguous()).to(dtype=complex_dtype)
        A = -torch.exp(self.log_A_real).to(dtype=param_dtype)
        A = A.to(dtype=complex_dtype) + 1j * self.A_imag.to(dtype=complex_dtype)

        dtA = A * dt.unsqueeze(-1)
        B_bar = torch.expm1(dtA) / A
        C_bar = C * B_bar
        A_bar = torch.exp(dtA)
        return A, A_bar, C_bar, dtA

    def forward(self, length: int, dtype: Optional[torch.dtype] = None) -> Tensor:
        """Return the real-valued depthwise convolution kernel with shape `(D, L)`."""
        if length <= 0:
            raise ValueError("length must be positive")

        _, _, C_bar, dtA = self._materialize_parameters()
        kernel = self._vandermonde_kernel(C_bar, dtA, length)
        if dtype is not None:
            kernel = kernel.to(dtype=dtype)
        return kernel

    def discrete_parameters(self) -> Tuple[Tensor, Tensor]:
        """Return `(A_bar, C_bar)` for recurrent single-step execution."""
        _, A_bar, C_bar, _ = self._materialize_parameters()
        return A_bar, C_bar

    def _vandermonde_kernel(self, C_bar: Tensor, dtA: Tensor, length: int) -> Tensor:
        real_dtype = _real_dtype(C_bar.dtype)
        times = torch.arange(length, device=dtA.device, dtype=real_dtype)
        chunk_size = self.kernel_chunk_size

        if chunk_size is None or length <= chunk_size:
            powers = torch.exp(dtA.unsqueeze(-1) * times)
            return 2.0 * torch.einsum("dn,dnl->dl", C_bar, powers).real

        kernel = torch.empty(self.d_model, length, device=dtA.device, dtype=real_dtype)
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            powers = torch.exp(dtA.unsqueeze(-1) * times[start:end])
            kernel[:, start:end] = 2.0 * torch.einsum("dn,dnl->dl", C_bar, powers).real
        return kernel


class S4D(nn.Module):
    """S4D sequence layer using FFT convolution over an S4D-generated kernel.

    Args:
        d_model: Feature dimension.
        d_state: Diagonal state size. Must be even because states are stored as
            complex-conjugate pairs.
        transposed: If False, use `(B, L, D)` input/output like `scan_wrapper.py`.
            If True, use `(B, D, L)`.
        return_state: `forward(..., return_state=True)` computes the final
            recurrent state by the exact recurrent path after the FFT output.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.0,
        transposed: bool = False,
        activation: Optional[str] = "gelu",
        use_skip: bool = True,
        use_output_linear: bool = True,
        gated_output: bool = True,
        input_norm: bool = False,
        input_linear: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
        **kernel_kwargs,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.transposed = transposed
        self.use_skip = use_skip
        self.use_output_linear = use_output_linear
        self.gated_output = gated_output

        self.kernel = S4DKernel(d_model, d_state=d_state, device=device, dtype=dtype, **kernel_kwargs)
        if use_skip:
            self.D = nn.Parameter(torch.ones(d_model, **factory_kwargs))
            self.D._no_weight_decay = True
        else:
            self.register_parameter("D", None)

        self.input_norm = nn.LayerNorm(d_model, **factory_kwargs) if input_norm else None
        self.input_linear = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs) if input_linear else None
        self.activation = _make_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if use_output_linear:
            output_dim = 2 * d_model if gated_output else d_model
            layers = [nn.Conv1d(d_model, output_dim, kernel_size=1, bias=bias, **factory_kwargs)]
            if gated_output:
                layers.append(nn.GLU(dim=1))
            self.output_linear = nn.Sequential(*layers)
        else:
            self.output_linear = nn.Identity()

    def forward(
        self,
        x: Tensor,
        ssm_state: Optional[Tensor] = None,
        return_state: bool = False,
        return_kernel: bool = False,
    ):
        """Apply S4D to a sequence.

        Args:
            x: `(B, L, D)` by default, or `(B, D, L)` when `transposed=True`.
            ssm_state: Optional complex recurrent state with shape
                `(B, D, d_state // 2)`. When provided, the recurrent path is
                used so the initial state is respected.
            return_state: Return the final recurrent state.
            return_kernel: Also return the generated convolution kernel.
        """
        u = self._prepare_input(x)

        if ssm_state is None:
            y, kernel = self._forward_kernel(u)
            state_out = self._final_state_recurrent(u, None) if return_state else None
        else:
            y, state_out = self._forward_recurrent(u, ssm_state)
            kernel = None
            if return_kernel:
                kernel = self.kernel(u.size(-1), dtype=u.dtype)

        y = self._finalize_output(y)

        outputs = [self._restore_output(y)]
        if return_state:
            outputs.append(state_out)
        if return_kernel:
            outputs.append(kernel)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def allocate_inference_cache(self, batch_size: int, dtype=None, device=None) -> Tensor:
        """Allocate a zero recurrent state with shape `(B, D, d_state // 2)`."""
        reference = self.kernel.log_A_real
        device = reference.device if device is None else device
        dtype = reference.dtype if dtype is None else dtype
        return torch.zeros(
            batch_size,
            self.d_model,
            self.kernel.n_complex,
            device=device,
            dtype=_complex_dtype(dtype),
        )

    def step(self, x: Tensor, ssm_state: Tensor) -> Tuple[Tensor, Tensor]:
        """Run one recurrent S4D step and return `(output, next_state)`."""
        u = self._prepare_input(x)
        if u.size(-1) != 1:
            raise ValueError("step() expects a single-step input with sequence length 1")
        if ssm_state.shape[:2] != (u.size(0), self.d_model):
            raise ValueError(
                f"Expected state shape prefix ({u.size(0)}, {self.d_model}), got {ssm_state.shape}"
            )

        A_bar, C_bar = self.kernel.discrete_parameters()
        state = A_bar.unsqueeze(0) * ssm_state + u.squeeze(-1).unsqueeze(-1).to(A_bar.dtype)
        y = 2.0 * torch.einsum("dn,bdn->bd", C_bar, state).real
        if self.use_skip:
            y = y + u.squeeze(-1) * self.D.to(dtype=u.dtype).unsqueeze(0)
        y = y.unsqueeze(-1).to(dtype=u.dtype)
        y = self._finalize_output(y)
        return self._restore_output(y), state

    def _prepare_input(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected a 3D tensor, got shape {tuple(x.shape)}")
        if self.transposed:
            if x.size(1) != self.d_model:
                raise ValueError(f"Expected channel dimension {self.d_model}, got {x.size(1)}")
            x_bld = x.transpose(1, 2)
        else:
            if x.size(-1) != self.d_model:
                raise ValueError(f"Expected feature dimension {self.d_model}, got {x.size(-1)}")
            x_bld = x

        if self.input_norm is not None:
            x_bld = self.input_norm(x_bld)
        if self.input_linear is not None:
            x_bld = self.input_linear(x_bld)
        return x_bld.transpose(1, 2).contiguous()

    def _restore_output(self, y: Tensor) -> Tensor:
        if self.transposed:
            return y
        return y.transpose(1, 2).contiguous()

    def _forward_kernel(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        length = u.size(-1)
        kernel = self.kernel(length, dtype=u.dtype)
        conv_dtype = _fft_dtype(torch.promote_types(u.dtype, kernel.dtype))
        u_fft = torch.fft.rfft(u.to(dtype=conv_dtype), n=2 * length)
        k_fft = torch.fft.rfft(kernel.to(dtype=conv_dtype), n=2 * length)
        y = torch.fft.irfft(u_fft * k_fft.unsqueeze(0), n=2 * length)[..., :length]
        if self.use_skip:
            y = y + u.to(dtype=conv_dtype) * self.D.to(dtype=conv_dtype).view(1, -1, 1)
        return y.to(dtype=u.dtype), kernel

    def _forward_recurrent(self, u: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        outputs = []
        next_state = state
        for t in range(u.size(-1)):
            y_t, next_state = self._step_prepared(u[:, :, t], next_state)
            outputs.append(y_t)
        return torch.stack(outputs, dim=-1).to(dtype=u.dtype), next_state

    def _step_prepared(self, u_t: Tensor, state: Tensor) -> Tuple[Tensor, Tensor]:
        A_bar, C_bar = self.kernel.discrete_parameters()
        state = A_bar.unsqueeze(0) * state + u_t.unsqueeze(-1).to(A_bar.dtype)
        y_t = 2.0 * torch.einsum("dn,bdn->bd", C_bar, state).real
        if self.use_skip:
            y_t = y_t + u_t * self.D.to(dtype=u_t.dtype).unsqueeze(0)
        return y_t, state

    def _final_state_recurrent(self, u: Tensor, state: Optional[Tensor]) -> Tensor:
        if state is None:
            state = self.allocate_inference_cache(u.size(0), dtype=u.dtype, device=u.device)
        _, state = self._forward_recurrent(u, state)
        return state

    def _finalize_output(self, y: Tensor) -> Tensor:
        y = self.activation(y)
        y = self.dropout(y)
        return self.output_linear(y)


S4DWrapper = S4D
