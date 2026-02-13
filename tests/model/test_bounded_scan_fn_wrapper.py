import torch
from einops import rearrange
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        self.output_range = output_range if output_range is not None else (0.5, 2)  # Default range [0.5, 2]

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

    def forward(self, x, delta, return_last_state=False):
        """
        x: Input tensor, shape (batch, length, d_model)
        delta: Time step tensor, shape (batch, length, d_model)
        """
        # Rearrange input tensor from (B, L, D) to (B, D, L) to match selective_scan_fn requirements
        u = rearrange(x, 'b l d -> b d l')
        delta_rearranged = rearrange(delta, 'b l d -> b d l')
        if return_last_state:
            y_out, ssm_state = selective_scan_fn(
                u=self._bounded_tanh(u),
                delta=delta_rearranged,
                A=self.A,
                B=self._bounded_tanh(self.B),  # Apply range constraints to B
                C=self._bounded_tanh(self.C),  # Apply range constraints to C
                D=self.D,
                delta_bias=self.delta_bias,
                return_last_state=return_last_state
            )
        else:
            y_out = selective_scan_fn(
                u=self._bounded_tanh(u),
                delta=delta_rearranged,
                A=self.A,
                B=self._bounded_tanh(self.B),  # Apply range constraints to B
                C=self._bounded_tanh(self.C),  # Apply range constraints to C
                D=self.D,
                delta_bias=self.delta_bias,
                return_last_state=return_last_state
            )

        # Rearrange output from (B, D, L) back to (B, L, D)
        y_out = rearrange(y_out, 'b d l -> b l d')    
        # Apply range constraints to the output
        y_out = self._bounded_tanh(y_out, *self.output_range)
        if return_last_state:
            return y_out, ssm_state
        else:
            return y_out

    def _bounded_tanh(self, input, min_val: float = -1, max_val: float = 1) -> torch.Tensor:
        """
        Use a tanh activation function with range constraints to limit the output within [min_val, max_val].
        """
        output = min_val + (max_val - min_val) * 0.5 * (torch.tanh(input) + 1)
        return output

    def _init_bounded_tensor(self, d_model, d_state, min_val, max_val):
        """
        Initialize a tensor with specified range constraints and return the tensor.
        """
        tensor = torch.randn(d_model, d_state)  # Initialize with standard normal distribution
        # Limit tensor values within the range [min_val, max_val]
        return min_val + (max_val - min_val) * 0.5 * (torch.tanh(tensor) + 1)

# --- Usage Example ---
B_batch, L_len, D_model, D_state = 10, 1000, 4, 16  # Example parameters
B_range = (0.2, 1.0)  # Set range for parameter B [0.2, 1.0]
C_range = (-2.0, 2.0)  # Set range for parameter C [-2.0, 2.0]
ssm_wrapper = BoundedSelectiveScanWrapper(d_model=D_model, d_state=D_state, device=device, B_range=B_range, C_range=C_range).to(device)
x_input = torch.randn(B_batch, L_len, D_model, device=device) * 100  # Random input
delta_input = torch.abs(torch.randn(B_batch, L_len, D_model, device=device))  # Ensure delta_input is positive
output, ssm_state = ssm_wrapper(x_input, delta_input, return_last_state=True)

print(f"ssm_state {ssm_state.shape}")

# Print input and output shapes to confirm consistency
print(f"Input shape: {x_input.shape}")
print(f"Output shape: {output.shape}")
# print(f"Input data: {x_input}")
# print(f"Output data: {output}")

# Generate save path and folder
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.makedirs(output_dir, exist_ok=True)

# Assume x_input and output are the model's input and output
# x_input.shape = (B_batch, L_len, D_model)
# output.shape = (B_batch, L_len, D_model)

# Here we assume a single batch (B_batch = 1)
x_input_single = x_input[0].detach().cpu().numpy()  # Convert to numpy format for plotting
output_single = output[0].detach().cpu().numpy()

# Generate time step indices
time_steps = range(L_len)

# Plot input and output graphs
plt.figure(figsize=(10, 6))

# Plot input graph
plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st subplot
plt.plot(time_steps, x_input_single, label="Input", color="blue")
plt.title("Input over Time")
plt.xlabel("Time Step")
plt.ylabel("Input Value")
plt.grid(True)
plt.legend()

# Plot output graph
plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd subplot
plt.plot(time_steps, output_single, label="Output", color="red")
plt.title("Output over Time")
plt.xlabel("Time Step")
plt.ylabel("Output Value")
plt.grid(True)
plt.legend()


plt.savefig(os.path.join(output_dir, "input_output_plot.png"))

plt.tight_layout()
plt.show()
