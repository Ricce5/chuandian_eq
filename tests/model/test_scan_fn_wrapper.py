import torch
from einops import rearrange
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SelectiveScanWrapper(nn.Module):
    def __init__(self, d_model, d_state, device, **kwargs):
        """
        d_model: Feature dimension of the model
        d_state: Dimension of the state space
        device: Current device, either CPU or CUDA
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.device = device

        # A: (d_model, d_state)
        self.A = torch.zeros(d_model, d_state, device=device)  # A is initialized to zero, adjust as needed
        # B and C are learnable parameters, we create trainable tensors for them
        self.B = nn.Parameter(torch.randn(d_model, d_state, device=device))
        self.C = nn.Parameter(torch.randn(d_model, d_state, device=device))
        # D is a tensor with the same dimension as d_model, initialized to zero
        self.D = torch.zeros(d_model, device=device)  # D is initialized to zero, adjust as needed
        # delta_bias parameter
        self.delta_bias = None  # nn.Parameter(torch.randn(d_model, device=device))  # Trainable bias (optional)

    def forward(self, x, delta):
        """
        x: Input tensor with shape (batch, length, d_model)
        delta: Time step tensor with shape (batch, length, d_model)
        """
        # Rearrange input tensor from (B, L, D) to (B, D, L) to match selective_scan_fn requirements
        u = rearrange(x, 'b l d -> b d l')
        delta_rearranged = rearrange(delta, 'b l d -> b d l')

        y_out = selective_scan_fn(
            u=u, 
            delta=delta_rearranged, 
            A=self.A, 
            B=self.B, 
            C=self.C, 
            D=self.D, 
            delta_bias=self.delta_bias
        )
        
        # Rearrange output from (B, D, L) back to (B, L, D)
        y_out = rearrange(y_out, 'b d l -> b l d')    
        return y_out

# --- Usage Example ---
B_batch, L_len, D_model, D_state = 1, 10, 1, 16  # Example dimensions: 4, 10, 128, 16

ssm_wrapper = SelectiveScanWrapper(d_model=D_model, d_state=D_state, device=device).to(device)

x_input = torch.randn(B_batch, L_len, D_model, device=device)
# x_input = torch.zeros(B_batch, L_len, D_model, device=device)
delta_input = torch.randn(B_batch, L_len, D_model, device=device)

output = ssm_wrapper(x_input, delta_input) 

print(f"Input shape: {x_input.shape}")
print(f"Output shape: {output.shape}")
print(output)
