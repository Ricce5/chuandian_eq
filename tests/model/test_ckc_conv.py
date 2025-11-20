import torch
import src
from src.models.layers.ckconv import LocalConv  # Assume the above code is saved in your_module.py



# Construct input data
batch_size, seq_len, d_model = 2, 5, 1
embed_seq = torch.randn(batch_size, seq_len, d_model)  # Embedded sequence
embed_seq.requires_grad = True

# Irregular timestamps, e.g., in milliseconds
time_seq = torch.tensor([
    [0.0, 0.1, 0.4, 0.8, 1.5],
    [0.0, 0.2, 0.3, 1.0, 2.0]
],requires_grad=True)

# Mask (0 indicates valid, 1 indicates pad or invalid)
mask = torch.zeros(batch_size, seq_len).bool()  # All valid

model = LocalConv(
    d_model=d_model,          # Input feature dimension
    siren_hid=16,       # SIREN hidden layer width
    siren_hid_num=2,    # Number of SIREN layers
    num_channel=2,      # Number of output channels per position in the kernel
    horizon=[2, 4],     # Receptive fields for two convolution layers
    omega=30            # SIREN frequency hyperparameter
)
# Run the model
output = model(embed_seq, time_seq, mask)
print("Output shape:", output.shape)
print("Output:", output)
##
gradient_tensor = torch.ones_like(output) 
output.backward(gradient=gradient_tensor)
grad_time_seq = time_seq.grad
print("Gradient w.r.t. time_seq:", grad_time_seq.shape)
grad_embed_seq = embed_seq.grad
print("Gradient w.r.t. embed_seq:", grad_embed_seq.shape)
grad_embed_seq.squeeze(-1)+grad_time_seq