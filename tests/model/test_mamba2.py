import torch
from mamba_ssm import Mamba2
# from src.models.mamba.mamba2_time import Mamba2
model = Mamba2(
    d_model=512,  # Hidden layer dimension
    d_state=128,  # State space dimension
    d_conv=4,  # Convolution layer size
    ngroups=1,  # Number of groups
    expand=2,  # Expansion factor
    headdim=64,  # Head dimension
    rmsnorm=True,  # Whether to use RMSNorm
    chunk_size=256,  # Chunk size
    device='cuda',  # Use CUDA
    use_mem_eff_path=False
)
import torch

seq_lengths = [5, 10, 6, 8, 3, 7, 9, 5]
batch_size = len(seq_lengths)
seqlen = max(seq_lengths)  # Maximum sequence length in the batch

cu_seqlens = torch.cumsum(torch.tensor([0] + seq_lengths[:-1]), dim=0).to('cuda')
seq_idx = torch.zeros((batch_size, seqlen), dtype=torch.int32).to('cuda')

for i, length in enumerate(seq_lengths):
    seq_idx[i, :length] = i

print("cu_seqlens:", cu_seqlens)
print("seq_idx:", seq_idx)

u = torch.randn(batch_size, seqlen, 512).to('cuda')  # Example input tensor, maximum sequence length
inference_params = None  # Placeholder for inference parameters

model.eval()
with torch.no_grad():
    output = model(u, seqlen=None, seq_idx=seq_idx, cu_seqlens=cu_seqlens, inference_params=inference_params)

print(output.shape)
