import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool
from torch.nn import Linear
from typing import List
from .SubLayers import CNNBlock
from typing import List, Type





class AttentionPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(AttentionPooling, self).__init__()
        self.attn_fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),                             # Gelu
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, mask=None):
        """
        x: Tensor of shape [batch_size, seq_len, input_dim]
        mask: Tensor of shape [batch_size, seq_len], 1 for valid, 0 for padding
        """
        # Compute attention scores
        scores = self.attn_fc(x).squeeze(-1)  # [batch_size, seq_len]

        # Mask padding positions (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Normalize scores with softmax
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, seq_len] 确保权重的和为1

        # Compute weighted sum
        output = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)  # [batch_size, input_dim]

        return output, attn_weights  # return both for optional inspection
