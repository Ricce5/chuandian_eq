import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool
from torch.nn import Linear
from typing import List
from .SubLayers import CNNBlock


class CNN(nn.Module):
    def __init__(self, input_size, d_model, output_size=16):
        super(CNN, self).__init__()
        self.cnn_block1 = CNNBlock(c_in=input_size, c_out=d_model)
        self.cnn_block2 = CNNBlock(c_in=d_model, c_out=d_model * 2)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化层：对每个通道取平均
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d_model * 2, d_model * 2),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(d_model * 2, output_size)
        )
    
    def forward(self, x):
        B, N, L, F = x.size()
        
        # 确保输入序列长度足够长，避免池化导致长度为0
        if L < 16:
            raise ValueError("Input sequence length is too small for the given pool size and stride")
        
        # 将输入从 (B, N, L, F) 转换为 (B * N, L, F)，以适应 Conv1d
        x = x.view(B * N, L, F)
        
        # 通过 CNN 层处理
        x = self.cnn_block1(x)  # 长度除4
        x = self.cnn_block2(x)  #

        # # 使用全局平均池化层
        x = self.global_avg_pool(x.transpose(1,2))  # (B * N, C, 1)
        x = x.view(B*N, -1)
        x = self.fc(x)
        x = x.view(B, N, -1)
        return x

class MLP(nn.Module):
    def __init__(self, hidden_layers_width=[128, 64, 32], input_size=30, output_size=32, dropout_rate=0.2):
        super(MLP, self).__init__()
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_layers_width = hidden_layers_width
        self.dropout_rate = dropout_rate

        layers: List[nn.Module] = []
        self.layers_width = [self.input_size] + self.hidden_layers_width

        for i in range(len(self.layers_width) - 1):
            layers += [
                nn.Linear(self.layers_width[i], self.layers_width[i + 1]),
                nn.LayerNorm(self.layers_width[i + 1]),
                nn.GELU(),
                nn.Dropout(self.dropout_rate)
            ]

        layers += [nn.Linear(self.layers_width[-1], self.output_size)]
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=4, jk_mode='cat', dropout=0.6):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.jk_mode = jk_mode

        # 定义GCN层和线性投影
        self.gcn_layers = nn.ModuleList()
        self.res_conns = []
        self.lin_proj = nn.ModuleList()

        # 输入层
        self.gcn_layers.append(GCNConv(in_channels, hidden_channels))
        self.res_conns.append(False)
        self.lin_proj.append(Linear(in_channels, hidden_channels))

        # 中间层
        for _ in range(1, num_layers):
            self.gcn_layers.append(GCNConv(hidden_channels, hidden_channels))
            self.res_conns.append(True)
            self.lin_proj.append(Linear(hidden_channels, hidden_channels))

        # Jumping Knowledge机制
        self.jump = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)

        # 输出层
        if jk_mode == 'cat':
            self.out_lin = Linear(hidden_channels * num_layers, 1)
        else:
            self.out_lin = Linear(hidden_channels, 1)
        
        # self.reset_parameters()
    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.gcn_layers):
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.res_conns[i]:
                res_proj = self.lin_proj[i](residual)
                x = x + res_proj

            xs.append(x)

        x = self.jump(xs)
        x = global_mean_pool(x, batch=None)
        x = self.out_lin(x)
        x = torch.sigmoid(x)
        return x.squeeze()
    
    def reset_parameters(self):
        for conv in self.gcn_layers:
            conv.reset_parameters()  # GCNConv 内置了合适的初始化（一般是 glorot）
        for lin in self.lin_proj:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        if isinstance(self.out_lin, Linear):
            nn.init.xavier_uniform_(self.out_lin.weight)
            nn.init.zeros_(self.out_lin.bias)
        if hasattr(self.jump, "reset_parameters"):
            self.jump.reset_parameters()

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
