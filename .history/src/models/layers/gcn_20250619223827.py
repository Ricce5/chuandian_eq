import torch
import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_mean_pool

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