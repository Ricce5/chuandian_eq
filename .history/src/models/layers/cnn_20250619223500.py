



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