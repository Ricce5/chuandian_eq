import torch
import torch.nn as nn

class TaskModel(nn.Module):
    """
    将 encoder + 表示提取器 + head 整合为一个完整任务模型。
    支持分类、回归、计数等场景。
    """

    def __init__(self, base_model, extractor, head, final_activation=None,device=None):
        """
        Args:
            base_model: BaseModel，负责生成 encoder 输出和 mask
            extractor: 表示提取器（从 encoder 输出中提取序列表示）
            head: MLP 或线性分类器等任务头
            final_activation: 可选激活函数，如 sigmoid / softplus
        """
        super().__init__()
        self.base_model = base_model
        self.extractor = extractor
        self.head = head
        self.final_activation = final_activation

    def forward(self, x):
        # 1. 编码器 + 输入适配器
        enc_out, non_pad_mask = self.base_model(x)

        # 2. 获取额外输入（如 t_n_seq 用于拼接 attention pooling）
        extra_inputs = {}
        if hasattr(self.base_model.input_adapter, "get_extra_inputs"):
            extra_inputs = self.base_model.input_adapter.get_extra_inputs(x)

        # 3. 表示提取（如取最后一步 or attention pooling）
        representation = self.extractor(enc_out, non_pad_mask, extra_inputs)

        # 4. head 输出
        out = self.head(representation)

        # 5. 可选激活函数（如回归任务常用 softplus，分类任务用 sigmoid）
        if self.final_activation:
            out = self.final_activation(out)

        return out.squeeze(1)
