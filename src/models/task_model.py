import torch
import torch.nn as nn

class TaskModel(nn.Module):
    """"
    A generic model for different tasks (e.g., classification, regression) built on top of a base model.
    """
    def __init__(self, base_model, extractor, head, final_activation=None):
        super().__init__()
        self.base_model = base_model
        self.extractor = extractor
        self.head = head
        self.final_activation = final_activation

    def forward(self, x,caches=None):
        enc_out, non_pad_mask,_ = self.base_model(x, caches)
        extra_inputs = {}
        if hasattr(self.base_model.input_adapter, "get_extra_inputs"):
            extra_inputs = self.base_model.input_adapter.get_extra_inputs(x)
        representation = self.extractor(enc_out, non_pad_mask, extra_inputs)
        out = self.head(representation)
        if self.final_activation:
            out = self.final_activation(out)

        return out.squeeze(1)
    
    def set_attn_type(self, new_type: str):
        self.base_model.set_attn_type(new_type)

    def set_attn_dropout(self, p: float):
        self.base_model.set_attn_dropout(p)

    def get_pooled_representation(self, x, caches=None):
        enc_out, non_pad_mask, _ = self.base_model(x, caches)
        extra_inputs = {}
        if hasattr(self.base_model.input_adapter, "get_extra_inputs"):
            extra_inputs = self.base_model.input_adapter.get_extra_inputs(x)
        representation = self.extractor(enc_out, non_pad_mask, extra_inputs)
        return representation
