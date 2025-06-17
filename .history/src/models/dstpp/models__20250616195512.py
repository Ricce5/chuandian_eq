import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import EncoderLayer,TimePositionalEncoding, RNN_layers
from .mask_utils import get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask,get_self_attn_mask

class BaseEncoder(nn.Module):
    def __init__(self, 
                 d_model, d_inner, n_layers, n_head, d_k, d_v, 
                 dropout, attn_type, 
                 stack_names=["default"]):
        """
        stack_names: List[str], e.g., ["temporal", "loc", "fusion"]
                     If only one stack is needed, use ["default"]
        """
        super().__init__()
        self.stack_names = stack_names
        self.layer_stacks = nn.ModuleDict({
            name: self._build_layer_stack(d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type)
            for name in stack_names
        })

    def _build_layer_stack(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type):
        return nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v,
                         dropout=dropout, attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)
        ])

    def build_attention_mask(self, event_time):
        slf_attn_mask_subseq = get_subsequent_mask(event_time)
        slf_attn_mask_keypad = get_attn_key_pad_mask(event_time, event_time).type_as(slf_attn_mask_subseq)
        return (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

    def forward_layer_stack(self, stack_name, x, non_pad_mask, slf_attn_mask):
        for layer in self.layer_stacks[stack_name]:
            x, _ = layer(x, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return x

    def forward_multi_stack(self, inputs_dict, non_pad_mask, slf_attn_mask):
        """
        inputs_dict: Dict[str, Tensor], where keys match self.stack_names
        Returns: Dict[str, Tensor] of processed outputs
        """
        outputs = {}
        for name, x in inputs_dict.items():
            outputs[name] = self.forward_layer_stack(name, x, non_pad_mask, slf_attn_mask)
        return outputs
