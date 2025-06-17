import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import EncoderLayer,TimePositionalEncoding, RNN_layers
from .mask_utils import get_non_pad_mask, get_attn_key_pad_mask, get_subsequent_mask

class MultiBranchEncoder(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type, input_embeddings):
        super().__init__()
        self.embeddings = nn.ModuleDict(input_embeddings)

        self.branch_stacks = nn.ModuleDict({
            name: nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, attn_type, normalize_before=False)
                for _ in range(n_layers)
            ])
            for name in input_embeddings
        })

        self.final_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout, attn_type, normalize_before=False)
            for _ in range(n_layers)
        ])

    def forward(self, inputs: dict, non_pad_mask, slf_attn_mask):
        enc_outputs = {}
        for name, emb_layer in self.embeddings.items():
            enc_outputs[name] = emb_layer(inputs[name])

        merged_output = sum(enc_outputs.values())

        for i in range(len(self.final_stack)):
            for name in enc_outputs:
                enc_outputs[name], _ = self.branch_stacks[name][i](
                    enc_outputs[name], non_pad_mask, slf_attn_mask
                )
            merged_output, _ = self.final_stack[i](merged_output, non_pad_mask, slf_attn_mask)

        return merged_output, enc_outputs
