import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional
from models.encoder_layer import EncoderLayer  # 确保你导入了这个类
from models.attn_mask import get_subsequent_mask, get_attn_key_pad_mask

class BaseEncoder(nn.Module):
    def __init__(self, 
                 d_model: int,
                 d_inner: int,
                 n_layers: int,
                 n_head: int,
                 d_k: int,
                 d_v: int,
                 dropout: float,
                 attn_type: str,
                 stack_names: List[str] = ["default"]):
        """
        Base Encoder that supports one or multiple stacks of EncoderLayer blocks.
        
        stack_names: list of stack identifiers, e.g., ["default"], or ["loc", "temporal", "fusion"]
        """
        super(BaseEncoder, self).__init__()
        self.stack_names = stack_names
        self.layer_stacks = nn.ModuleDict({
            name: self._build_layer_stack(
                d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type
            )
            for name in stack_names
        })

    def _build_layer_stack(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type):
        return nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,
                         attn_type=attn_type, normalize_before=False)
            for _ in range(n_layers)
        ])

    def build_attention_mask(self, event_time: torch.Tensor) -> torch.Tensor:
        slf_attn_mask_subseq = get_subsequent_mask(event_time)
        slf_attn_mask_keypad = get_attn_key_pad_mask(event_time, event_time).type_as(slf_attn_mask_subseq)
        return (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

    def forward_layer_stack(
        self,
        stack_name: str,
        x: torch.Tensor,
        non_pad_mask: torch.Tensor,
        slf_attn_mask: torch.Tensor,
        pre_layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Forward through one layer stack, with optional hook before each layer.
        """
        for layer in self.layer_stacks[stack_name]:
            if pre_layer_hook is not None:
                x = pre_layer_hook(x)
            x, _ = layer(x, non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)
        return x

    def forward_multi_stack(
        self,
        inputs_dict: Dict[str, torch.Tensor],
        non_pad_mask: torch.Tensor,
        slf_attn_mask: torch.Tensor,
        hooks: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward through multiple stacks. `inputs_dict` keys must match self.stack_names.
        Optionally apply a `pre_layer_hook` for each stack.
        """
        outputs = {}
        for name in self.stack_names:
            hook = hooks[name] if hooks and name in hooks else None
            outputs[name] = self.forward_layer_stack(
                name, inputs_dict[name], non_pad_mask, slf_attn_mask, pre_layer_hook=hook
            )
        return outputs
