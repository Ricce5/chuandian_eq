import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple
from .layers import EncoderLayer,TimePositionalEncoding, RNN_layers
from .mask_utils import get_subsequent_mask, get_attn_key_pad_mask, get_non_pad_mask

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

class Encoder_type(BaseEncoder):
    """A reusable encoder for event types with time-aware attention."""

    def __init__(self,
                 d_model, d_inner,
                 n_layers, n_head, d_k, d_v, dropout, device, attn_type,
                 num_event_types_pad, pad_token_id):
        super().__init__(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_type=attn_type,
            stack_names=["default"]  # only one layer stack
        )

        self.d_model = d_model
        self.temporal_enc = TimePositionalEncoding(d_model, device=device)
        self.event_emb = nn.Embedding(num_event_types_pad, d_model, padding_idx=pad_token_id)
        
    def forward(self, event_type, event_time, non_pad_mask):
        # === 构造 Mask ===
        slf_attn_mask = self.build_attention_mask(event_time)

        # === Embedding + Temporal Encoding ===
        tem_enc = self.temporal_enc(event_time) * non_pad_mask
        enc_output = self.event_emb(event_type) * non_pad_mask
        print(torch.sum(tem_enc))
        print(torch.sum(enc_output))

        # === 使用 hook 注入每层前加时间偏置 ===
        return self.forward_layer_stack(
            stack_name="default",
            x=enc_output,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask,
            pre_layer_hook=lambda x: x + tem_enc
        )

    
class Encoder_ST(BaseEncoder):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v,
                 dropout, device, loc_dim, attn_type):
        super().__init__(d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type,
                         stack_names=["loc", "temporal", "fusion"])
        
        self.loc_dim = loc_dim
        self.d_model = d_model
        self.temporal_enc = TimePositionalEncoding(d_model, device=device)

        self.event_emb_loc = self.make_mlp(input_dim=loc_dim, output_dim=d_model)

    def make_mlp(self, input_dim, output_dim, hidden_dim=None, num_layers=4):
        hidden_dim = hidden_dim or output_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, event_loc, event_time, non_pad_mask):
        slf_attn_mask = self.build_attention_mask(event_time)

        # input embeddings
        enc_output_temporal = self.temporal_enc(event_time) * non_pad_mask
        enc_output_loc = self.event_emb_loc(event_loc) * non_pad_mask
        enc_output_fusion = enc_output_temporal + enc_output_loc

        # forward through each stack
        outputs = self.forward_multi_stack(
            {
                "temporal": enc_output_temporal,
                "loc": enc_output_loc,
                "fusion": enc_output_fusion
            },
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask
        )

        return outputs["fusion"], outputs["temporal"], outputs["loc"]



class Encoder_STM(BaseEncoder):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v,
                 dropout, device, loc_dim, attn_type):
        super().__init__(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_type=attn_type,
            stack_names=["loc", "temporal", "magnitude", "fusion"]
        )

        self.loc_dim = loc_dim
        self.d_model = d_model

        self.temporal_enc = TimePositionalEncoding(d_model, device=device)

        self.event_emb_loc = nn.Sequential(
            nn.Linear(loc_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.event_emb_magnitude = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, event_loc, event_time, event_magnitude, non_pad_mask):
        slf_attn_mask = self.build_attention_mask(event_time)
        enc_output_temporal = self.temporal_enc(event_time) * non_pad_mask
        enc_output_loc = self.event_emb_loc(event_loc) * non_pad_mask
        enc_output_mag = self.event_emb_magnitude(event_magnitude) * non_pad_mask

        enc_output_fusion = enc_output_temporal + enc_output_loc + enc_output_mag

        outputs = self.forward_multi_stack(
            {
                "temporal": enc_output_temporal,
                "loc": enc_output_loc,
                "magnitude": enc_output_mag,
                "fusion": enc_output_fusion
            },
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask
        )

        return outputs["fusion"], outputs["temporal"], outputs["loc"], outputs["magnitude"]

class Encoder_SE(BaseEncoder):
    """Spatial + Magnitude + Temporal encoder using self-attention."""

    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v,
                 dropout, device, loc_dim, attn_type, CosSin=False):
        super().__init__(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_type=attn_type,
            stack_names=["temporal", "mark", "fusion"]
        )

        self.loc_dim = loc_dim
        self.d_model = d_model

        self.temporal_enc = TimePositionalEncoding(d_model, device=device)

        self.event_emb_loc = nn.Sequential(
            nn.Linear(loc_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.event_emb_mag = nn.Sequential(
            nn.Linear(1, d_model),
             nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, event_loc, event_mag, event_time, non_pad_mask):
        # === Attention mask
        slf_attn_mask = self.build_attention_mask(event_time)

        # === Input encodings (all masked)
        enc_output_temporal = self.temporal_enc(event_time) * non_pad_mask
        enc_output_loc = self.event_emb_loc(event_loc) * non_pad_mask
        enc_output_mag = self.event_emb_mag(event_mag) * non_pad_mask
        enc_output_mark = enc_output_mag + enc_output_loc
        enc_output_fusion = enc_output_temporal + enc_output_mark

        # === Multi-branch attention stacks
        outputs = self.forward_multi_stack(
            {
                "temporal": enc_output_temporal,
                "mark": enc_output_mark,
                "fusion": enc_output_fusion
            },
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask
        )

        return outputs["fusion"], outputs["temporal"], outputs["mark"]



class BaseTransformer(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 rnn_modules: Dict[str, nn.Module],
                 post_rnn_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.rnns = nn.ModuleDict(rnn_modules)
        self.dropout = nn.Dropout(post_rnn_dropout)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        inputs: must include 'event_time' and any args encoder needs
        encoder should return one or multiple outputs (dict or tuple)
        """
        non_pad_mask = get_non_pad_mask(inputs["event_time"])

        # forward encoder
        enc_outputs = self.encoder(**inputs, non_pad_mask=non_pad_mask)
        if not isinstance(enc_outputs, (list, tuple)):
            enc_outputs = (enc_outputs,)

        # run each rnn module over respective encoder output
        assert len(enc_outputs) == len(self.rnns), \
            f"Encoder outputs {len(enc_outputs)} != RNN branches {len(self.rnns)}"

        processed = []
        for (name, rnn), enc_out in zip(self.rnns.items(), enc_outputs):
            rnn_out = rnn(enc_out, non_pad_mask)
            processed.append(rnn_out)

        output = torch.cat(processed, dim=-1) if len(processed) > 1 else processed[0]
        output = self.dropout(output)
        return output, non_pad_mask



class Transformer_type(BaseTransformer):
    def __init__(self, **kwargs):
        encoder = Encoder_type(**kwargs)
        rnn_modules = {
            "main": RNN_layers(kwargs['d_model'], kwargs['d_rnn'])
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=kwargs.get("dropout_post_rnn", 0.3))

class Transformer_ST(BaseTransformer):
    def __init__(self, **kwargs):
        encoder = Encoder_ST(**kwargs)
        rnn_modules = {
            "fusion": RNN_layers(kwargs['d_model'], kwargs['d_rnn']),
            "temporal": RNN_layers(kwargs['d_model'], kwargs['d_rnn']),
            "spatial": RNN_layers(kwargs['d_model'], kwargs['d_rnn']),
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=kwargs.get("dropout_post_rnn", 0.3))

class Transformer_STM(BaseTransformer):
    def __init__(self, **kwargs):
        encoder = Encoder_STM(**kwargs)
        rnn_modules = {
            "fusion": RNN_layers(kwargs['d_model'], kwargs['d_rnn']),
            "temporal": RNN_layers(kwargs['d_model'], kwargs['d_rnn']),
            "spatial": RNN_layers(kwargs['d_model'], kwargs['d_rnn']),
            "magnitude": RNN_layers(kwargs['d_model'], kwargs['d_rnn']),
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules)
