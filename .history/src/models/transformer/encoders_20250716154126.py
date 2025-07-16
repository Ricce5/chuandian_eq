import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple
from .layers import EncoderLayer,TimePositionalEncoding, RNN_layers
from src.utils.mask_utils import get_subsequent_mask, get_attn_key_pad_mask
from src.utils.registrable import Registrable


class BaseEncoder(nn.Module, Registrable):
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
        self.n_layers = n_layers
        self.stack_names = stack_names
        self._attn_type = attn_type
        self.layer_stacks = nn.ModuleDict({
            name: self._build_layer_stack(
                d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type
            )
            for name in stack_names
        })
        self.set_attn_type(attn_type)

    def make_mlp(self, input_dim, output_dim, hidden_dim=None, num_layers=4):
        hidden_dim = hidden_dim or output_dim
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        mlp =nn.Sequential(*layers)

        # for m in mlp.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.zeros_(m.bias)
        return mlp


    def _build_layer_stack(self, d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type):
        return nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout,
                         normalize_before=False)
            for _ in range(n_layers)
        ])
    
    def set_attn_type(self, attn_type: str):
        self._attn_type = attn_type
        for stack in self.layer_stacks.values():
            for layer in stack:
                layer.set_attn_type(attn_type)
    
    def set_attn_dropout(self, p: float):
        for stack in self.layer_stacks.values():
            for layer in stack:
                layer.set_attn_dropout(p)


    def build_attention_mask(self, event_time: torch.Tensor) -> torch.Tensor:
        slf_attn_mask_subseq = get_subsequent_mask(event_time)
        slf_attn_mask_keypad = get_attn_key_pad_mask(event_time, event_time).type_as(slf_attn_mask_subseq)
        return (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

    def forward_layer_stack(
        self,
        stack_name: str,
        x: torch.Tensor,
        non_pad_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pre_layer_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        caches: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> torch.Tensor:
        """
        Forward through one layer stack, with optional hook before each layer.
        """
        for i,layer in enumerate(self.layer_stacks[stack_name]):
            if pre_layer_hook is not None:
                x = pre_layer_hook(x)
            layer_cache = caches[i] if caches is not None else None
            x, _ = layer(x, non_pad_mask=non_pad_mask, attn_mask=attn_mask, cache=layer_cache)
        return x

    def forward_multi_stack(
        self,
        inputs_dict: Dict[str, torch.Tensor],
        non_pad_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        hooks: Optional[Dict[str, Callable[[torch.Tensor], torch.Tensor]]] = None,
        caches_dict: Optional[Dict[str, List[Dict[str, torch.Tensor]]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward through multiple stacks. `inputs_dict` keys must match self.stack_names.
        Optionally apply a `pre_layer_hook` for each stack.
        """
        outputs = {}
        for name in self.stack_names:
            hook = hooks[name] if hooks and name in hooks else None
            stack_caches = caches_dict[name] if caches_dict and name in caches_dict else None
            outputs[name] = self.forward_layer_stack(
                name, inputs_dict[name], non_pad_mask,attn_mask, pre_layer_hook=hook,
                caches=stack_caches
            )
        return outputs
    @property
    def attn_type(self):
        return self._attn_type


@BaseEncoder.register("Encoder")
class Encoder(BaseEncoder):
    """A reusable encoder for event locations with time-aware attention."""

    def __init__(self,
                 d_model, d_inner,
                 n_layers, n_head, d_k, d_v, dropout,
                 device, attn_type, dim, emb_n_layer):
        super().__init__(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            attn_type=attn_type,
            stack_names=["default"]
        )

        self.d_model = d_model
        self.temporal_enc = TimePositionalEncoding(d_model, device=device)
        self.event_emb = self.make_mlp(input_dim=dim, output_dim=d_model,num_layers=emb_n_layer)
        

    def forward(
    self,
    event_mark,
    event_time,
    non_pad_mask,
    attn_mask: Optional[torch.Tensor] = None,
    caches: Optional[List[Dict[str, torch.Tensor]]] = None
):
        # === Embedding + Temporal Encoding ===
        tem_enc = self.temporal_enc(event_time) * non_pad_mask
        enc_output = self.event_emb(event_mark) * non_pad_mask
        out = self.event_emb(event_mark)
        if torch.isnan(out).any():
            print("❌ event_emb output contains NaN!")
            print("stats:", out.mean(), out.std(), out.min(), out.max())
            raise ValueError("event_emb 输出包含 NaN")
        enc_output += tem_enc  # 注入时间偏置
        # === 注入时间偏置进行注意力处理 ===
        return self.forward_layer_stack(
            stack_name="default",
            x=enc_output,
            non_pad_mask=non_pad_mask,
            attn_mask=attn_mask,
            caches=caches
        )


@BaseEncoder.register("Encoder_type")
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
        
    def forward(
    self,
    event_mark,
    event_time,
    non_pad_mask,
    attn_mask: Optional[torch.Tensor] = None,
    caches: Optional[List[Dict[str, torch.Tensor]]] = None
):
        # === 构造 Mask ===
         # === Embedding + Temporal Encoding ===
        tem_enc = self.temporal_enc(event_time) * non_pad_mask
        enc_output = self.event_emb(event_mark) * non_pad_mask
        out = self.event_emb(event_mark)

        # === 使用 hook 注入每层前加时间偏置 ===
        return self.forward_layer_stack(
            stack_name="default",
            x=enc_output,
            non_pad_mask=non_pad_mask,
            slf_attn_mask=slf_attn_mask,
            pre_layer_hook=lambda x: x + tem_enc
        )

@BaseEncoder.register("Encoder_ST")
class Encoder_ST(BaseEncoder):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v,
                 dropout, device, dim, attn_type):
        super().__init__(d_model, d_inner, n_layers, n_head, d_k, d_v, dropout, attn_type,
                         stack_names=["loc", "temporal", "fusion"])
        
        self.dim = dim
        self.d_model = d_model
        self.temporal_enc = TimePositionalEncoding(d_model, device=device)

        self.event_emb_loc = self.make_mlp(input_dim=dim, output_dim=d_model)

    

    def forward( self,
    event_mark,
    event_time,
    non_pad_mask,
    attn_mask: Optional[torch.Tensor] = None,
    caches: Optional[Dict[str, List[Dict[str, torch.Tensor]]]] = None
    ):

        # input embeddings
        enc_output_temporal = self.temporal_enc(event_time) * non_pad_mask
        enc_output_loc = self.event_emb_loc(event_mark) * non_pad_mask
        enc_output_fusion = enc_output_temporal + enc_output_loc

        # forward through each stack
        outputs = self.forward_multi_stack(
            inputs_dict={
                "temporal": enc_output_temporal,
                "loc": enc_output_loc,
                "fusion": enc_output_fusion
            },
            non_pad_mask=non_pad_mask,
            attn_mask=attn_mask,
            caches_dict=caches
        )

        return outputs["fusion"], outputs["temporal"], outputs["loc"]


@BaseEncoder.register("Encoder_STM")
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

    def forward(self, event_loc, event_time, event_mag, non_pad_mask):
        slf_attn_mask = self.build_attention_mask(event_time)
        enc_output_temporal = self.temporal_enc(event_time) * non_pad_mask
        enc_output_loc = self.event_emb_loc(event_loc) * non_pad_mask
        enc_output_mag = self.event_emb_magnitude(event_mag) * non_pad_mask

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


@BaseEncoder.register("Encoder_SE")
class Encoder_SE(BaseEncoder):
    """Spatial + Magnitude + Temporal encoder using self-attention."""

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

    def forward(self, event_loc, event_time, event_mag, non_pad_mask):
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



