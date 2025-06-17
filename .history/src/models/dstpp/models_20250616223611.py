import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple
from .layers import RNN_layers
from .mask_utils import  get_non_pad_mask


class BaseTransformer(nn.Module):
    def __init__(self, encoder: nn.Module, rnn_modules: Dict[str, nn.Module], post_rnn_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.rnns = nn.ModuleDict(rnn_modules)
        self.dropout = nn.Dropout(post_rnn_dropout)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        # 支持参数形式：forward(loc, time, [mag...]) 或 forward(**dict)

        # 尝试从 kwargs 中获取 event_time，用于构建 mask
        if "event_time" in kwargs:
            event_time = kwargs["event_time"]
        elif len(args) >= 2:
            event_time = args[1]  # 假设第二个位置参数是 event_time
        else:
            raise ValueError("Missing event_time for mask creation")

        non_pad_mask = get_non_pad_mask(event_time)

        # 编码器支持 *args, non_pad_mask=...
        enc_outputs = self.encoder(*args, non_pad_mask=non_pad_mask, **kwargs)

        if not isinstance(enc_outputs, (list, tuple)):
            enc_outputs = (enc_outputs,)

        processed = [
            self.rnns[name](output, non_pad_mask)
            for name, output in zip(self.rnns.keys(), enc_outputs)
        ]
        out = torch.cat(processed, dim=-1) if len(processed) > 1 else processed[0]
        return self.dropout(out), non_pad_mask




class Transformer(BaseTransformer):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64,
                 dropout=0.1, device=None, loc_dim=2, attn_type='full'):

        from .encoders import Encoder
        encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim=loc_dim,
            attn_type=attn_type
        )

        rnn_modules = {
            "fusion": RNN_layers(d_model, d_rnn)
        }

        super().__init__(encoder=encoder, rnn_modules=rnn_modules)


# ===== Transformer_type =====
class Transformer_type(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024, n_layers=4,
                 n_head=4, d_k=64, d_v=64, dropout=0.1, dropout_post_rnn=0.3,
                 device=None, attn_type='full', num_event_types_pad=None, pad_token_id=-100):

        from .encoders import Encoder_type
        encoder = Encoder_type(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            attn_type=attn_type,
            num_event_types_pad=num_event_types_pad,
            pad_token_id=pad_token_id
        )
        rnn_modules = {"fusion": RNN_layers(d_model, d_rnn)}
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=dropout_post_rnn)


# ===== Transformer_SE =====
class Transformer_SE(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
                 device=None, loc_dim=2, CosSin=False, attn_type='full'):

        from .encoders import Encoder_SE
        encoder = Encoder_SE(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim=loc_dim,
            CosSin=CosSin,
            attn_type=attn_type
        )
        rnn_modules = {
            "fusion": RNN_layers(d_model, d_rnn),
            "temporal": RNN_layers(d_model, d_rnn),
            "mark": RNN_layers(d_model, d_rnn)
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules)
        self.alpha = nn.Parameter(torch.tensor(-0.1))
        self.beta = nn.Parameter(torch.tensor(1.0))


# ===== Transformer_ST =====
class Transformer_ST(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, dropout_post_rnn=0.3,
                 device=None, loc_dim=2, CosSin=False, attn_type='full'):

        from .encoders import Encoder_ST
        encoder = Encoder_ST(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim=loc_dim,
            CosSin=CosSin,
            attn_type=attn_type
        )
        rnn_modules = {
            "fusion": RNN_layers(d_model, d_rnn),
            "temporal": RNN_layers(d_model, d_rnn),
            "loc": RNN_layers(d_model, d_rnn)
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=dropout_post_rnn)

class Transformer_STM(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,
                 device=None, loc_dim=2, attn_type='full'):

        from .encoders import Encoder_STM
        encoder = Encoder_STM(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            loc_dim=loc_dim,
            attn_type=attn_type
        )
        rnn_modules = {
            "fusion": RNN_layers(d_model, d_rnn),
            "temporal": RNN_layers(d_model, d_rnn),
            "loc": RNN_layers(d_model, d_rnn),
            "magnitude": RNN_layers(d_model, d_rnn)
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules)