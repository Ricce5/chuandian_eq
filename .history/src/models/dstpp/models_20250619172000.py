import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple
from .layers import RNN_layers
from .mask_utils import  get_non_pad_mask
from src.utils.registrable import Registrable


class BaseTransformer(nn.Module, Registrable):
    def __init__(self, encoder: nn.Module, rnn_modules: Dict[str, nn.Module], post_rnn_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.rnns = nn.ModuleDict(rnn_modules)
        self.dropout = nn.Dropout(post_rnn_dropout)

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    
        # === 1. 获取 event_time ===
        event_time = kwargs.get("event_time", None)
        if event_time is None:
            if len(args) >= 2:
                event_time = args[1]
            else:
                raise ValueError("BaseTransformer.forward() 需要提供 'event_time' 作为第二个位置参数或关键词参数。")

        if not isinstance(event_time, torch.Tensor):
            raise TypeError(f"'event_time' 必须是 torch.Tensor，但收到的是 {type(event_time)}。")

        non_pad_mask = get_non_pad_mask(event_time)

        try:
            enc_outputs = self.encoder(*args, non_pad_mask=non_pad_mask, **kwargs)
        except Exception as e:
            raise RuntimeError(f"调用 encoder 时出错: {e}")

        # === 4. 标准化 encoder 输出为 tuple ===
        if isinstance(enc_outputs, torch.Tensor):
            enc_outputs = (enc_outputs,)
        elif not isinstance(enc_outputs, (tuple, list)):
            raise TypeError("encoder 输出必须是 Tensor 或 (Tensor, ...)，但收到类型: {}".format(type(enc_outputs)))

        # === 5. 每个 encoder 输出走对应的 RNN 模块 ===
        if len(enc_outputs) != len(self.rnns):
            raise ValueError(
                f"encoder 输出数量（{len(enc_outputs)}）与 RNN 分支数量（{len(self.rnns)}）不一致，"
                f"请确保 encoder 和 rnn_modules 对齐。"
            )

        processed = []
        for name, output in zip(self.rnns.keys(), enc_outputs):
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Encoder 输出中的 '{name}' 不是 Tensor，而是 {type(output)}")
            processed.append(self.rnns[name](output, non_pad_mask))

        # === 6. 拼接或单输出 ===
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
                 device=None, loc_dim=2, attn_type='full'):

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
            attn_type=attn_type
        )
        rnn_modules = {
            "fusion": RNN_layers(d_model, d_rnn),
            "temporal": RNN_layers(d_model, d_rnn),
            "mark": RNN_layers(d_model, d_rnn)
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules)


# ===== Transformer_ST =====
class Transformer_ST(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, dropout_post_rnn=0.3,
                 device=None, loc_dim=2, attn_type='full'):

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