import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple
from .layers import RNN_layers
from src.utils.mask_utils import  get_non_pad_mask
from src.utils.registrable import Registrable


class BaseTransformer(nn.Module, Registrable):
    def __init__(self, encoder: nn.Module, rnn_modules: Dict[str, nn.Module], post_rnn_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.rnns = nn.ModuleDict(rnn_modules)
        self.dropout = nn.Dropout(post_rnn_dropout)

    def forward(self, features_dict: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(features_dict, dict):
            raise TypeError(f"BaseTransformer 期望 dict 输入，但收到 {type(features_dict)}")

        if "event_time" not in features_dict:
            raise ValueError("features_dict 中缺少 'event_time' 键")

        event_time = features_dict["event_time"]

        if event_time.dim() == 3 and event_time.size(-1) == 1:
            event_time = event_time.squeeze(-1)
        assert event_time.dim() == 2, f"'event_time' 维度必须为 [B, L]，当前为 {event_time.shape}"

        non_pad_mask = get_non_pad_mask(event_time)

        encoder_inputs = {k: v for k, v in features_dict.items() if k != "event_time"}
        try:
            enc_outputs = self.encoder(**encoder_inputs, event_time=event_time, non_pad_mask=non_pad_mask)
        except Exception as e:
            raise RuntimeError(f"调用 encoder 时出错: {e}")

        # 标准化 encoder 输出
        if isinstance(enc_outputs, torch.Tensor):
            enc_outputs = (enc_outputs,)
        elif not isinstance(enc_outputs, (tuple, list)):
            raise TypeError(f"encoder 输出必须为 Tensor 或 Tuple，但收到 {type(enc_outputs)}")

        # 送入对应 RNN
        if len(enc_outputs) != len(self.rnns):
            raise ValueError(f"encoder 输出数量（{len(enc_outputs)}）与 RNN 分支数量（{len(self.rnns)}）不一致")

        processed = []
        for name, output in zip(self.rnns.keys(), enc_outputs):
            processed.append(self.rnns[name](output, non_pad_mask))

        out = torch.cat(processed, dim=-1) if len(processed) > 1 else processed[0]
        return self.dropout(out), non_pad_mask



@BaseTransformer.register("Transformer")
class Transformer(BaseTransformer):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64,dropout_post_rnn=0.3,
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
            dim=dim,
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


@BaseTransformer.register("Transformer_SE")
class Transformer_SE(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1,dropout_post_rnn=0.3,
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
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=dropout_post_rnn)


@BaseTransformer.register("Transformer_ST")
class Transformer_ST(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, dropout_post_rnn=0.3,
                 device=None, dim=2, attn_type='full'):

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
            dim=dim,
            attn_type=attn_type
        )
        rnn_modules = {
            "fusion": RNN_layers(d_model, d_rnn),
            "temporal": RNN_layers(d_model, d_rnn),
            "loc": RNN_layers(d_model, d_rnn)
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=dropout_post_rnn)

@BaseTransformer.register("Transformer_STM")
class Transformer_STM(BaseTransformer):
    def __init__(self, d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1, dropout_post_rnn=0.3,
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
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=dropout_post_rnn)