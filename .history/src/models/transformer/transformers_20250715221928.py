import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple, Union, Any
from .layers import RNN_layers
from eq.utils.mask_utils import  get_non_pad_mask
from eq.utils.registrable import Registrable
from eq.data.dot_dict import DotDict

class BaseTransformer(nn.Module, Registrable):
    def __init__(self, encoder: nn.Module, rnn_modules: Dict[str, nn.Module], post_rnn_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.rnns = nn.ModuleDict(rnn_modules)
        self.dropout = nn.Dropout(post_rnn_dropout)

    def init_cache(self) -> Dict[str, Any]:
        return DotDict({
        "encoder": {
            f"layer_{i}": {"k": None, "v": None}
            for i in range(self.encoder.n_layers)
        },
        "rnn": {name: None for name in self.rnns}
    })


    def _init_or_extract_cache(self, caches: Optional[Dict[str, Any]]) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
        """
        Helper function to cleanly handle cache extraction or initialization.
        """
        if self.training or caches is None:
            encoder_cache = None
            rnn_cache_dict = {name: None for name in self.rnns}
        else:
            encoder_cache = caches.get("encoder", None)
            rnn_cache_dict = caches.get("rnn", {name: None for name in self.rnns})
            if set(rnn_cache_dict.keys()) != set(self.rnns.keys()):
                raise KeyError(
                    f"caches['rnn'] keys 与 RNN 模块不一致：{list(rnn_cache_dict.keys())} vs {list(self.rnns.keys())}"
                )
        return encoder_cache, rnn_cache_dict

    def forward(
        self,
        features_dict: dict,
        caches: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:

        # === 基本输入校验 ===
        if not isinstance(features_dict, dict):
            raise TypeError(f"BaseTransformer 期望 dict 输入，但收到 {type(features_dict)}")

        event_time = features_dict.get("event_time")
        if event_time is None:
            raise ValueError("features_dict 中缺少 'event_time' 键")

        if event_time.dim() == 3 and event_time.size(-1) == 1:
            event_time = event_time.squeeze(-1)
        assert event_time.dim() == 2, f"'event_time' 应为 [B, L]，当前为 {event_time.shape}"

        input_mask = features_dict.get("input_mask", None)
        if input_mask is None:
            non_pad_mask = get_non_pad_mask(event_time)
        else:
            non_pad_mask = input_mask.unsqueeze(-1)

        # === cache 提取 or 初始化 ===
        encoder_cache, rnn_cache_dict = self._init_or_extract_cache(caches)

        # === encoder 输入准备 & 前向 ===
        encoder_inputs = {k: v for k, v in features_dict.items() if k not in ["event_time", "input_mask"]}
        try:
            encoder_outputs = self.encoder(
                **encoder_inputs,
                event_time=event_time,
                non_pad_mask=non_pad_mask,
                caches=encoder_cache
            )
        except Exception as e:
            raise RuntimeError(f"调用 encoder 时出错: {e}")
        ## encoder_outputs = features_dict['event_mark'] 

        if isinstance(encoder_outputs, torch.Tensor):
            encoder_outputs = (encoder_outputs,)
        elif not isinstance(encoder_outputs, (tuple, list)):
            raise TypeError(f"encoder 输出必须为 Tensor 或 Tuple，但收到 {type(encoder_outputs)}")

        if len(encoder_outputs) != len(self.rnns):
            raise ValueError(f"encoder 输出数量（{len(encoder_outputs)}）与 RNN 模块数量（{len(self.rnns)}）不一致")
        # === RNN 分支处理 ===
        processed_outputs = []
        updated_rnn_cache = {}

        for (name, rnn_layer), enc_out in zip(self.rnns.items(), encoder_outputs):
            rnn_cache = rnn_cache_dict[name]
            rnn_out, new_cache = rnn_layer(enc_out, non_pad_mask, cache=rnn_cache)
            processed_outputs.append(self.dropout(rnn_out))
            updated_rnn_cache[name] = new_cache
        # === 合并输出 ===
        final_output = (
            torch.cat(processed_outputs, dim=-1)
            if len(processed_outputs) > 1 else
            processed_outputs[0]
        )

        new_cache = DotDict({
            "encoder": encoder_cache,    
            "rnn": updated_rnn_cache
        })
        return final_output, non_pad_mask, new_cache
    
    def set_attn_type(self, new_type: str):
        if hasattr(self.encoder, "set_attn_type"):
            self.encoder.set_attn_type(new_type)
    
    def set_attn_dropout(self, p: float):
        if hasattr(self.encoder, "set_attn_dropout"):
            self.encoder.set_attn_dropout(p)
        

    


@BaseTransformer.register("Transformer")
class Transformer(BaseTransformer):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4,emb_n_layer=2,
                 n_head=4, d_k=64, d_v=64,dropout_post_rnn=0,
                 dropout=0, device=None, dim=2, attn_type='flash'):

        from .encoders import Encoder
        
        encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            emb_n_layer=emb_n_layer,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            device=device,
            dim=dim,
            attn_type=attn_type
        )

        rnn_modules = {
            # "fusion": RNN_layers(d_model, d_rnn)
            "fusion": RNN_layers(d_model, d_rnn)
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=dropout_post_rnn)


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