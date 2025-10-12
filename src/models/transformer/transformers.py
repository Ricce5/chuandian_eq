# Enhance transformers with building blocks
# Reference: Spatio-temporal Diffusion Point Processes https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes
import torch
import torch.nn as nn
from typing import List, Dict, Callable, Optional, Tuple, Union, Any
from .layers import RNN_layers
from src.utils.mask_utils import  get_non_pad_mask
from src.utils.registrable import Registrable
from src.data.dot_dict import DotDict

class BaseTransformer(nn.Module, Registrable):
    def __init__(self, encoder: nn.Module, rnn_modules: Dict[str, nn.Module], post_rnn_dropout: float = 0.0):
        super().__init__()
        self.encoder = encoder
        self.rnns = nn.ModuleDict(rnn_modules)
        self.dropout = nn.Dropout(post_rnn_dropout)

    def init_cache(self) -> Dict[str, Any]:
        return DotDict({
        "encoder": {
            name: [{"k": None, "v": None} for _ in range(self.encoder.n_layers)]
            for name in self.encoder.stack_names
        },
        "rnn": {name: None for name in self.rnns}
    })


    def _init_or_extract_cache(self, caches: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, List[Dict[str, torch.Tensor]]]], Dict[str, Any]]:
        """
        Helper function to cleanly handle cache extraction or initialization,
        supporting both single and multi-stack encoder cache.
        """
        stack_names = getattr(self.encoder, "stack_names", ["default"])

        if self.training or caches is None:
            encoder_cache = {
                name: None for name in stack_names
            }
            rnn_cache_dict = {
                name: None for name in self.rnns
            }
        else:
            encoder_cache = caches.get("encoder", None)
            rnn_cache_dict = caches.get("rnn", {name: None for name in self.rnns})

            if set(rnn_cache_dict.keys()) != set(self.rnns.keys()):
                raise KeyError(
                    f"The keys in caches['rnn'] do not match the RNN modules: {list(rnn_cache_dict.keys())} vs {list(self.rnns.keys())}"
                )

            if encoder_cache is not None:
                if isinstance(encoder_cache, list):
                    if len(stack_names) != 1:
                        raise ValueError(f"encoder_cache is a List, but encoder.stack_names has multiple entries: {stack_names}")
                    encoder_cache = {stack_names[0]: encoder_cache}
                elif isinstance(encoder_cache, dict):
                    if set(encoder_cache.keys()) != set(stack_names):
                        raise KeyError(
                            f"The keys in encoder_cache do not match the stack names: {list(encoder_cache.keys())} vs {stack_names}"
                        )
                else:
                    raise TypeError(f"Invalid type for encoder_cache, received: {type(encoder_cache)}")

        return encoder_cache, rnn_cache_dict


    def forward(
        self,
        features_dict: dict,
        caches: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:

        if not isinstance(features_dict, dict):
            raise TypeError(f"BaseTransformer expects a dict as input, but received {type(features_dict)}")

        event_time = features_dict.get("event_time")
        if event_time is None:
            raise ValueError("The 'features_dict' is missing the 'event_time' key")

        if event_time.dim() == 3 and event_time.size(-1) == 1:
            event_time = event_time.squeeze(-1)
        assert event_time.dim() == 2, f"'event_time' should be of shape [B, L], but got {event_time.shape}"

        input_mask = features_dict.get("input_mask", None)
        if input_mask is None:
            non_pad_mask = get_non_pad_mask(event_time)
        else:
            non_pad_mask = input_mask.unsqueeze(-1)

        encoder_cache, rnn_cache_dict = self._init_or_extract_cache(caches)

        encoder_inputs = {k: v for k, v in features_dict.items() if k not in ["event_time", "input_mask"]}
        try:
            encoder_outputs = self.encoder(
                **encoder_inputs,
                event_time=event_time,
                non_pad_mask=non_pad_mask,
                caches=encoder_cache
            )
        except Exception as e:
            raise RuntimeError(f"Error occurred while calling the encoder: {e}")
        ## encoder_outputs = features_dict['event_mark'] 

        if isinstance(encoder_outputs, torch.Tensor):
            encoder_outputs = (encoder_outputs,)
        elif not isinstance(encoder_outputs, (tuple, list)):
            raise TypeError(f"encoder outputs must be a Tensor or a Tuple, but received {type(encoder_outputs)}")

        if len(encoder_outputs) != len(self.rnns):
            raise ValueError(f"The number of encoder outputs ({len(encoder_outputs)}) does not match the number of RNN modules ({len(self.rnns)})")
        processed_outputs = []
        updated_rnn_cache = {}

        for (name, rnn_layer), enc_out in zip(self.rnns.items(), encoder_outputs):
            rnn_cache = rnn_cache_dict[name]
            rnn_out, new_cache = rnn_layer(enc_out, non_pad_mask, cache=rnn_cache)
            processed_outputs.append(self.dropout(rnn_out))
            updated_rnn_cache[name] = new_cache
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

@BaseTransformer.register("Transformer_Logdeltat")
class Transformer_Logdeltat(BaseTransformer):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4,emb_n_layer=2,
                 n_head=4, d_k=64, d_v=64,dropout_post_rnn=0,
                 dropout=0, device=None, dim=2, attn_type='flash'):

        from .encoders import Encoder_Logdeltat
        
        encoder = Encoder_Logdeltat(
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
            "fusion": RNN_layers(d_model+1, d_rnn)
        }
        super().__init__(encoder=encoder, rnn_modules=rnn_modules, post_rnn_dropout=dropout_post_rnn)



@BaseTransformer.register("Transformer_Conv")
class Transformer_Conv(BaseTransformer):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self,
                 d_model=256, d_rnn=128, d_inner=1024,
                 n_layers=4,emb_n_layer=2,
                 n_head=4, d_k=64, d_v=64,dropout_post_rnn=0,
                 dropout=0, device=None, dim=2, attn_type='flash'):

        from .encoders import Encoder_Conv
        
        encoder = Encoder_Conv(
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