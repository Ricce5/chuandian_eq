from typing import Dict, Tuple

import torch
import torch.nn as nn


class RNNEncoder(nn.Module):
    """Generic LSTM/GRU encoder for task models, compatible with BaseModel."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        rnn_type: str = "lstm",
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        rnn_key = str(rnn_type).strip().lower()
        rnn_cls_map = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
        }
        if rnn_key not in rnn_cls_map:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}. Use 'lstm' or 'gru'.")

        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.output_dim = self.hidden_dim * (2 if self.bidirectional else 1)

        rnn_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.rnn = rnn_cls_map[rnn_key](
            input_size=int(input_dim),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=rnn_dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.post_dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, features: torch.Tensor, non_pad_mask: torch.Tensor) -> torch.Tensor:
        lengths = non_pad_mask.squeeze(-1).sum(dim=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            features,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=features.size(1),
        )
        out = self.post_dropout(out)
        return out * non_pad_mask


class RNNEncoderWrapper(nn.Module):
    """
    Wrap RNNEncoder to match BaseModel encoder interface:
    forward(features=..., non_pad_mask=..., **kwargs) -> (out, non_pad_mask, cache)
    """

    def __init__(self, encoder: RNNEncoder):
        super().__init__()
        self.encoder = encoder
        self.output_dim = encoder.output_dim

    def forward(self, inputs: Dict[str, torch.Tensor], caches=None) -> Tuple[torch.Tensor, torch.Tensor, None]:
        features = inputs["features"]
        non_pad_mask = inputs["non_pad_mask"]
        out = self.encoder(features=features, non_pad_mask=non_pad_mask)
        return out, non_pad_mask, None
