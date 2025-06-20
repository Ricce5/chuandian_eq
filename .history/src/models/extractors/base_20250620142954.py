# models/extractors/base.py
import torch.nn as nn

class RepresentationExtractor(nn.Module):
    def forward(self, enc_out, non_pad_mask, extra_inputs=None):
        """
        Args:
            enc_out: Tensor of shape [B, L, D]
            non_pad_mask: Tensor of shape [B, L, 1]
            extra_inputs: dict or None (e.g., {'t_n_seq': [B, L]})
        Returns:
            representation: Tensor of shape [B, D']
        """
        raise NotImplementedError("Subclasses must implement this method.")
