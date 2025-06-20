from .base import RepresentationExtractor
import torch

@RepresentationExtractor.register("last")
class LastStepExtractor(RepresentationExtractor):
    def forward(self, enc_out, non_pad_mask, extra_inputs=None):
        lengths = non_pad_mask.sum(dim=1).squeeze(-1).long()  # [B]
        idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(enc_out.size(0), device=enc_out.device)
        return enc_out[batch_idx, idx]  # [B, D]
