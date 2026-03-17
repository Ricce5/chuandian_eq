
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import src
import src.distributions as dist

from .tpp_model import TPPModel


class NHPP(TPPModel):
    def __init__(self, args,device=None,bg_model=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.bg_model = bg_model
        self.reduction = getattr(args, "loss_reduction", "per_time")
        self.to(self.device)


    def nll_loss(self, 
                 batch: src.data.Batch,
                 *,
                 reduction: str | None = None,
                 return_dict: bool = False,
                 eps: float = 1e-10
                 ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.bg_model is None:
            raise ValueError("NHPP requires a background model to compute NLL.")

        reduction = self.reduction if reduction is None else reduction
        nll_bg = self.bg_model.nll(batch)
        out = self.reduce_nll_dict(
            {
                "bg": nll_bg,
                "total": nll_bg,
            },
            batch,
            reduction=reduction,
            eps=eps,
        )
        if return_dict:
            return out
        return out["total"]


    
