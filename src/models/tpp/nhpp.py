
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
        self.to(self.device)


    def nll_loss(self, 
                 batch: src.data.Batch,
                 eps: float = 1e-10
                 ) -> torch.Tensor:
        
        nll_total = self.bg_model.nll(batch)
        return  nll_total / (batch.t_end - batch.t_nll_start)  # negated as NLL


    