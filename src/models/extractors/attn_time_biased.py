import  torch
import torch.nn as nn
import torch.nn.functional as F
from .base import RepresentationExtractor

@RepresentationExtractor.register("attn_time_biased")
class TimeAwareAttnPool(nn.Module):
    """
    attention pooling with time bias and FiLM conditioning on event times.
    """
    def __init__(self, d_model, d_hidden, t_dim=16, bias_type="linear", alpha0=10.0,device=None):
        super().__init__()
        self.t_mlp = nn.Sequential(
            nn.Linear(1, t_dim), nn.GELU(),
            nn.Linear(t_dim, 2*d_model)  # -> gamma, beta
        )
        self.W = nn.Linear(d_model, d_hidden)
        self.v = nn.Linear(d_hidden, 1)
        self.g = nn.Parameter(torch.tensor(0.0))   
        self.bias_type = bias_type  
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(alpha0)))  # for "log" bias
        self.device = device 
        self.to(device) if self.device is not None else None

    def _time_bias(self, t, mask):
        # t in [0,1], shape: [B,L,1]; mask: [B,L] True=valid
        t_star = torch.where(mask, t.squeeze(-1), torch.tensor(0., device=t.device)).max(dim=1, keepdim=True).values
        dt = (t_star - t.squeeze(-1)).clamp(0.0, 1.0)      # [B,L], in [0,1]
        if self.bias_type == "linear":
            phi = - dt                                     # [-1,0]
        else:  # "log"
            alpha = torch.exp(self.log_alpha)
            phi = - torch.log1p(alpha * dt)                # [~ -log(1+alpha), 0]
        return phi

    def forward(self, x, mask,  extra_inputs=None ,return_score=False):
        if extra_inputs is None or 'event_time' not in extra_inputs:
            raise ValueError("Missing 'event_time' in extra_inputs for  TimeAwareAttnPool")
        t = extra_inputs['event_time'] if extra_inputs['event_time'].dim() == 3 else extra_inputs['event_time'].unsqueeze(-1)
        if mask.dim() == 3: mask = mask.squeeze(-1)
        mask =mask.bool()  # [B,L]
       
        # FiLM：t in [0,1]
        gamma, beta = self.t_mlp(t).chunk(2, dim=-1)       # [B,L,D]
        x_t = gamma * x + beta

        # addictional attention
        h = torch.tanh(self.W(x_t))                        # [B,L,H]
        e = self.v(h).squeeze(-1)                          # [B,L]

        # time bias
        phi = self._time_bias(t, mask)                     # [B,L]
        e = e + self.g * phi

        # masked softmax
        e = e.masked_fill(~mask,  -torch.inf)
        alpha = torch.softmax(e, dim=1)
        alpha = alpha * mask.float()
        alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-9)
        pooled = torch.einsum('bl,bld->bd', alpha, x_t)
        if return_score:
            return pooled, alpha
        return pooled
