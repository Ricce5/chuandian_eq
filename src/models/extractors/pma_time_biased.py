import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import RepresentationExtractor


@RepresentationExtractor.register("pma_time_biased")
class TimeBiasedPMA(nn.Module):
    """
    PMA + Time Bias + Optional FiLM
      - Q = seeds; K, V = x or FiLM(x, t01)
      - logits += time-bias (linear/log); head-wise g, and head-wise α in log mode
    """

    def __init__(self, d_model: int, n_heads: int = 4, r: int = 4,
                 agg: str = "concat",
                 use_film: bool = True, t_dim: int = 16,
                 bias_type: str = "log",         # 'linear' or 'log'
                 alpha0: float = 10.0,           # for 'log'
                 residual_last: bool = True,
                 device=None):
        super().__init__()
        assert agg in ("concat", "mean")
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.r = r
        self.agg = agg
        self.use_film = use_film
        self.bias_type = bias_type
        self.residual_last = residual_last
        self.device = device

        d_head = d_model // n_heads

        # seeds: queries
        self.seeds = nn.Parameter(torch.randn(r, d_model) / math.sqrt(d_model))

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)

        self.out = nn.Linear(n_heads * d_head, d_model)
        self.ln = nn.LayerNorm(d_model)

    
        self.g = nn.Parameter(torch.zeros(n_heads))
        if bias_type == "log":
            self.log_alpha = nn.Parameter(torch.log(torch.full((n_heads,), alpha0)))
        else:
            self.register_parameter("log_alpha", None)

        if use_film:
            self.t_mlp = nn.Sequential(
                nn.Linear(1, t_dim),
                nn.SiLU(),
                nn.Linear(t_dim, 2 * d_model)
            )

        if device is not None:
            self.to(device)

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9):
        """
        logits: (B,H,r,L)
        mask:   (B,L)  True=valid
        """
        #  (B,1,1,L)
        mask_exp = mask.unsqueeze(1).unsqueeze(1)
        logits = logits.masked_fill(~mask_exp, float('-inf'))

        attn = torch.softmax(logits, dim=dim)
        attn = attn * mask_exp.float()
        attn = attn / (attn.sum(dim=dim, keepdim=True) + eps)
        return attn

    def _time_bias(self, t01: torch.Tensor, mask: torch.Tensor):
        """
        t01: (B,L,1)∈[0,1]; mask: (B,L)
        returns:
          linear -> phi:   (B,L)∈[-1,0]
          log    -> phi_h: (B,H,L)
        """
        # t_star = max event time in each sequence
        t01_s = t01.squeeze(-1)  # (B,L)
        t_star = torch.where(mask, t01_s, torch.tensor(0., device=t01.device)).max(dim=1, keepdim=True).values
        dt = (t_star - t01_s).clamp(0.0, 1.0)  # (B,L)

        if self.bias_type == "linear":
            return -dt
        elif self.bias_type == "log":
            # Logarithmic decay: -log(1 + α * dt)
            alpha = torch.exp(self.log_alpha).view(1, -1, 1)   # (1,H,1)
            dt_b = dt.unsqueeze(1)                             # (B,1,L)
            return -torch.log1p(alpha * dt_b)                  # (B,H,L)
        else:
            raise ValueError(f"Unknown bias_type: {self.bias_type}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, extra_inputs=None, return_score: bool = False):
        """
        x:            (B, L, D) - Input tensor with batch size B, sequence length L, and feature dimension D.
        mask:         (B, L) or (B, L, 1) - Boolean mask where True indicates valid positions.
        extra_inputs: Dictionary that must contain 'event_time' in the range [0, 1] with shape (B, L) or (B, L, 1).
        """

        if x.size(-1) != self.d_model:
            raise ValueError(f"x last dim {x.size(-1)} != d_model {self.d_model}")
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask = mask.bool()

        if extra_inputs is None or 'event_time' not in extra_inputs:
            raise ValueError("TimeBiasedPMA requires extra_inputs['event_time']")
        t01 = extra_inputs['event_time']
        if t01.dim() == 2:
            t01 = t01.unsqueeze(-1)

        B, L, D = x.shape
        H = self.n_heads
        dh = self.d_model // self.n_heads

    
        x_kv = x
        if self.use_film:
            gamma, beta = self.t_mlp(t01).chunk(2, dim=-1)  # (B,L,D)
            x_kv = gamma * x + beta

        # Q = seeds, K/V = x_kv
        S = self.seeds.unsqueeze(0).expand(B, self.r, D)                   # (B,r,D)
        Q = self.q_proj(S).view(B, self.r, H, dh).transpose(1, 2)          # (B,H,r,dh)
        K = self.k_proj(x_kv).view(B, L, H, dh).transpose(1, 2)            # (B,H,L,dh)
        V = self.v_proj(x_kv).view(B, L, H, dh).transpose(1, 2)            # (B,H,L,dh)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dh)      # (B,H,r,L)

        g_pos = F.softplus(self.g) 
        
        if self.bias_type == "linear":
            phi = self._time_bias(t01, mask)                               # (B,L)
            bias = phi[:, None, None, :] * g_pos[None, :, None, None]     # (B,H,1,L)
        else:
            phi_h = self._time_bias(t01, mask)                             # (B,H,L)
            bias = phi_h[:, :, None, :] * g_pos[None, :, None, None]      # (B,H,1,L)

        logits = logits + bias                                             # (B,H,r,L)


        alpha = self._masked_softmax(logits, mask, dim=-1)                 # (B,H,r,L)
        ctx = torch.matmul(alpha, V)                                       # (B,H,r,dh)
        # Flatten multi-heads into H*dh, then apply a linear projection to D 
        # (avoiding the hard assumption that H*dh == D for reshaping)
        Bc, Hc, r, dhc = ctx.shape
        assert Hc == H and dhc == dh
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(Bc, r, Hc * dhc)   # (B,r,H*dh)
        out = self.out(ctx)                                                # (B,r,D)

        # Aggregate r outputs
        if self.agg == "concat":
            pooled = out.reshape(B, self.r * D)                            # (B, r*D)
        elif self.agg == "mean":
            pooled = out.mean(dim=1)                                       # (B,D)
            if self.residual_last:
                last_idx = mask.sum(dim=1).long().clamp_min(1) - 1
                last = x[torch.arange(B, device=x.device), last_idx]       # (B,D)
                pooled = self.ln(pooled + last)
        else:
            raise ValueError(f"Unknown agg: {self.agg}")

        if return_score:
            return pooled, alpha
        return pooled
