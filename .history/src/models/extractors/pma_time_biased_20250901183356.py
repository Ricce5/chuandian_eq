import math
import torch
import torch.nn as nn
from .base import RepresentationExtractor

@RepresentationExtractor.register("pma_time_biased")
class TimeBiasedPMA(nn.Module):
    """
    PMA + 时间偏置 + 可选 FiLM
      - Q = seeds; K,V = x 或 FiLM(x, t01)
      - logits += time-bias( linear/log ); head-wise g，log模式下 head-wise α
    """
    def __init__(self, d_model: int, n_heads: int = 4, r: int = 2,
                 agg: str = "concat",
                 use_film: bool = True, t_dim: int = 16,
                 bias_type: str = "log",         # 'linear' or 'log'
                 alpha0: float = 10.0,           # for 'log'
                 residual_last: bool = True,
                 device=None):
        super().__init__()
        assert agg in ("concat", "mean")
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.r, self.agg = d_model, n_heads, r, agg
        self.use_film = use_film
        self.bias_type = bias_type
        self.residual_last = residual_last

        self.seeds = nn.Parameter(torch.randn(r, d_model) / math.sqrt(d_model))
        d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out    = nn.Linear(n_heads * d_head, d_model)
        self.ln     = nn.LayerNorm(d_model)

        # head-wise 时间强度门 g_h；log 模式下 head-wise α_h
        self.g = nn.Parameter(torch.zeros(n_heads))
        if bias_type == "log":
            self.log_alpha = nn.Parameter(torch.log(torch.full((n_heads,), alpha0)))
        else:
            self.register_parameter("log_alpha", None)

        # 可选 FiLM: 用 t01 -> (γ,β) 调制 K/V 输入
        if use_film:
            self.t_mlp = nn.Sequential(
                nn.Linear(1, t_dim),
                nn.SiLU(),
                nn.Linear(t_dim, 2 * d_model)
            )

        self.device = device
        if device is not None:
            self.to(device)

    @staticmethod
    def _masked_softmax(logits, mask, dim=-1, eps=1e-9):
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(1)
        mask = mask.unsqueeze(1)                      # -> (B,1,1,L)
        logits = logits.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(logits, dim=dim)
        attn = attn * mask.float()
        attn = attn / (attn.sum(dim=dim, keepdim=True) + eps)
        return attn

    def _time_bias(self, t01, mask):
        """
        t01: (B,L,1)∈[0,1]; mask: (B,L)
        返回:
          linear -> phi: (B,L)∈[-1,0]
          log    -> phi_h: (B,H,L)
        """
        t_star = torch.where(mask, t01.squeeze(-1), torch.tensor(0., device=t01.device)) \
                    .max(dim=1, keepdim=True).values
        dt = (t_star - t01.squeeze(-1)).clamp(0.0, 1.0)          # (B,L)
        if self.bias_type == "linear":
            return -dt
        else:
            alpha = torch.exp(self.log_alpha).view(1, -1, 1)     # (1,H,1)
            dt_b  = dt.unsqueeze(1)                               # (B,1,L)
            return -torch.log1p(alpha * dt_b)                    # (B,H,L)

    def forward(self, x, mask, extra_inputs=None, return_score=False):
        """
        x:            (B,L,D)
        mask:         (B,L) True=valid
        extra_inputs: 需要 'event_time' in [0,1]  (B,L) 或 (B,L,1)
        """
        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask = mask.bool()

        if extra_inputs is None or 'event_time' not in extra_inputs:
            raise ValueError("TimeBiasedPMA 需要 extra_inputs['event_time']")
        t01 = extra_inputs['event_time']
        if t01.dim() == 2:
            t01 = t01.unsqueeze(-1)

        B, L, D = x.shape
        H, dh = self.n_heads, D // self.n_heads

        # K/V 输入的 FiLM 调制
        x_kv = x
        if self.use_film:
            gamma, beta = self.t_mlp(t01).chunk(2, dim=-1)       # (B,L,D)
            x_kv = gamma * x + beta

        # Q = seeds, K/V = x_kv
        S = self.seeds.unsqueeze(0).expand(B, self.r, D)         # (B,r,D)
        Q = self.q_proj(S).view(B, self.r, H, dh).transpose(1, 2)      # (B,H,r,dh)
        K = self.k_proj(x_kv).view(B, L, H, dh).transpose(1, 2)         # (B,H,L,dh)
        V = self.v_proj(x_kv).view(B, L, H, dh).transpose(1, 2)         # (B,H,L,dh)

        logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(dh)   # (B,H,r,L)

       # --- 计算时间偏置 ---
        if self.bias_type == "linear":
            phi = self._time_bias(t01, mask)                    # (B, L)
            # (B, L) -> (B, 1, 1, L)  然后乘以 g: (1, H, 1, 1) -> (B, H, 1, L)，在 r 维自动广播到 r
            bias = phi[:, None, None, :] * self.g[None, :, None, None]       # (B, H, 1, L)
        else:
            phi_h = self._time_bias(t01, mask)                  # (B, H, L)
            # 关键：在 dim=2 (r) 处 unsqueeze，得到 (B, H, 1, L)，再与 g 相乘
            bias = phi_h[:, :, None, :] * self.g[None, :, None, None]        # (B, H, 1, L)
        logits = logits + bias                                  # (B, H, r, L) + (B, H, 1, L) -> OK


        logits = logits + bias

        alpha = self._masked_softmax(logits, mask, dim=-1)               # (B,H,r,L)
        ctx   = torch.matmul(alpha, V)                                    # (B,H,r,dh)

        ctx = ctx.transpose(1, 2).contiguous().view(B, self.r, D)        # (B,r,D)
        out = self.out(ctx)                                              # (B,r,D)

        if self.agg == "concat":
            pooled = out.reshape(B, self.r * D)
        else:
            pooled = out.mean(dim=1)                                     # (B,D)
            if self.residual_last:
                last_idx = mask.sum(dim=1).long().clamp_min(1) - 1
                last = x[torch.arange(B, device=x.device), last_idx]
                pooled = self.ln(pooled + last)

        if return_score:
            return pooled, alpha
        return pooled
