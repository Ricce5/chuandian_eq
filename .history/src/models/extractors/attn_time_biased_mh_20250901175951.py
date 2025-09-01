import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import RepresentationExtractor

@RepresentationExtractor.register("attn_time_biased_mh")
class TimeAwareAttnPoolMH(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 t_dim: int = 16,
                 bias_type: str = "linear",     # "linear" 或 "log"
                 alpha0: float = 10.0,          # 仅对 "log" 初值有效
                 n_heads: int = 4,
                 agg: str = "concat",           # 'concat' 或 'mean'
                 device=None):
        super().__init__()
        assert agg in ("concat", "mean")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.t_dim = t_dim
        self.bias_type = bias_type
        self.n_heads = n_heads
        self.agg = agg
        self.device = device

        # FiLM: t -> (gamma, beta)   (与单头相同, 共享给所有头)
        self.t_mlp = nn.Sequential(
            nn.Linear(1, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, 2 * d_model)
        )
        self.W = nn.Linear(d_model, n_heads * d_hidden)

        # 每个头的 v_h （用于 e^content = v_h^T tanh(W_h x + b_h)）
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden))
        self.b = nn.Parameter(torch.zeros(n_heads, d_hidden))  # head-wise bias after W

        # 时间偏置强度门: 每个头一个 g_h
        self.g = nn.Parameter(torch.zeros(n_heads))            # shape: (H,)

        # log 类型时: 每个头一个 alpha_h（用 log_alpha 存）
        if bias_type == "log":
            self.log_alpha = nn.Parameter(torch.log(torch.full((n_heads,), alpha0)))
        else:
            self.register_parameter("log_alpha", None)

        if self.device is not None:
            self.to(self.device)

    @staticmethod
    def _masked_softmax(scores, mask, dim=-1, eps=1e-9):
        # scores: (..., L)
        # mask:   (B, L)  True=valid
        # 将 mask broadcast 到 scores 的 batch 维度
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)  # 在 head 或其他前导维度扩展
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=dim)
        attn = attn * mask.float()
        attn = attn / (attn.sum(dim=dim, keepdim=True) + eps)
        return attn

    def _time_bias(self, t, mask):
        """
        t:    [B, L, 1]  (已缩放到 [0,1] 的事件时间)
        mask: [B, L]     True=valid
        返回:
          linear: phi ∈ [B, L]，所有头共享同一 phi（每头只是在外面乘以自己的 g_h）
          log:    phi_h ∈ [B, H, L]，每头有独立 alpha_h
        """
        # t_star: 每个样本中“最晚的有效时间”
        t_star = torch.where(mask, t.squeeze(-1), torch.tensor(0., device=t.device)) \
                     .max(dim=1, keepdim=True).values   # [B, 1]
        dt = (t_star - t.squeeze(-1)).clamp(0.0, 1.0)   # [B, L] in [0,1]

        if self.bias_type == "linear":
            # 线性偏置: phi = -dt  ∈ [-1, 0]
            phi = -dt                                    # [B, L]
            return phi
        else:
            # 对数偏置: phi_h = -log(1 + alpha_h * dt)
            # alpha_h: (H,) -> broadcast to (B, H, L)
            alpha = torch.exp(self.log_alpha)            # (H,)
            # reshape 便于广播: (1,H,1) * (B,1,L) -> (B,H,L)
            alpha = alpha.view(1, -1, 1)
            dt_b = dt.unsqueeze(1)                       # (B,1,L)
            phi_h = - torch.log1p(alpha * dt_b)          # (B,H,L), 取值 (~[-log(1+alpha), 0])
            return phi_h

    def forward(self, x, mask, extra_inputs=None, return_score=False):
        """
        x:            [B, L, D]
        mask:         [B, L] 或 [B, L, 1]，True=valid
        extra_inputs: 必须包含 'event_time'，且已缩放到 [0, 1]
                      event_time: [B, L] 或 [B, L, 1]
        return_score: 返回注意力权重（按头）
        输出:
          agg='concat' -> [B, H*D]
          agg='mean'   -> [B, D]
        """
        if extra_inputs is None or 'event_time' not in extra_inputs:
            raise ValueError("Missing 'event_time' in extra_inputs for TimeAwareAttnPoolMH")
        t = extra_inputs['event_time']
        if t.dim() == 2:  # [B,L] -> [B,L,1]
            t = t.unsqueeze(-1)

        if mask.dim() == 3:
            mask = mask.squeeze(-1)
        mask = mask.bool()  # [B, L]

        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_hidden

        gamma, beta = self.t_mlp(t).chunk(2, dim=-1)     # [B, L, D]
        x_t = gamma * x + beta                           # [B, L, D]

        # 先做一次线性: [B, L, D] -> [B, L, H*Dh] -> [B, L, H, Dh]
        proj = self.W(x_t).view(B, L, H, Dh)             # 内容投影
        # 加 head-wise 偏置 b_h: (H,Dh) -> (1,1,H,Dh)
        proj = torch.tanh(proj + self.b.view(1, 1, H, Dh))
        # 与 v_h 做点积: (B,L,H,Dh) · (H,Dh) -> (B,L,H)
        content = torch.einsum('blhd,hd->blh', proj, self.v)

        # === 3) 时间偏置 ===
        if self.bias_type == "linear":
            # phi: [B, L]，扩展到 [B, L, H] 再乘以各头 g_h
            phi = self._time_bias(t, mask)               # [B, L]
            bias = (phi.unsqueeze(-1) * self.g.view(1, 1, H))  # [B, L, H]
        else:
            # phi_h: [B, H, L] -> 转置到 [B, L, H]
            phi_h = self._time_bias(t, mask).transpose(1, 2)   # (B,L,H)
            bias = phi_h * self.g.view(1, 1, H)                # (B,L,H)

        e = content + bias                                # [B, L, H]
        # === 4) masked softmax（按 L 归一化, 对每个头独立）===
        # 先把 e 变为 [B, H, L]，方便按 L softmax
        e_h = e.transpose(1, 2)                           # [B, H, L]
        alpha = self._masked_softmax(e_h, mask, dim=-1)   # [B, H, L]

        # === 5) 加权求和得到每头的 pooled 表示 ===
        # 先把 x_t 变为 [B, L, 1, D] 以便广播
        x_exp = x_t.unsqueeze(2)                          # [B, L, 1, D]
        # alpha: [B, H, L] -> [B, L, H, 1]
        alpha_exp = alpha.transpose(1, 2).unsqueeze(-1)   # [B, L, H, 1]
        # 加权求和: sum_L alpha * x
        z_heads = torch.sum(alpha_exp * x_exp, dim=1)     # [B, H, D]

        # === 6) 聚合各头 ===
        if self.agg == "concat":
            pooled = z_heads.reshape(B, H * D)            # [B, H*D]
        else:  # 'mean'
            pooled = z_heads.mean(dim=1)                  # [B, D]

        if return_score:
            # 返回按头的注意力: [B, H, L]
            return pooled, alpha
        return pooled
