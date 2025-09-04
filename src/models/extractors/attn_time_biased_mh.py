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
                 bias_type: str = "linear",     # "linear" or "log"
                 alpha0: float = 10.0,          # only for "log"
                 n_heads: int = 4,
                 agg: str = "concat",           # "concat" or "mean"
                 # ====== 新增：融合相关开关 ======
                 fuse_mode: str = "none",       # "none" | "add" | "concat" | "gate"
                 use_ln: bool = True,           # 融合后是否做 LayerNorm
                 use_var_scale: bool = True,    # "add" 模式是否用方差匹配 s
                 repr_dim: int = None,          # "concat" 时输出维度(默认 d_model)
                 device=None):
        super().__init__()
        assert agg in ("concat", "mean")
        assert fuse_mode in ("none", "add", "concat", "gate")
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.t_dim = t_dim
        self.bias_type = bias_type
        self.n_heads = n_heads
        self.agg = agg

        self.fuse_mode = fuse_mode
        self.use_ln = use_ln
        self.use_var_scale = use_var_scale
        self.repr_dim = repr_dim or d_model
        self.device = device

        # ---- 原有组件 ----
        self.t_mlp = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, 2 * d_model)
        )
        self.W = nn.Linear(d_model, n_heads * d_hidden)
        self.v = nn.Parameter(torch.randn(n_heads, d_hidden))
        self.b = nn.Parameter(torch.zeros(n_heads, d_hidden))
        self.g = nn.Parameter(torch.zeros(n_heads))
        if bias_type == "log":
            self.log_alpha = nn.Parameter(torch.log(torch.full((n_heads,), alpha0)))
        elif bias_type == "linear":
            self.register_parameter("log_alpha", None)
        else:
            raise ValueError(f"Unknown bias_type: {bias_type}")

        # ---- 新增：融合需要的层 ----
        pooled_dim = d_model if self.agg == "mean" else (n_heads * d_model)
        if self.fuse_mode in ("add", "concat", "gate") and self.use_ln:
            self.ln_last = nn.LayerNorm(d_model)
            # pooled 的维度取决于 agg
            self.ln_attn = nn.LayerNorm(pooled_dim)
            self.ln_out  = nn.LayerNorm(self.repr_dim)

        if self.fuse_mode == "concat":
            self.fuse_proj = nn.Linear(d_model + pooled_dim, self.repr_dim)
        elif self.fuse_mode == "gate":
            self.gate_proj = nn.Linear(d_model + pooled_dim, 1)

        if self.device is not None:
            self.to(self.device)

        if self.fuse_mode == "none":
            self.output_dim = pooled_dim
        else:
            self.output_dim = self.repr_dim



    @staticmethod
    def _masked_softmax(scores, mask, dim=-1, eps=1e-9):
        while mask.dim() < scores.dim():
            mask = mask.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=dim)
        attn = attn * mask.float()
        attn = attn / (attn.sum(dim=dim, keepdim=True) + eps)
        return attn

    def _time_bias(self, t, mask):
        # t_star = torch.where(mask, t.squeeze(-1), torch.tensor(0., device=t.device)) \
        #             .max(dim=1, keepdim=True).values        # [B,1]
        t_star = torch.ones((t.size(0), 1), device=t.device)  # [B,1]
        dt = (t_star - t.squeeze(-1)).clamp(0.0, 1.0)       # [B,L]
        if self.bias_type == "linear":
            return -dt                                      # [B,L]
        elif self.bias_type == "log":
            alpha = torch.exp(self.log_alpha).view(1, -1, 1)  # (1,H,1)
            dt_b = dt.unsqueeze(1)                            # (B,1,L)
            return -torch.log1p(alpha * dt_b).transpose(1, 2) # (B,L,H)
        else:
            raise ValueError(f"Unknown bias_type: {self.bias_type}")

    def forward(self, x, mask, extra_inputs=None, return_score=False, last_token=None):
        """
        x: [B,L,D]
        mask: [B,L] or [B,L,1]  True=valid
        extra_inputs['event_time']: [B,L] or [B,L,1]  (scaled to [0,1])
        last_token: [B,D]  可选；当 fuse_mode != "none" 时建议提供
        """
        if extra_inputs is None or 'event_time' not in extra_inputs:
            raise ValueError("Missing 'event_time' in extra_inputs for TimeAwareAttnPoolMH")
        t = extra_inputs['event_time']
        if t.dim() == 2: t = t.unsqueeze(-1)
        if mask.dim() == 3: mask = mask.squeeze(-1)
        mask = mask.bool()

        B, L, D = x.shape
        H, Dh = self.n_heads, self.d_hidden

        # FiLM
        gamma, beta = self.t_mlp(t).chunk(2, dim=-1)     # [B,L,D]
        x_t = gamma * x + beta                           # [B,L,D]

        # 内容打分
        proj = self.W(x_t).view(B, L, H, Dh)
        proj = torch.tanh(proj + self.b.view(1, 1, H, Dh))
        content = torch.einsum('blhd,hd->blh', proj, self.v)    # [B,L,H]

        # 时间偏置
        if self.bias_type == "linear":
            phi = self._time_bias(t, mask)                        # [B,L]
            bias = (phi.unsqueeze(-1) * self.g.view(1, 1, H))     # [B,L,H]
        else:  # "log"
            phi_h = self._time_bias(t, mask)                      # [B,L,H]
            bias = phi_h * self.g.view(1, 1, H)                   # [B,L,H]

        e = content + bias                                        # [B,L,H]
        e_h = e.transpose(1, 2)                                   # [B,H,L]
        alpha = self._masked_softmax(e_h, mask, dim=-1)           # [B,H,L]

        # 聚合
        x_exp = x_t.unsqueeze(2)                                  # [B,L,1,D]
        alpha_exp = alpha.transpose(1, 2).unsqueeze(-1)           # [B,L,H,1]
        z_heads = torch.sum(alpha_exp * x_exp, dim=1)             # [B,H,D]

        if self.agg == "concat":
            pooled = z_heads.reshape(B, H * D)                    # [B,H*D]
        elif self.agg == "mean":
            pooled = z_heads.mean(dim=1)                          # [B,D]
        else:
            raise ValueError(f"Unknown agg: {self.agg}")

        # =============== 可控融合 ===============
        if self.fuse_mode == "none":
            out = pooled
        else:
            if last_token is None:
                # 若未显式提供 last_token，这里用 mask 取每个样本的最后有效 x_t 作为退路
                # idx: [B] 为每个样本最后一个 True 的位置
                idx = mask.long().argmax(dim=1)  # 注意：如果 mask 是 [True...True, False...] 结构，argmax 给 0
                # 更稳妥的方式（如果你的 mask 是前缀 True）：用长度减一
                lengths = mask.long().sum(dim=1) - 1               # [B]
                last_token = x_t[torch.arange(B, device=x.device), lengths]  # [B,D]

            if self.use_ln:
                lt = self.ln_last(last_token)
                pt = self.ln_attn(pooled)
            else:
                lt, pt = last_token, pooled

            if self.fuse_mode == "add":
                if self.use_var_scale:
                    # 方差匹配 s
                    # 在 batch 维做方差，保持每个 batch 一个标量 s；也可改为通道维度
                    var_lt = lt.var(dim=-1, unbiased=False, keepdim=True) + 1e-8
                    var_pt = pt.var(dim=-1, unbiased=False, keepdim=True) + 1e-8
                    s = torch.sqrt(var_lt / var_pt)               # [B,1]
                    fused = lt + s * pt
                else:
                    fused = lt + pt
                out = self.ln_out(fused) if self.use_ln else fused

            elif self.fuse_mode == "concat":
                fused = torch.cat([lt, pt], dim=-1)
                fused = F.dropout(self.fuse_proj(fused), p=0.1, training=self.training)
                out = self.ln_out(fused) if self.use_ln else fused

            elif self.fuse_mode == "gate":
                logit = self.gate_proj(torch.cat([lt, pt], dim=-1))  # [B,1]
                g = torch.sigmoid(logit)                              # [B,1]
                fused = g * lt + (1 - g) * pt
                out = self.ln_out(fused) if self.use_ln else fused

        if return_score:
            return out, alpha
        return out
