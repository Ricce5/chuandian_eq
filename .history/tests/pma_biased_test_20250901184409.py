import torch
from src.models.extractors.pma_time_biased import TimeBiasedPMA
B,L,D = 2, 6, 8
H,r   = 4, 2
x = torch.randn(B,L,D)
mask = torch.ones(B,L, dtype=torch.bool); mask[1,4:] = False
t01  = torch.linspace(0.2, 1.0, L).unsqueeze(0).repeat(B,1); t01[1,4:] = 0.0



pma_t = TimeBiasedPMA(d_model=D, n_heads=H, r=r, agg="mean",
                      use_film=True, bias_type="log", alpha0=10.0)
out2, a2 = pma_t(x, mask, extra_inputs={"event_time": t01}, return_score=True)
print("TimeBiasedPMA:", out2.shape, a2.shape)  # (2,8), (2,4,2,6)
print(a2)
