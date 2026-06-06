import torch

from src.models.tpp.etas import ETAS


def test_set_params_mu_zero_keeps_log_mu_finite():
    model = ETAS(
        base_rate_init=0.26,
        fix_mu=False,
        device=torch.device("cpu"),
        enforce_subcritical=False,
        enforce_p_gt_one=False,
    )
    model.set_params(mu=0.0)
    assert torch.isfinite(model.log_mu).item()
    tiny = torch.finfo(model.log_mu.dtype).tiny
    assert model.mu.item() > 0.0
    assert torch.isclose(model.mu.detach(), torch.tensor(tiny, dtype=model.mu.dtype)).item()
