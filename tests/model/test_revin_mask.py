import torch
from src.models.layers.revin import RevIN


def masked_stats(x: torch.Tensor, mask: torch.Tensor):
    """
    Compute mean and std over valid positions indicated by mask.
    x: (B, L, C)
    mask: (B, L) or (B, L, 1), bool/float where True/1.0 is valid.
    Returns (mean, std) with shape (B, 1, C).
    """
    if mask.dtype != torch.float32 and mask.dtype != torch.float64:
        mask_f = mask.float()
    else:
        mask_f = mask
    # expand to (B, L, 1) if needed
    if mask_f.ndim == 2:
        mask_f = mask_f.unsqueeze(-1)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask_f).sum(dim=1, keepdim=True) / denom
    var = ((x - mean) ** 2 * mask_f).sum(dim=1, keepdim=True) / denom
    std = torch.sqrt(var + 1e-5)
    return mean, std


def test_revin_masked_normalization_and_denorm():
    device = torch.device("cpu")
    B, L, C = 2, 6, 1

    # Construct input with obvious per-sample stats and padding at tail
    x0 = torch.tensor([[1., 2., 3., 4., 0., 0.],
                       [10., 12., 14., 0., 0., 0.]], dtype=torch.float32).unsqueeze(-1)  # (B=2, L=6, C=1)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0, 0]], dtype=torch.bool)  # (B, L)

    revin = RevIN(num_features=C, affine=True).to(device)

    # norm with mask
    y = revin(x0, mode='norm', mask=mask)

    # expected masked stats
    mean_exp, std_exp = masked_stats(x0, mask)

    # Check masked mean≈0, std≈1 on valid positions
    valid = mask.unsqueeze(-1)
    y_valid = y[valid]
    # Allow small numerical tolerance
    assert torch.allclose(y_valid.mean(), torch.zeros_like(y_valid.mean()), atol=1e-4, rtol=1e-4)
    assert torch.allclose(y_valid.std(unbiased=False), torch.ones_like(y_valid.std(unbiased=False)), atol=1e-3, rtol=1e-3)

    # Denorm should recover original x at valid positions (and keep pads at 0)
    x_rec = revin(y, mode='denorm')
    assert torch.allclose(x_rec, x0, atol=1e-5, rtol=1e-5)


def test_revin_mask_types_and_shapes():
    B, L, C = 1, 5, 2
    x = torch.arange(B * L * C, dtype=torch.float32).view(B, L, C)
    # mask as float (B, L, 1)
    mask_f = torch.tensor([[1., 1., 1., 0., 0.]]).unsqueeze(-1)
    revin = RevIN(num_features=C, affine=False)
    y = revin(x, 'norm', mask=mask_f)
    x_rec = revin(y, 'denorm')
    assert torch.allclose(x_rec, x, atol=1e-6)

    # mask as bool (B, L)
    mask_b = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    y2 = revin(x, 'norm', mask=mask_b)
    x_rec2 = revin(y2, 'denorm')
    assert torch.allclose(x_rec2, x, atol=1e-6)


def test_revin_all_pad_sample_safe():
    """Ensure no NaN when a sample has all pads (denom clamp)."""
    B, L, C = 2, 4, 1
    x = torch.tensor([[[0.], [0.], [0.], [0.]],
                      [[1.], [2.], [3.], [0.]]], dtype=torch.float32)
    mask = torch.tensor([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    revin = RevIN(num_features=C, affine=True)
    y = revin(x, 'norm', mask=mask)
    assert torch.isfinite(y).all()
    x_rec = revin(y, 'denorm')
    assert torch.isfinite(x_rec).all()
