import torch


def sample_noise_like(x: torch.Tensor, scale: float, noise_type: str) -> torch.Tensor:
    if noise_type == "uniform":
        return (torch.rand_like(x) - 0.5) * scale
    return torch.randn_like(x) * scale


def augment_tpp_batch_inplace(batch, model) -> None:
    """Apply optional training-time perturbation on magnitude only."""
    mag_noise_scale = float(getattr(model, "mag_noise_scale", 0.0) or 0.0)
    if mag_noise_scale <= 0.0:
        return

    mask = getattr(batch, "input_mask", None)
    assert  mask is not None, "To use mag_noise augmentation, the batch must have an 'input_mask' attribute to specify valid positions for noise application."
    if mag_noise_scale > 0.0 and hasattr(batch, "mag"):
        mag_noise_type = getattr(model, "mag_noise_type", "gaussian")
        mag_noise = sample_noise_like(batch.mag, mag_noise_scale, mag_noise_type)
        batch.mag = batch.mag + mag_noise * mask
