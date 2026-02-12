import logging
import torch

logger = logging.getLogger(__name__)

def watch_tensor(name, tensor):
    def hook_fn(grad):
        if torch.isnan(grad).any():
            logger.error("NaN in gradient of %s", name)
        elif torch.isinf(grad).any():
            logger.error("Inf in gradient of %s", name)
        else:
            logger.debug(
                "Gradient of %s OK: min=%.5f, max=%.5f",
                name,
                grad.min().item(),
                grad.max().item(),
            )
    tensor.register_hook(hook_fn)

def check_tensor_anomaly(tensor: torch.Tensor, name="tensor"):
    logger.info("[%s] shape: %s", name, tensor.shape)
    if torch.isnan(tensor).any():
        logger.warning("contains NaN")
    if torch.isinf(tensor).any():
        logger.warning("contains Inf")
    logger.info("min: %.4e", tensor.min().item())
    logger.info("max: %.4e", tensor.max().item())
    logger.info("mean: %.4e", tensor.mean().item())
    logger.info("std: %.4e", tensor.std().item())
