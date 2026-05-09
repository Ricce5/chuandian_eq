import logging

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


class EpochAwareWeightedRandomSampler(Sampler[int]):
    def __init__(
        self,
        weights,
        num_samples=None,
        replacement: bool = True,
        seed: int = 0,
    ):
        weights_tensor = torch.as_tensor(weights, dtype=torch.double)
        if weights_tensor.ndim != 1:
            raise ValueError("weights must be a 1D sequence")
        if weights_tensor.numel() == 0:
            raise ValueError("weights cannot be empty")
        if torch.any(weights_tensor < 0):
            raise ValueError("weights must be non-negative")

        self.weights = weights_tensor
        self.num_samples = int(num_samples) if num_samples is not None else int(weights_tensor.numel())
        self.replacement = bool(replacement)
        self.seed = int(seed)
        self.epoch = 0

        if not self.replacement and self.num_samples > self.weights.numel():
            raise ValueError("num_samples must not exceed number of weights when replacement=False")

    @property
    def current_seed(self) -> int:
        return self.seed + self.epoch

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_generator(self) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(self.current_seed)
        return generator

    def __iter__(self):
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=self._make_generator(),
        )
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def build_balanced_epoch_sampler(dataset, seed: int = 0, replacement: bool = True):
    labels = [label for _, label in dataset]
    if len(labels) == 0:
        raise ValueError("Cannot build a sampler from an empty dataset.")

    labels_tensor = torch.as_tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(labels_tensor)
    class_weights = torch.zeros_like(class_counts, dtype=torch.double)
    valid_classes = class_counts > 0
    class_weights[valid_classes] = 1.0 / class_counts[valid_classes].double()
    sample_weights = class_weights[labels_tensor]

    return EpochAwareWeightedRandomSampler(
        sample_weights,
        num_samples=len(labels),
        replacement=replacement,
        seed=seed,
    )


def set_loader_epoch(data_loader, epoch: int) -> None:
    sampler = getattr(data_loader, "sampler", None)
    if sampler is None:
        return
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)
        logger.debug("Set sampler epoch to %s", epoch)
