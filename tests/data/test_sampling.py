import torch

from src.data.sampling import (
    EpochAwareWeightedRandomSampler,
    build_balanced_epoch_sampler,
    set_loader_epoch,
)


class _DummyDataset:
    def __init__(self, labels):
        self._labels = list(labels)

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return None, self._labels[idx]


class _DummyLoader:
    def __init__(self, sampler):
        self.sampler = sampler


def test_epoch_aware_sampler_is_deterministic_for_same_seed_and_epoch():
    sampler_a = EpochAwareWeightedRandomSampler([1.0, 2.0, 3.0, 4.0], seed=7)
    sampler_b = EpochAwareWeightedRandomSampler([1.0, 2.0, 3.0, 4.0], seed=7)

    sampler_a.set_epoch(3)
    sampler_b.set_epoch(3)

    assert sampler_a.current_seed == 10
    assert sampler_b.current_seed == 10
    assert list(iter(sampler_a)) == list(iter(sampler_b))


def test_build_balanced_epoch_sampler_exposes_epoch_control():
    dataset = _DummyDataset([0, 0, 1, 1, 1, 1])
    sampler = build_balanced_epoch_sampler(dataset, seed=11)

    assert isinstance(sampler, EpochAwareWeightedRandomSampler)
    assert len(sampler) == len(dataset)

    loader = _DummyLoader(sampler)
    set_loader_epoch(loader, 4)
    assert sampler.current_seed == 15


def test_set_loader_epoch_ignores_missing_sampler():
    loader = _DummyLoader(sampler=None)
    set_loader_epoch(loader, 2)

