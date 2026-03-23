import os
import sys

import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.sequence import Sequence
from src.data.tpp_dataset import TppDataset


def make_sequence():
    return Sequence(
        inter_times=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=2.5,
        mag=torch.tensor([2.1, 2.2, 2.3], dtype=torch.float32),
    )


def test_sequence_drop_events_keeps_valid_nll_region():
    seq = make_sequence()

    thinned = seq.drop_events(
        1.0,
        generator=torch.Generator().manual_seed(0),
        min_nll_events=1,
    )

    assert thinned.t_start == pytest.approx(seq.t_start)
    assert thinned.t_end == pytest.approx(seq.t_end)
    assert thinned.t_nll_start == pytest.approx(seq.t_nll_start)
    assert thinned.num_events == 1
    assert thinned.num_nll_events == 1
    assert float(thinned.inter_times.sum().item()) == pytest.approx(seq.t_end - seq.t_start)
    assert thinned.arrival_times[0].item() in {3.0, 6.0}
    assert thinned.mag.shape[0] == thinned.num_events


def test_tpp_dataset_sequence_transform_returns_valid_batch():
    seq = make_sequence()

    def transform(sequence):
        return sequence.drop_events(
            1.0,
            generator=torch.Generator().manual_seed(0),
            min_nll_events=1,
        )

    dataset = TppDataset([seq], sequence_transform=transform)
    sample = dataset[0]
    assert sample.num_nll_events >= 1

    batch = next(iter(dataset.get_dataloader(batch_size=1, shuffle=False)))
    assert batch.nll_event_mask.sum().item() >= 1
    assert batch.input_mask.sum().item() >= 1
    assert batch.t_end[0].item() == pytest.approx(seq.t_end)
