import os
import sys
import warnings

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import src.utils.catalog_utils as catalog_utils
from src.data.sequence import Sequence
from src.data.tpp_dataset import TppDataset


def make_sequence_from_arrivals(arrival_times, t_start=0.0, t_nll_start=0.0, t_end=None):
    arrival_times = np.asarray(arrival_times, dtype=np.float32)
    if t_end is None:
        t_end = float(arrival_times[-1] + 1.0)
    inter_times = Sequence.compute_inter_times(arrival_times, t_start=t_start, t_end=t_end)
    mag = torch.ones(len(arrival_times), dtype=torch.float32)
    return Sequence(
        inter_times=torch.tensor(inter_times, dtype=torch.float32),
        t_start=t_start,
        t_nll_start=t_nll_start,
        mag=mag,
    )


def test_split_sequence_uses_ceil_when_single_window_would_be_too_large():
    history = np.linspace(1.0, 40.0, 100, dtype=np.float32)
    forecast = np.linspace(50.0, 100.0, 567, dtype=np.float32)
    seq = make_sequence_from_arrivals(np.concatenate([history, forecast]), t_nll_start=50.0, t_end=101.0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found 1 zero inter-event times.*")
        dataset = catalog_utils.split_sequence(seq, mean_batch_size=300, max_events=512)

    assert len(dataset) == 2
    assert all(subseq.num_events <= 512 for subseq in dataset)
    assert sum(subseq.num_nll_events for subseq in dataset) == seq.num_nll_events


def test_split_sequence_refines_dense_windows_instead_of_skipping_them():
    history = np.linspace(1.0, 48.0, 120, dtype=np.float32)
    dense_cluster = np.linspace(50.0, 50.2, 600, dtype=np.float32)
    sparse_tail = np.linspace(60.0, 70.0, 60, dtype=np.float32)
    seq = make_sequence_from_arrivals(
        np.concatenate([history, dense_cluster, sparse_tail]),
        t_nll_start=50.0,
        t_end=71.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found 1 zero inter-event times.*")
        dataset = catalog_utils.split_sequence(seq, mean_batch_size=300, max_events=512)

    assert len(dataset) >= 3
    assert all(subseq.num_events <= 512 for subseq in dataset)
    assert sum(subseq.num_nll_events for subseq in dataset) == seq.num_nll_events


def test_split_sequence_without_max_events_preserves_window_start():
    history = np.linspace(1.0, 48.0, 120, dtype=np.float32)
    dense_cluster = np.linspace(50.0, 50.2, 600, dtype=np.float32)
    sparse_tail = np.linspace(60.0, 70.0, 60, dtype=np.float32)
    seq = make_sequence_from_arrivals(
        np.concatenate([history, dense_cluster, sparse_tail]),
        t_nll_start=50.0,
        t_end=71.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found 1 zero inter-event times.*")
        dataset = catalog_utils.split_sequence(seq, mean_batch_size=2000, max_events=None)

    assert len(dataset) == 1
    assert dataset[0].t_start == seq.t_nll_start
    assert dataset[0].t_nll_start == seq.t_nll_start
    assert dataset[0].num_events > 512


def test_split_sequence_supports_explicit_num_splits_for_nll_range():
    history = np.linspace(1.0, 48.0, 120, dtype=np.float32)
    forecast = np.linspace(50.0, 70.0, 200, dtype=np.float32)
    seq = make_sequence_from_arrivals(
        np.concatenate([history, forecast]),
        t_nll_start=50.0,
        t_end=71.0,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found 1 zero inter-event times.*")
        dataset = catalog_utils.split_sequence(
            seq,
            mean_batch_size=300,
            max_events=None,
            num_splits=4,
        )

    expected_starts = np.linspace(seq.t_nll_start, seq.t_end, 5)[:-1]
    assert len(dataset) == 4
    assert [subseq.t_nll_start for subseq in dataset] == list(expected_starts)


def test_split_sequence_rejects_non_positive_num_splits():
    seq = make_sequence_from_arrivals(np.linspace(1.0, 10.0, 12, dtype=np.float32), t_nll_start=5.0, t_end=11.0)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Found 1 zero inter-event times.*")
        with np.testing.assert_raises(ValueError):
            catalog_utils.split_sequence(seq, num_splits=0)


def test_split_minibatches_falls_back_to_original_sequence_for_empty_split(monkeypatch):
    train_seq = make_sequence_from_arrivals(np.linspace(1.0, 10.0, 12, dtype=np.float32), t_nll_start=5.0, t_end=11.0)
    val_seq = make_sequence_from_arrivals(np.linspace(1.0, 8.0, 10, dtype=np.float32), t_nll_start=4.0, t_end=9.0)
    test_seq = make_sequence_from_arrivals(np.linspace(1.0, 12.0, 14, dtype=np.float32), t_nll_start=6.0, t_end=13.0)

    def fake_split_sequence(seq, mean_batch_size, max_events):
        if seq is val_seq:
            return TppDataset([])
        return TppDataset([seq])

    monkeypatch.setattr(catalog_utils, "split_sequence", fake_split_sequence)

    catalog = catalog_utils.DummyCatalog(
        train=TppDataset([train_seq]),
        val=TppDataset([val_seq]),
        test=TppDataset([test_seq]),
        metadata={},
        full_sequence=test_seq,
    )

    split_catalog = catalog_utils.split_minibatches(catalog, mean_batch_size=300, max_events=512)

    assert len(split_catalog.train) == 1
    assert len(split_catalog.val) == 1
    assert len(split_catalog.test) == 1
    assert split_catalog.val[0] is val_seq
