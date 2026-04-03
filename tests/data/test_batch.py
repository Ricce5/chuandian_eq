import os
import sys
import numpy as np
import torch
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.sequence import Sequence
from src.data.batch import Batch
from src.data.constants import PAD


def make_seq(inter_times, t_start=0.0, time_series=None, time_series_times=None):
    return Sequence(inter_times=inter_times, t_start=t_start, time_series=time_series, time_series_times=time_series_times)


def test_batch_from_list_without_time_series():
    seqs = [make_seq([1.0, 2.0]), make_seq([0.5, 1.5, 2.5])]
    batch = Batch.from_list(seqs)
    # No time_series field expected
    assert 'time_series' not in batch
    assert 'time_series_times' not in batch
    assert batch.inter_times.shape[0] == 2


def test_batch_from_list_with_time_series_all_present():
    s1_ts = np.linspace(0.0, 2.0, 5)
    s1_vals = np.arange(5.0)
    s2_ts = np.linspace(0.0, 3.0, 7)
    s2_vals = np.arange(7.0) * 2.0

    seqs = [make_seq([1.0, 1.0, 1.0], time_series=s1_vals, time_series_times=s1_ts),
            make_seq([1.0, 1.0, 1.0], time_series=s2_vals, time_series_times=s2_ts)]

    batch = Batch.from_list(seqs)
    assert 'time_series' in batch
    assert 'time_series_times' in batch
    ts = batch['time_series']
    ts_times = batch['time_series_times']
    assert ts.shape[0] == 2
    # padded length should be max(5,7)=7 along dim1
    assert ts.shape[1] == 7
    # values preserved in prefix
    assert torch.allclose(ts[0, :5], torch.as_tensor(s1_vals, dtype=ts.dtype))
    assert torch.allclose(ts[1, :7], torch.as_tensor(s2_vals, dtype=ts.dtype))
    # padded positions equal PAD
    assert torch.all(ts[0, 5:] == PAD)


def test_batch_multivariate_time_series_all_present():
        # Two sequences, each with multivariate time series (T, C)
        s1_ts = np.linspace(0.0, 2.0, 5)
        s1_vals = np.vstack([np.arange(5.0), np.arange(5.0) + 10.0]).T  # shape (5,2)
        s2_ts = np.linspace(0.0, 3.0, 7)
        s2_vals = np.vstack([np.arange(7.0) * 2.0, np.arange(7.0) * -1.0]).T  # shape (7,2)

        seqs = [make_seq([1.0, 1.0, 1.0], time_series=s1_vals, time_series_times=s1_ts),
            make_seq([1.0, 1.0, 1.0], time_series=s2_vals, time_series_times=s2_ts)]

        batch = Batch.from_list(seqs)
        assert 'time_series' in batch
        ts = batch['time_series']
        # shape should be (batch, max_T, C)
        assert ts.shape[0] == 2
        assert ts.shape[1] == 7
        assert ts.shape[2] == 2

        # check that first two channels preserved in prefix
        assert torch.allclose(ts[0, :5, 0], torch.as_tensor(s1_vals[:, 0], dtype=ts.dtype))
        assert torch.allclose(ts[0, :5, 1], torch.as_tensor(s1_vals[:, 1], dtype=ts.dtype))
        assert torch.allclose(ts[1, :7, 0], torch.as_tensor(s2_vals[:, 0], dtype=ts.dtype))
        assert torch.allclose(ts[1, :7, 1], torch.as_tensor(s2_vals[:, 1], dtype=ts.dtype))


def test_batch_from_list_with_time_series_some_missing():
    s1_ts = np.linspace(0.0, 2.0, 4)
    s1_vals = np.arange(4.0)

    seqs = [make_seq([1.0, 1.0], time_series=s1_vals, time_series_times=s1_ts),
            make_seq([0.5, 0.5, 0.5])]  # second seq has no time_series

    batch = Batch.from_list(seqs)
    assert 'time_series' in batch
    ts = batch['time_series']
    ts_times = batch['time_series_times']
    # padded length = 4
    assert ts.shape[1] == 4
    # first row preserved
    assert torch.allclose(ts[0, :4], torch.as_tensor(s1_vals, dtype=ts.dtype))
    # second row should be all PAD (padded empty tensor)
    assert torch.all(ts[1, :] == PAD)


def test_nll_start_idx_zero_length_window_is_end_idx():
    # inter_times -> arrival_times [1.0, 3.0], t_end=6.0
    seq = Sequence(inter_times=[1.0, 2.0, 3.0], t_start=0.0, t_nll_start=6.0)
    batch = Batch.from_list([seq])

    assert int(batch.start_idx.item()) == int(batch.end_idx.item())
    assert float(batch.nll_event_mask.sum().item()) == 0.0


def test_nll_start_idx_regular_window_keeps_first_future_event():
    # inter_times -> arrival_times [1.0, 3.0], t_end=6.0, first event > 1.5 is at index 1.
    seq = Sequence(inter_times=[1.0, 2.0, 3.0], t_start=0.0, t_nll_start=1.5)
    batch = Batch.from_list([seq])

    assert int(batch.start_idx.item()) == 1
    assert int(batch.end_idx.item()) == 2
    assert float(batch.nll_event_mask.sum().item()) == 1.0


