#%%
import os
import sys
import numpy as np
import torch
import pytest
# Ensure the repository root is on sys.path so `src` package can be imported.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.sequence import Sequence, EventSequence
#%%

def test_sequence_basic():
    # inter_times length = num_events + 1
    inter_times = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    seq = Sequence(inter_times, t_start=0.0)

    # arrival_times = cumsum(inter_times)[:-1]
    expected_arrivals = np.cumsum(inter_times)[:-1]
    assert np.allclose(seq.arrival_times.numpy(), expected_arrivals)

    # number of events
    assert seq.num_events == len(expected_arrivals)

    # t_end = sum(inter_times) + t_start
    assert pytest.approx(seq.t_end) == float(inter_times.sum())


def test_sequence_with_time_series_and_slice():
    inter_times = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)  # 3 events + survival
    t_start = 0.0
    seq_length = 41
    ts = np.linspace(0.0, 4.0, seq_length)
    values = np.sin(2 * np.pi * ts / 4.0)  # some waveform

    seq = Sequence(inter_times, t_start=t_start, time_series=values, time_series_times=ts)

    # original time series stored
    assert 'time_series' in seq
    assert 'time_series_times' in seq
    assert seq['time_series'].shape[0] == seq_length
    assert seq['time_series_times'].shape[0] == seq_length

    # slice between 1.0 and 3.0
    sub = seq.get_subsequence(1.0, 3.0)
    assert sub.t_start == pytest.approx(1.0)
    # duration should be 2.0
    assert pytest.approx(float(sub.inter_times.sum().item())) == 2.0

    # sliced time series times fall within [1, 3]
    if 'time_series_times' in sub:
        times = sub['time_series_times'].numpy()
        assert np.all(times >= 1.0 - 1e-8)
        assert np.all(times <= 3.0 + 1e-8)


def test_to_event_sequence_preserves_time_series():
    inter_times = np.array([0.5, 0.5, 0.5], dtype=float)  # 2 events + survival
    ts = np.linspace(0.0, 1.5, 16)
    values = np.cos(2 * np.pi * ts / 1.5)

    seq = Sequence(inter_times, t_start=0.0, time_series=values, time_series_times=ts)
    evt = seq.to_event_sequence()

    # EventSequence should carry time_series through
    assert hasattr(evt, 'time_series')
    assert hasattr(evt, 'time_series_times')
    assert evt.time_series.shape[0] == len(values)
    assert evt.time_series_times.shape[0] == len(ts)

    # arrival_times in EventSequence should equal inter_times[:-1] cumsum with first set to 0
    expected_inter = torch.as_tensor(inter_times[:-1].copy())
    expected_inter[0] = 0.0
    expected_arr = expected_inter.cumsum(dim=0)
    assert np.allclose(evt.arrival_times.numpy(), expected_arr.numpy())

# pytest -q tests/data/test_sequence.py

#%%
inter_times = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)  # 3 events + survival
t_start = 0.0
seq_length = 41
ts = np.linspace(0.0, 4.0, seq_length)
values = np.sin(2 * np.pi * ts / 4.0)  # some waveform

seq = Sequence(inter_times, t_start=t_start, time_series=values, time_series_times=ts)

# original time series stored
assert 'time_series' in seq
assert 'time_series_times' in seq
assert seq['time_series'].shape[0] == seq_length
assert seq['time_series_times'].shape[0] == seq_length

# slice between 1.0 and 3.0
sub = seq.get_subsequence(1.0, 3.0)
assert sub.t_start == pytest.approx(1.0)
# duration should be 2.0
assert pytest.approx(float(sub.inter_times.sum().item())) == 2.0

# %%
