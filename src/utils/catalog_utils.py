# Reference: https://zenodo.org/records/8161777 - Using Deep Learning for Flexible and Scalable Earthquake Forecasting.
import logging
import pandas as pd
import src
from src.data import Sequence
from dataclasses import dataclass
import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def train_test_split_sequence(
    seq: Sequence,
    start_ts: pd.Timestamp,
    train_start_ts: pd.Timestamp,
    test_start_ts: pd.Timestamp,
    freq: pd.Timedelta = pd.Timedelta("1 day"),
):
    """Generate train and test subsequences.
    Original sequence with events in [start_ts, end_ts] is split into 2 parts:
    1) train: Includes events in [start_ts, test_start_ts], t_nll_start = start_ts
    2) test: Includes events in [start_ts, end_ts], t_nll_start = test_start_ts
    """
    # Start of the train / val / test intervals as timestamps
    freq_td = pd.Timedelta(freq)

    assert (train_start_ts is None or train_start_ts <= test_start_ts), f"train_start_ts{train_start_ts} must be <= test_start_ts{test_start_ts}"
    assert (test_start_ts >= start_ts), f"test_start_ts{test_start_ts} must be >= start_ts{start_ts}"
    t_test_start = (test_start_ts - start_ts) / freq_td
    assert (seq.t_end >= float(t_test_start)-1e-3), f"Sequence end time {seq.t_end} must be >= test_start_ts {t_test_start}"
    start_ts = pd.Timestamp(start_ts)
    if train_start_ts is not None:
        train_start_ts = pd.Timestamp(train_start_ts)
        t_train_start = (train_start_ts - start_ts) / freq
    else:
        t_train_start = seq.t_start

    test_start_ts = pd.Timestamp(test_start_ts)
    t_test_start = (test_start_ts - start_ts) / freq
    # 数值误差可能导致 t_test_start 比 seq.t_end 略大，做一下截断
    t_test_start = min(float(t_test_start), float(seq.t_end))
    freq = pd.Timedelta(freq)

    # Start of the train / val / test intervals as floats
    seq_train = seq.get_subsequence(seq.t_start, t_test_start)
    seq_train.t_nll_start = t_train_start
  
    seq_test = seq.get_subsequence(seq.t_start, seq.t_end)
    seq_test.t_nll_start = t_test_start
    return seq_train, seq_test




def train_val_test_split_sequence(
    seq: Sequence,
    start_ts: pd.Timestamp,
    train_start_ts: pd.Timestamp,
    val_start_ts: pd.Timestamp,
    test_start_ts: pd.Timestamp,
    freq: pd.Timedelta = pd.Timedelta("1 day"),
):
    """Generate train, validation and test subsequences.
    Original sequence with events in [start_ts, end_ts] is split into 3 parts:
    1) train: Includes events in [start_ts, val_start_ts], t_nll_start = start_ts
    2) val: Includes events in [start_ts, test_start_ts], t_nll_start = val_start_ts 
    3) test: Includes events in [start_ts, end_ts], t_nll_start = test_start_ts
    """
    # Start of the train / val / test intervals as timestamps
    start_ts = pd.Timestamp(start_ts)
    if train_start_ts is not None:
        train_start_ts = pd.Timestamp(train_start_ts)
        t_train_start = (train_start_ts - start_ts) / freq
    else:
        t_train_start = seq.t_start

    val_start_ts = pd.Timestamp(val_start_ts)
    t_val_start = (val_start_ts - start_ts) / freq

    test_start_ts = pd.Timestamp(test_start_ts)
    t_test_start = (test_start_ts - start_ts) / freq
    freq = pd.Timedelta(freq)

    # Start of the train / val / test intervals as floats
    seq_train = seq.get_subsequence(seq.t_start, t_val_start)
    seq_train.t_nll_start = t_train_start
  
    seq_val = seq.get_subsequence(seq.t_start, t_test_start)
    seq_val.t_nll_start = t_val_start
    seq_test = seq.get_subsequence(seq.t_start, seq.t_end)
    seq_test.t_nll_start = t_test_start
    return seq_train, seq_val, seq_test


def train_val_test_split_sequence_float(
    seq: Sequence,
    start_ts: float,
    train_start_ts: float,
    val_start_ts: float,
    test_start_ts: float,
):
    """
    Split the sequence into training, validation, and testing sets using float timestamps.

    Parameters:
    - seq: Input Sequence object
    - start_ts: Overall start time (float)
    - train_start_ts: Training NLL start time (float, relative to start_ts)
    - val_start_ts: Validation NLL start time (float, relative to start_ts)
    - test_start_ts: Testing NLL start time (float, relative to start_ts)

    Returns:
    - seq_train: Training set (NLL starts at train_start_ts)
    - seq_val: Validation set (NLL starts at val_start_ts)
    - seq_test: Testing set (NLL starts at test_start_ts)
    """

    t_train_start = train_start_ts if train_start_ts is not None else seq.t_start
    t_val_start = val_start_ts
    t_test_start = test_start_ts

    seq_train = seq.get_subsequence(seq.t_start, val_start_ts)
    seq_train.t_nll_start = t_train_start

    seq_val = seq.get_subsequence(seq.t_start, test_start_ts)
    seq_val.t_nll_start = t_val_start

    seq_test = seq.get_subsequence(seq.t_start, seq.t_end)
    seq_test.t_nll_start = t_test_start

    return seq_train, seq_val, seq_test


@dataclass
class DummyCatalog(src.data.Catalog):
    train: src.data.TppDataset
    val: src.data.TppDataset
    test: src.data.TppDataset
    metadata: dict
    full_sequence: src.data.Sequence


def trim_train_and_test(
    catalog: src.data.Catalog,
    train_frac: float,
    val_frac: float,
) -> src.data.Catalog:
    """Shorten the train and val sequences.

    Args:
        catalog: Catalog, where the sequences must be shortened.
        train_frac: Fraction of the train sequence used for training.
        val_frac: Fraction of the val sequence used for training.
    """
    if not len(catalog.train) == len(catalog.val) == len(catalog.test) == 1:
        raise ValueError("Expected # of train/val/test sequences to be 1")
    if not 0 < train_frac <= 1:
        raise ValueError(f"train_frac must be in (0, 1] (got {train_frac})")
    if not 0 < val_frac <= 1:
        raise ValueError(f"val_frac must be in (0, 1] (got {val_frac})")
    # Cut the training sequence
    train_seq = catalog.train[0]
    train_duration = train_seq.t_end - train_seq.t_start
    new_train_start = train_seq.t_end - train_duration * train_frac
    new_train_seq = train_seq.get_subsequence(new_train_start, train_seq.t_end)
    new_train_seq.t_nll_start = new_train_start

    # Cut the validation sequence
    val_seq = catalog.val[0]
    val_duration = val_seq.t_end - val_seq.t_nll_start
    new_val_end = val_seq.t_nll_start + val_duration * val_frac
    new_val_seq = val_seq.get_subsequence(new_train_start, new_val_end)
    new_catalog = DummyCatalog(
        train=src.data.TppDataset([new_train_seq]),
        val=src.data.TppDataset([new_val_seq]),
        test=catalog.test,
        metadata=catalog.metadata.copy(),
        full_sequence=catalog.full_sequence,
    )
    return new_catalog

def find_t_start_from_t_end(seq, t_nll_start, t_end, max_events=500, min_events_required=10):
    arrival_times = seq.arrival_times  # Tensor
    valid_idx = (arrival_times <= t_end).nonzero(as_tuple=True)[0]

    if len(valid_idx) < min_events_required:
        raise ValueError(f"Only {len(valid_idx)} events before t_end ({t_end}), less than min required ({min_events_required})")

    selected_idx = valid_idx[-max_events:]
    t_start_candidate = arrival_times[selected_idx[0]].item()

    if t_start_candidate > t_nll_start:
        raise ValueError(f"Computed t_start ({t_start_candidate}) is not < t_nll_start ({t_nll_start})")

    return t_start_candidate


def split_sequence(seq, mean_batch_size=300, max_events=30000):
    num_nll_events = (seq.arrival_times >= seq.t_nll_start).sum().item()
    num_splits = max(1, int(num_nll_events / mean_batch_size))
    linspace = np.linspace(seq.t_nll_start, seq.t_end, num_splits + 1)

    short_sequences = []
    window_durations = []  
    skipped = 0

    for start, end in tqdm(zip(linspace[:-1], linspace[1:]), total=len(linspace) - 1):
        try:
            t_start = find_t_start_from_t_end(seq, t_nll_start=start, t_end=end, max_events=max_events)
            duration = end - t_start
            window_durations.append(duration)
            logger.debug("[split_sequence] t_start: %.4f, t_end: %.4f", t_start, end)
            logger.debug("[split_sequence] Window duration: %.4f", duration)

            short_seq = seq.get_subsequence(t_start, end)
            # print(f"[split_sequence] Short sequence from {t_start} to {end}, NLL start at {start}")
            short_seq.t_nll_start = max(start, short_seq.t_start)
            logger.debug("[split_sequence] Short sequence NLL start set to %.4f", short_seq.t_nll_start)
            # print(f"[split_sequence] Short sequence {short_seq}")
            short_sequences.append(short_seq)
        except ValueError as e:
            skipped += 1
            continue

    if skipped > 0:
        logger.warning("[split_sequence] Skipped %s windows due to too few events or invalid t_start.", skipped)

    if window_durations:
        mean_duration = np.mean(window_durations)
        logger.info("[split_sequence] Average window duration: %.4f", mean_duration)

    return src.data.TppDataset(short_sequences)



def split_minibatches(catalog: src.data.Catalog,mean_batch_size:int,max_events: int=30000) -> src.data.Catalog:
    if not len(catalog.train) == len(catalog.val) == len(catalog.test) == 1:
        raise ValueError("Expected # of train/val/test sequences to be 1")
    d_train = split_sequence(catalog.train[0], mean_batch_size,max_events)
    d_val = split_sequence(catalog.val[0], mean_batch_size,max_events)
    d_test = split_sequence(catalog.test[0], mean_batch_size,max_events)
    return DummyCatalog(
        train=d_train, val=d_val, test=d_test, metadata=catalog.metadata,full_sequence=catalog.full_sequence
    )

