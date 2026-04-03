# Enhanced Batch to handle sample updates and convert to event batches used in EasyTPP.
# Reference: https://zenodo.org/records/8161777 - Using Deep Learning for Flexible and Scalable Earthquake Forecasting.
from typing import List, Optional

import numpy as np
import torch

from .dot_dict import DotDict
from .sequence import Sequence,EventSequence
from .constants import PAD_TOKEN_ID,PAD
from copy import deepcopy

class Batch(DotDict):
    """Batch of padded variable-length sequences.

    Should always be created from a list with Batch.from_list.

    In addition to basic information (such as arrival times) a Batch object contains
    other information that can be useful when computing the NLL of a TPP model, such as
    inter-event times, indices of first and last events in each sequence, and a
    mask indicating which entries correspond to actual events (and not to padding).

    See batch.keys() for the list of all available attributes.

    Attributes:
        inter_times: Inter-event times padded with zeros, shape [batch_size, seq_len]
        arrival_times: Padded arrival times, shape [batch_size, seq_len]
        t_start: Start of the observed interval for each sequence, shape [batch_size]
        t_end: End of the observed interval for each sequence, shape [batch_size]
        t_nll_start: Time from which the NLL is computed for each sequence.
            Defaults to t_start, shape [batch_size]
        mask: Binary mask indicating for which events the NLL must be computed,
            shape [batch_size, seq_len]
        start_idx: Index of the first event in each sequence, for which NLL must
            be computed, shape [batch_size]
        end_idx: Index of the last inter-event time in each sequence
            (survival time from last event t_N to t_end), shape [batch_size]
        **kwargs: Additional attributes associated with each event (e.g., magnitude,
            location), each with shape [batch_size, seq_len, ...].

    """

    default_batch_attrs = {
        "arrival_times",
        "inter_times",
        "t_start",
        "t_end",
        "t_nll_start",
        "start_idx",
        "end_idx",
        "mask",
    }

    @staticmethod
    def from_list(sequences: List[Sequence],pad_token_id: int = PAD_TOKEN_ID,pad: int = PAD, max_sample_len:int=0) -> "Batch":
        """Construct a batch from a list of variable-length sequences."""
        batch_size = len(sequences)
        dtype = sequences[0].arrival_times.dtype
        device = sequences[0].arrival_times.device
        padded_seq_len = max(len(seq.inter_times) for seq in sequences)+max_sample_len

        inter_times = pad_sequence(
            [seq.inter_times for seq in sequences],
            padding_value=pad,  
            max_len=padded_seq_len,
        )

        t_start = torch.zeros(batch_size, dtype=dtype, device=device)
        t_end = torch.zeros(batch_size, dtype=dtype, device=device)
        t_nll_start = torch.zeros(batch_size, dtype=dtype, device=device)
        end_idx = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i, seq in enumerate(sequences):
            t_start[i] = seq.t_start
            t_end[i] = seq.t_end
            t_nll_start[i] = seq.t_nll_start
            end_idx[i] = len(seq.arrival_times)

        # Get index of the first event that happened after t_nll_start
        
        arrival_times = torch.cumsum(inter_times, dim=-1) + t_start[:, None] #  arrival_times的pad位为t_end
        start_idx = get_start_idx(arrival_times, t_nll_start, end_idx=end_idx)
        nll_event_mask = get_mask(inter_times, start_idx, end_idx)
        input_mask =  get_mask(inter_times,torch.zeros_like(end_idx), end_idx) 

        # Handle other attributes (e.g., marks, locations)
        other_attr_names = [
            k
            for k in sequences[0].keys()
            if k not in sequences[0].default_sequence_attrs
        ]
        other_attr = {}
        for name in other_attr_names:
            values = [seq[name] for seq in sequences]
            # Tensors are padded into shape (batch_size, padded_seq_len, ...)
            other_attr[name] = pad_sequence(
                values, padding_value=pad, max_len=padded_seq_len
            )

        # Handle optional continuous time series (time_series, time_series_times)
        # These are not per-event attributes and may have different lengths
        # per sequence. We pad them along the time dimension to the maximum
        # length across the batch and include them as extra batch fields.
        if any(getattr(seq, 'time_series', None) is not None for seq in sequences):
            # Build lists, replacing missing series with empty tensors so padding works.
            ts_list = []
            ts_times_list = []
            for seq in sequences:
                seq_ts = getattr(seq, 'time_series', None)
                seq_ts_times = getattr(seq, 'time_series_times', None)
                if seq_ts is None:
                    ts_list.append(torch.empty(0, dtype=dtype, device=device))
                    ts_times_list.append(torch.empty(0, dtype=dtype, device=device))
                else:
                    ts_list.append(torch.as_tensor(seq_ts))
                    assert seq_ts_times is not None, "If time_series is provided, time_series_times is required."
                    ts_times_list.append(torch.as_tensor(seq_ts_times))

            ts_list = [t.to(device=device, dtype=dtype) for t in ts_list]
            ts_times_list = [t.to(device=device, dtype=dtype) for t in ts_times_list]

            max_ts_len = max(t.size(0) for t in ts_list)
            ts_lengths = torch.tensor([t.size(0) for t in ts_list], device=device, dtype=torch.long)
            ts_mask = torch.arange(max_ts_len, device=device).unsqueeze(0) < ts_lengths.unsqueeze(1)

            ts_padded = pad_sequence(ts_list, padding_value=0.0, max_len=max_ts_len)
            ts_times_padded = pad_sequence(ts_times_list, padding_value=0.0, max_len=max_ts_len)

            # For padded positions, repeat the last valid value to keep the grid monotonic and stable.
            for i, L in enumerate(ts_lengths.tolist()):
                if L == 0:
                    continue
                ts_padded[i, L:] = ts_padded[i, L - 1]
                ts_times_padded[i, L:] = ts_times_padded[i, L - 1]

            other_attr['time_series'] = ts_padded
            other_attr['time_series_times'] = ts_times_padded
            other_attr['time_series_mask'] = ts_mask.float()

        non_pad_mask = (inter_times != pad).float()
        non_pad_mask[:, 0] = 1. 


        if "type_event" in other_attr:
            type_event = other_attr["type_event"]  # shape: [batch_size, padded_seq_len]

            type_seq = torch.full_like(type_event, fill_value=pad_token_id)
            type_seq[non_pad_mask.bool()] = type_event[non_pad_mask.bool()].long()
            # del other_attr["type_event"]
        else:
            type_seq = build_type_seq(non_pad_mask, pad_token_id)


        return Batch(                         
            inter_times=inter_times,
            arrival_times=arrival_times,
            t_start=t_start,
            t_end=t_end,
            t_nll_start=t_nll_start,
            nll_event_mask=nll_event_mask,
            input_mask=input_mask,
            start_idx=start_idx,
            end_idx=end_idx,
            type_seq=type_seq,
            **other_attr,
        )
    
    @staticmethod
    def init_sample_batch(
        batch_size: int,
        max_sample_len: int,
        past_seq: Optional[Sequence] = None,
    ) -> "Batch":
        if past_seq is None:
            raise ValueError("past_seq cannot be None")
        seqs = [deepcopy(past_seq) for _ in range(batch_size)]
        return Batch.from_list(seqs, max_sample_len=max_sample_len)
        

    def get_sample_batch(self) -> "Batch":
        valid_len = (self.end_idx.max().item() + 1)
        ##
        end_idx= valid_len-1
        ##
        core_fields = dict(
            inter_times   = self.inter_times[:, :end_idx],
            arrival_times = self.arrival_times[:, :end_idx],
            # t_start       = self.t_start,
            # t_end         = self.t_end,
            # t_nll_start   = self.t_nll_start,
            # start_idx     = self.start_idx,
            end_idx       = self.end_idx,
            input_mask    = self.input_mask[:, :end_idx],
            # nll_event_mask= self.nll_event_mask[:, :end_idx],
        )
        extras = {k: v[:, :end_idx] for k, v in self.items()
                if k not in self.default_batch_attrs and k not in core_fields}
        return Batch(**core_fields, **extras)
    

    def get_tmp_batch(self) -> "Batch":
        valid_len = (self.end_idx.max().item() + 1)
        start_idx= valid_len-2
        end_idx = valid_len-1
        core_fields = dict(
            inter_times   = self.inter_times[:, start_idx:end_idx],
            arrival_times = self.arrival_times[:, start_idx:end_idx],
            # t_start       = self.t_start,
            # t_end         = self.t_end,
            # t_nll_start   = self.t_nll_start,
            # start_idx     = self.start_idx,
            end_idx       = self.end_idx,
            input_mask    = self.input_mask[:, start_idx:end_idx],
            # nll_event_mask= self.nll_event_mask[:, :end_idx],
        )
        extras = {k: v[:, start_idx:end_idx] for k, v in self.items()
                if k not in self.default_batch_attrs and k not in core_fields}
        return Batch(**core_fields, **extras)
        

    def update_sample_batch(self, next_inter_times: torch.Tensor,
                                next_mag: Optional[torch.Tensor] = None) -> None:
        batch_idx = torch.arange(self.batch_size, device=self.inter_times.device)
        write_idx = self.end_idx                       # (B,)

        if next_inter_times.shape != (self.batch_size, 1):
            raise ValueError(
            f"Expected next_inter_times shape to be ({self.batch_size}, 1), "
            f"but got {next_inter_times.shape}"
            )
            
        if next_mag is not None and next_mag.shape != (self.batch_size, 1):
            raise ValueError(
            f"Expected next_mag shape to be ({self.batch_size}, 1), "
            f"but got {next_mag.shape}"
            )
        
        if (write_idx >= self.seq_len).any():
            raise RuntimeError("Exceeded max_sample_len; please re-init with larger buffer.")

        self.inter_times[batch_idx, write_idx] = next_inter_times.squeeze(-1)  
        if torch.all(write_idx == 0):   
            last_times= self.t_start
        else:
            assert torch.all(write_idx > 0)
            prev_idx = write_idx - 1
            last_times= self.arrival_times[batch_idx, prev_idx]
        self.arrival_times[batch_idx, write_idx] = (
            last_times + next_inter_times.squeeze(-1)
        )

        if next_mag is not None and "mag" in self:
            self["mag"][batch_idx, write_idx] = next_mag.squeeze(-1)  

        self.input_mask[batch_idx, write_idx] = 1.0
        self.end_idx += 1



    @property
    def batch_size(self):
        return self.arrival_times.shape[0]

    def __len__(self):
        return self.batch_size

    @property
    def seq_len(self):
        return self.arrival_times.shape[1]

    def get_sequence(self, idx: int) -> Sequence:
        length = int(self.end_idx[idx])
        inter_times = self.inter_times[idx, : length + 1].clone()
        t_start = float(self.t_start[idx])
        t_nll_start = float(self.t_nll_start[idx])
        other_attr = {}
        for k in self.keys():
            if k not in self.default_batch_attrs:
                other_attr[k] = self[k][idx, :length].clone()
        return Sequence(
            inter_times=inter_times,
            t_start=t_start,
            t_nll_start=t_nll_start,
            **other_attr,
        )

    def to_list(self) -> List[Sequence]:
        """Convert a batch into a list of variable-length sequences."""
        return [self.get_sequence(idx) for idx in range(self.batch_size)]
    
    def __getitem__(self, key):
        # batch[:, slice]
        if isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None):
            return self._slice_sequences(key[1])
        return super().__getitem__(key)
    
    def _slice_sequences(self, seq_slice: slice) -> "Batch":
        sliced_data = {}
        for k in self.__dict__['_data']: 
            v = self.__dict__['_data'][k]
            if (
                isinstance(v, torch.Tensor)
                and v.ndim >= 2
                and v.shape[1] == self.seq_len
            ):
                sliced_data[k] = v[:, seq_slice, ...]
            else:
                sliced_data[k] = v
        return Batch(**sliced_data)


    


def get_start_idx(
    arrival_times: torch.Tensor,
    t_nll_start: torch.Tensor,
    end_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Get index of the first event that happened after t_nll_start.

    If no event is strictly after t_nll_start in a row, fall back to end_idx so the
    NLL event mask is empty for that sequence.
    """
    x = torch.masked_fill(arrival_times, arrival_times <= t_nll_start[:, None], np.inf)
    start_idx = x.argmin(-1)
    has_candidate = torch.isfinite(x).any(dim=-1)
    if end_idx is None:
        fallback_idx = torch.full_like(start_idx, arrival_times.shape[1] - 1)
    else:
        fallback_idx = end_idx
    return torch.where(has_candidate, start_idx, fallback_idx)


def get_mask(
    inter_times: torch.Tensor,
    start_idx: torch.Tensor,
    end_idx: torch.Tensor,
) -> torch.Tensor:
    """Get a binary mask indicating which arrival times NLL must be computed."""
    arange = torch.arange(inter_times.shape[1], device=inter_times.device)[None, :]
    mask = (start_idx[:, None] <= arange) & (arange < end_idx[:, None])
    return mask.float()


def pad_sequence(
    sequences: List[torch.Tensor],
    padding_value: float = 0,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    """Pad a list of variable length Tensors with `padding_value`."""
    dtype = sequences[0].dtype
    device = sequences[0].device
    max_size = sequences[0].size()
    trailing_dims = max_size[1:] 
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_dims = (len(sequences), max_len) + trailing_dims 

    out_tensor = torch.empty(*out_dims, dtype=dtype, device=device).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        out_tensor[i, :length, ...] = tensor

    return out_tensor

def build_type_seq(non_pad_mask: torch.Tensor, 
                   pad_token_id: int = -100
) -> torch.Tensor:
        type_seq = torch.full_like(non_pad_mask, fill_value=pad_token_id, dtype=torch.long)
        type_seq[non_pad_mask.bool()] = 0
        return type_seq


class EventBatch(DotDict):
    """
    Batch of padded EventSequence instances.

    Attributes:
        arrival_times: Padded arrival times [batch_size, seq_len]
        inter_times: Padded inter-event times [batch_size, seq_len]
        type_seq: Token IDs with padding handled [batch_size, seq_len]
        t_start: Start time of each sequence [batch_size]
        t_end: End time of each sequence [batch_size]
        Other attributes (e.g., mag, loc) are padded similarly.
"""


    @staticmethod
    def from_list(sequences: List["EventSequence"], pad_token_id: int = PAD_TOKEN_ID, pad: int =PAD) -> "EventBatch":
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)

        def collect_and_pad(attr_name, dtype=torch.float32):
            values = [getattr(seq, attr_name) for seq in sequences]
            return pad_sequence(values, padding_value=pad, max_len=max_len).to(dtype)

        arrival_times = collect_and_pad("arrival_times")
        inter_times = collect_and_pad("inter_times")

        # Build non-pad mask
        non_pad_mask = (inter_times != pad).float()
        non_pad_mask[:, 0] = 1.0  # First inter-time is always valid

        # Handle type_seq
        type_seq = None
        if any("type_event" in seq.attributes for seq in sequences):
            types = [seq.attributes["type_event"] for seq in sequences]
            padded_types = pad_sequence(types, padding_value=pad_token_id, max_len=max_len).long()
            type_seq = torch.full_like(padded_types, pad_token_id)
            type_seq[non_pad_mask.bool()] = padded_types[non_pad_mask.bool()]
        else:
            type_seq = build_type_seq(non_pad_mask, pad_token_id=pad_token_id)

        # Handle other attributes like mag, loc (excluding type_event)
        other_attr = {}
        for key in sequences[0].attributes:
            if key == "type_event":
                continue
            values = [seq.attributes[key] for seq in sequences]
            other_attr[key] = pad_sequence(values, padding_value=pad, max_len=max_len)

        t_start = torch.tensor([seq.t_start for seq in sequences], dtype=torch.float32)
        t_end = torch.tensor([seq.t_end for seq in sequences], dtype=torch.float32)
        return EventBatch(
            arrival_times=arrival_times,
            inter_times=inter_times,
            type_seq=type_seq,
            t_start=t_start,
            t_end=t_end,
            **other_attr
        )
    
    def __getitem__(self, key):
        # support batch[:, slice]
        if isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None):
            return self._slice_sequences(key[1])
        return super().__getitem__(key)
    
    def _slice_sequences(self, seq_slice: slice) -> "EventBatch":
        sliced_data = {}
        for k in self.__dict__['_data']:  
            v = self.__dict__['_data'][k]
            if (
                isinstance(v, torch.Tensor)
                and v.ndim >= 2
                and v.shape[1] == self.seq_len
            ):
                sliced_data[k] = v[:, seq_slice, ...]
            else:
                sliced_data[k] = v
        return EventBatch(**sliced_data)

    @property
    def batch_size(self):
        return self.arrival_times.shape[0]

    def __len__(self):
        return self.batch_size

    @property
    def seq_len(self):
        return self.arrival_times.shape[1]