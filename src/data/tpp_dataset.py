# Reference: https://github.com/ant-research/EasyTemporalPointProcess/blob/main/easy_tpp/preprocess/dataset.py
from pathlib import Path
from typing import List, Union

import torch
import torch.utils.data

from .batch import Batch,EventBatch
from .sequence import Sequence,EventSequence
from functools import partial

class TppDataset(torch.utils.data.Dataset):
    """Dataset represented by a list of event sequences stored in memory."""

    def __init__(self, sequences: List[Union[Sequence, EventSequence]]):
        if any(not isinstance(seq, (Sequence, EventSequence)) for seq in sequences):
            raise ValueError("sequences must be a list of Sequence or EventSequence")
        self.sequences = sequences

    def __getitem__(self, key: int) -> Union[Sequence, EventSequence]:
        return self.sequences[key]

    def __len__(self):
        return len(self.sequences)

    def __repr__(self) -> str: 
        return f"{self.__class__.__name__}({len(self)})"

    def __add__(self, other: "TppDataset") -> "TppDataset":
        return TppDataset(self.sequences + other.sequences)

    @staticmethod
    def load_from_disk(path: Union[str, Path]) -> "TppDataset":
        data = torch.load(path, weights_only=False)

        sequences = []
        for seq_data in data:
            if "arrival_times" in seq_data and "inter_times" in seq_data:
                # It's an EventSequence
                sequences.append(EventSequence(**seq_data))
            elif "inter_times" in seq_data:
                # It's a Sequence
                sequences.append(Sequence(**seq_data))
            else:
                raise ValueError("Unrecognized sequence format in saved file.")
        
        return TppDataset(sequences=sequences)


    def save_to_disk(self, path: Union[str, Path]):
        torch.save([seq.state_dict() for seq in self.sequences], path)

    def apply_(self, function):
        """Apply function to all sequences in the dataset."""
        self.sequences = [function(seq) for seq in self.sequences]
        return self

    def to(self, device):
        """Move all sequences in the dataset to the specified device."""

        def to_device(seq: Union[Sequence, EventSequence]) -> Union[Sequence, EventSequence]:
            return seq.to(device=device)

        return self.apply_(to_device)

    def get_dataloader(self, batch_size=1, shuffle=False, pad_token_id=None, **kwargs):

        first_seq = self.sequences[0]
        if isinstance(first_seq, EventSequence):
            collate_fn = EventBatch.from_list
        elif isinstance(first_seq, Sequence):
            collate_fn = Batch.from_list
        else:
            raise TypeError("Unsupported sequence type in dataset.")
        
        if pad_token_id is not None:
            collate_fn = partial(collate_fn, pad_token_id=pad_token_id)

        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn= collate_fn,
            **kwargs,
        )
