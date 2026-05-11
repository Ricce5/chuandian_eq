from __future__ import annotations

import torch


def run_rnn_with_chunking(
    *,
    rnn: torch.nn.Module,
    features: torch.Tensor,
    context_size: int,
    chunk_len: int,
    hidden_state=None,
):
    """Run an RNN with optional chunking for very long sequences.

    cuDNN-backed recurrent kernels can fail for very long sequences on some
    environments. This helper splits sequences into contiguous chunks while
    preserving hidden state semantics.
    """
    seq_len = features.size(1)
    if seq_len == 0:
        empty_output = features.new_zeros(features.size(0), 0, context_size)
        return empty_output, hidden_state

    if seq_len <= chunk_len:
        return rnn(features.contiguous(), hidden_state)

    outputs = []
    next_hidden = hidden_state
    for start in range(0, seq_len, chunk_len):
        end = min(start + chunk_len, seq_len)
        out_chunk, next_hidden = rnn(features[:, start:end, :].contiguous(), next_hidden)
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=1), next_hidden


__all__ = ["run_rnn_with_chunking"]

