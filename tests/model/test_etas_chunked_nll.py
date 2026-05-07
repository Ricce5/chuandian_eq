from argparse import Namespace

import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.builders import ModelBuilder


def _build_batch() -> Batch:
    torch.manual_seed(7)
    seq1 = Sequence(
        inter_times=torch.rand(64) + 0.1,
        t_start=0.0,
        t_nll_start=0.2,
        mag=torch.rand(63) + 2.0,
    )
    seq2 = Sequence(
        inter_times=torch.rand(71) + 0.1,
        t_start=0.0,
        t_nll_start=0.3,
        mag=torch.rand(70) + 2.0,
    )
    return Batch.from_list([seq1, seq2])


def _build_args(etas_query_chunk_size: int):
    return Namespace(
        model="etas",
        tau_mean=1.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        mag_max=10.0,
        bg_model=None,
        base_rate_init=0.26,
        fix_mu=False,
        fixed_mu_value=None,
        loss_reduction="none",
        etas_query_chunk_size=etas_query_chunk_size,
    )


def test_etas_chunked_nll_matches_full_nll():
    batch = _build_batch()
    full_model = ModelBuilder.by_name("etas")()(_build_args(etas_query_chunk_size=0), torch.device("cpu"))
    chunked_model = ModelBuilder.by_name("etas")()(_build_args(etas_query_chunk_size=13), torch.device("cpu"))
    chunked_model.load_state_dict(full_model.state_dict())
    full_model.eval()
    chunked_model.eval()

    full_nll = full_model.nll_loss(batch, reduction="none")
    chunked_nll = chunked_model.nll_loss(batch, reduction="none")
    torch.testing.assert_close(full_nll, chunked_nll, rtol=1e-5, atol=1e-7)
