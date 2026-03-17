from argparse import Namespace

import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.tpp.etas import ETAS
from src.models.tpp.recurrent import RecurrentTPP


def _build_sequences() -> list[Sequence]:
    return [
        Sequence(
            inter_times=torch.tensor([1.0, 0.5, 1.0]),
            t_start=0.0,
            t_nll_start=0.5,
            mag=torch.tensor([3.0, 3.5]),
        ),
        Sequence(
            inter_times=torch.tensor([0.8, 0.4, 0.6, 0.5]),
            t_start=0.0,
            t_nll_start=0.0,
            mag=torch.tensor([2.5, 2.8, 3.2]),
        ),
    ]


def test_etas_nll_reduction_modes_match_legacy_behavior():
    batch = Batch.from_list(_build_sequences())
    model = ETAS(device=torch.device("cpu"))

    legacy = model.nll_loss(batch)
    explicit = model.nll_loss(batch, reduction="per_time")
    raw = model.nll_loss(batch, reduction="none")
    out = model.nll_loss(batch, reduction="per_time", return_dict=True)

    span = batch.t_end - batch.t_nll_start

    torch.testing.assert_close(legacy, explicit)
    torch.testing.assert_close(explicit, raw / span)
    torch.testing.assert_close(out["time"], explicit)
    torch.testing.assert_close(out["total"], explicit)


def test_recurrent_nll_reduction_modes_match_legacy_behavior():
    torch.manual_seed(0)
    batch = Batch.from_list(_build_sequences())
    args = Namespace(
        d_model=8,
        num_components=2,
        rnn_type="GRU",
        rnn_dropout=0.0,
        tau_mean=1.0,
        mag_mean=3.0,
        time_max=10.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        loss_reduction="per_time",
    )
    model = RecurrentTPP(args, device=torch.device("cpu"))

    legacy = model.nll_loss(batch)
    explicit = model.nll_loss(batch, reduction="per_time")
    raw = model.nll_loss(batch, reduction="none")
    out = model.nll_loss(batch, reduction="per_time", return_dict=True)

    span = batch.t_end - batch.t_nll_start

    torch.testing.assert_close(legacy, explicit)
    torch.testing.assert_close(explicit, raw / span)
    torch.testing.assert_close(out["time"], explicit)
    torch.testing.assert_close(out["total"], explicit)
