import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.tpp.etas import ETAS
from src.models.bg.base import BGModel


def test_nll_loss_handles_batch_size_gt_one():
    """nll_loss should return one value per sequence and match per-item calls."""

    # two simple sequences with different lengths and t_nll_start
    seq1 = Sequence(
        inter_times=torch.tensor([1.0, 0.5, 1.0]),
        t_start=0.0,
        t_nll_start=0.5,
        mag=torch.tensor([3.0, 3.5]),
    )
    seq2 = Sequence(
        inter_times=torch.tensor([0.8, 0.4, 0.6, 0.5]),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5, 2.8, 3.2]),
    )

    batch = Batch.from_list([seq1, seq2])
    model = ETAS(device=torch.device("cpu"))

    nll_batch = model.nll_loss(batch)
    nll_seq1 = model.nll_loss(Batch.from_list([seq1]))
    nll_seq2 = model.nll_loss(Batch.from_list([seq2]))

    assert nll_batch.shape == (2,)
    torch.testing.assert_close(nll_batch[0], nll_seq1[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(nll_batch[1], nll_seq2[0], rtol=1e-4, atol=1e-4)


class _DummyBGModel(BGModel):
    """Simple background model with constant intensity λ=scale_init."""

    def __init__(self, device: torch.device):
        super().__init__(device=device, scale_init=1.0)

    def scaled_intensity(self, time_series: torch.Tensor) -> torch.Tensor:  # (B, T, 1)
        return torch.ones_like(time_series[..., :1])


def test_nll_loss_with_background_constant_intensity():
    """With ETAS intensity suppressed, NLL should reduce to constant background λ=1."""

    device = torch.device("cpu")

    # two sequences, different lengths; provide time_series so bg_model works
    seq1 = Sequence(
        inter_times=torch.tensor([1.0, 1.0, 1.0]),  # two events + survival
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([3.0, 3.0]),
        time_series=torch.zeros(4, 1),
        time_series_times=torch.tensor([0.0, 1.0, 2.0, 3.0]),
    )
    seq2 = Sequence(
        inter_times=torch.tensor([0.5, 0.5, 0.5, 0.5]),  # three events + survival
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5, 2.5, 2.5]),
        time_series=torch.zeros(5, 1),
        time_series_times=torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0]),
    )

    batch = Batch.from_list([seq1, seq2])

    # ETAS with zero background (mu=0, k≈0) so only bg_model contributes
    base_state = ETAS(device=device, fix_mu=True).state_dict()
    bg_model = _DummyBGModel(device=device)
    model = ETAS(device=device, bg_model=bg_model, fix_mu=True)
    model.load_state_dict(base_state, strict=False)
    model.set_params(k=1e-12)  # suppress triggering component

    nll = model.nll_loss(batch)

    # With λ_bg=1, NLL for each seq = integral/duration = (t_end - t_start)/(t_end - t_start) = 1
    torch.testing.assert_close(nll, torch.ones_like(nll), rtol=1e-4, atol=1e-4)
