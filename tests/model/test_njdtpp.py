from argparse import Namespace
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.builders import ModelBuilder
from src.models.tpp.njdtpp import NJDTPP
from src.train.model_routing import get_model_family


def _build_args(**overrides) -> Namespace:
    defaults = dict(
        model="njdtpp",
        task_type="tpp",
        loss_reduction="per_time",
        njdtpp_num_divide=4,
        njdtpp_dim_eta=6,
        njdtpp_dim_hidden=16,
        njdtpp_num_hidden=1,
        njdtpp_sigma=0.01,
        njdtpp_euler_eps=1e-8,
        njdtpp_log_intensity_clip=20.0,
        njdtpp_mark_b_floor=1e-3,
        njdtpp_sample_step=0.2,
        njdtpp_sample_max_events=64,
        njdtpp_sample_intensity_floor=1e-6,
        njdtpp_sample_init_noise_scale=0.0,
        mag_completeness=2.0,
        mag_max=6.0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _build_sequences() -> list[Sequence]:
    return [
        Sequence(
            inter_times=torch.tensor([0.5, 1.0, 0.8], dtype=torch.float64),
            t_start=0.0,
            t_nll_start=0.6,
            mag=torch.tensor([2.4, 3.1], dtype=torch.float64),
        ),
        Sequence(
            inter_times=torch.tensor([0.4, 0.6], dtype=torch.float64),
            t_start=0.0,
            t_nll_start=0.4,
            mag=torch.tensor([4.5], dtype=torch.float64),
        ),
    ]


def test_njdtpp_builder_and_routing_work():
    args = _build_args()
    model = ModelBuilder.by_name("njdtpp")()(args, torch.device("cpu"))

    assert isinstance(model, NJDTPP)
    assert get_model_family("njdtpp") == "tpp"


def test_njdtpp_nll_outputs_are_finite_and_consistent():
    torch.manual_seed(0)
    args = _build_args(loss_reduction="none")
    model = NJDTPP(args, device=torch.device("cpu"))
    batch = Batch.from_list(_build_sequences())

    torch.manual_seed(123)
    raw = model.nll_loss(batch, reduction="none", return_dict=True)
    torch.manual_seed(123)
    reduced = model.nll_loss(batch, reduction="per_time", return_dict=True)
    torch.manual_seed(123)
    legacy = model.nll_loss(batch)

    assert set(raw.keys()) == {"time", "mark", "total"}
    assert all(torch.isfinite(value).all() for value in raw.values())
    torch.testing.assert_close(raw["time"] + raw["mark"], raw["total"])

    span = (batch.t_end - batch.t_nll_start).to(raw["time"].dtype)
    torch.testing.assert_close(reduced["time"], raw["time"] / span)
    torch.testing.assert_close(reduced["mark"], raw["mark"] / span)
    torch.testing.assert_close(reduced["total"], raw["total"] / span)
    torch.testing.assert_close(legacy, raw["total"])


def test_njdtpp_sample_returns_valid_sequences_with_and_without_history():
    torch.manual_seed(0)
    model = NJDTPP(_build_args(), device=torch.device("cpu"))

    samples = model.sample(batch_size=3, duration=1.0, t_start=0.5, return_sequences=True)

    assert len(samples) == 3
    for seq in samples:
        assert isinstance(seq, Sequence)
        assert seq.t_start == 0.5
        assert seq.t_end >= seq.t_start
        assert seq.inter_times.ndim == 1
        assert seq.inter_times.shape[0] == seq.num_events + 1
        assert "mag" in seq
        assert seq.mag.shape[0] == seq.num_events

    past_seq = Sequence(
        inter_times=torch.tensor([0.3, 0.4, 0.8], dtype=torch.float32),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5, 2.8], dtype=torch.float32),
    )
    conditioned = model.sample(batch_size=1, duration=0.7, past_seq=past_seq, return_sequences=True)

    assert len(conditioned) == 1
    assert conditioned[0].t_start == past_seq.t_end


def test_njdtpp_evaluate_functions_return_sorted_finite_outputs():
    torch.manual_seed(0)
    model = NJDTPP(_build_args(), device=torch.device("cpu"))
    sequence = Sequence(
        inter_times=torch.tensor([0.3, 0.4, 0.8], dtype=torch.float64),
        t_start=0.0,
        t_nll_start=0.0,
        mag=torch.tensor([2.5, 2.8], dtype=torch.float64),
    )

    grid_i, intensity = model.evaluate_intensity(sequence, num_grid_points=8)
    grid_c, compensator = model.evaluate_compensator(sequence, num_grid_points=8)

    assert grid_i.shape == intensity.shape
    assert grid_c.shape == compensator.shape
    assert torch.isfinite(intensity).all()
    assert torch.isfinite(compensator).all()
    assert torch.all(intensity >= 0)
    assert torch.all(grid_i[1:] >= grid_i[:-1])
    assert torch.all(grid_c[1:] >= grid_c[:-1])
    assert torch.all(compensator[1:] >= compensator[:-1])


def test_njdtpp_config_file_exists_with_expected_model_name():
    config_path = Path("config/njdtpp.yaml")
    text = config_path.read_text(encoding="utf-8")

    assert config_path.exists()
    assert "model: njdtpp" in text
    assert "task_type: tpp" in text
