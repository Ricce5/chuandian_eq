from argparse import Namespace

import torch

from src.data.batch import Batch
from src.data.sequence import Sequence
from src.models.tpp.common.inter_time_decoding import WeibullMixtureDecoder
from src.models.tpp.common.oracle_blocks import OracleDistDecoder
from src.models.tpp.common.sequence_ops import build_sample_batch
from src.models.tpp.oracle import Oracle
from src.models.tpp.recurrent import RecurrentTPP
from src.models.tpp.recurrent.utils import run_rnn_with_chunking


def _build_sequences() -> list[Sequence]:
    return [
        Sequence(
            inter_times=torch.tensor([1.0, 0.5, 1.0]),
            t_start=0.0,
            t_nll_start=0.5,
            mag=torch.tensor([3.0, 3.5]),
            aRs=torch.tensor([0.1, 0.2]),
            vm=torch.tensor([0.3, 0.4]),
            dVc=torch.tensor([0.5, 0.6]),
            sv=torch.tensor([0.7, 0.8]),
            dTS=torch.tensor([0.9, 1.0]),
        ),
        Sequence(
            inter_times=torch.tensor([0.8, 0.4, 0.6, 0.5]),
            t_start=0.0,
            t_nll_start=0.0,
            mag=torch.tensor([2.5, 2.8, 3.2]),
            aRs=torch.tensor([0.2, 0.3, 0.4]),
            vm=torch.tensor([0.4, 0.5, 0.6]),
            dVc=torch.tensor([0.6, 0.7, 0.8]),
            sv=torch.tensor([0.8, 0.9, 1.0]),
            dTS=torch.tensor([1.0, 1.1, 1.2]),
        ),
    ]


def _recurrent_args() -> Namespace:
    return Namespace(
        d_model=8,
        num_components=2,
        rnn_type="GRU",
        rnn_dropout=0.0,
        tau_mean=1.0,
        mag_mean=3.0,
        time_max=10.0,
        richter_b_mle=1.0,
        mag_completeness=2.0,
        loss_reduction="none",
    )


def _oracle_args() -> Namespace:
    return Namespace(
        loss_reduction="none",
        input_magnitude=True,
        input_injection=True,
        train_to_forecast=False,
        supplementary_mark_list=[],
        oracle_base_mark_order=("time", "aRs", "mag", "Mc", "vm", "dVc", "sv", "dTS"),
        oracle_future_feature_names=("vm", "sv", "dTS", "Mc"),
        num_components=4,
        tau_mean=1.0,
        mag_mean=3.0,
        mag_completeness=2.0,
        oracle_encoder_type="GRU",
        num_rnn_layers=1,
        rnn_dropout=0.0,
        oracle_decoder_type="FCN",
        oracle_lookback_size=1,
        oracle_decoder_hidden_layers=1,
    )


def test_shared_weibull_decoder_smoke():
    decoder = WeibullMixtureDecoder(
        num_components=2,
        parametrization="legacy",
        scale_range="positive",
    )
    context = torch.randn(3, 5, 8)
    hypernet = torch.nn.Linear(8, 6)
    dist = decoder.from_context(context, hypernet)
    sample = dist.sample()
    assert sample.shape == (3, 5)
    log_prob = dist.log_prob(torch.rand(3, 5) + 1e-3)
    assert log_prob.shape == (3, 5)


def test_recurrent_nll_matches_pre_refactor_behavior():
    torch.manual_seed(0)
    batch = Batch.from_list(_build_sequences())
    model = RecurrentTPP(_recurrent_args(), device=torch.device("cpu"))

    context = model.get_context(batch)
    inter_time_dist = model.get_inter_time_dist(context)
    log_like = model.time_log_likelihood(
        batch=batch,
        inter_time_dist=inter_time_dist,
        state=context,
        dist_from_state=model.get_inter_time_dist,
        pdf_inter_times=batch.inter_times,
        survival_inter_times=batch.inter_times,
    )
    expected = -log_like
    got = model.nll_loss(batch, reduction="none", return_dict=False)
    torch.testing.assert_close(got, expected)


def test_oracle_uses_common_decoder_components():
    batch = Batch.from_list(_build_sequences())
    model = Oracle(_oracle_args(), device=torch.device("cpu"))
    assert isinstance(model.dist_decoder, OracleDistDecoder)
    out = model.nll_loss(batch, reduction="none", return_dict=True)
    assert "time" in out and "total" in out
    assert out["time"].shape == torch.Size([batch.batch_size])


def test_run_rnn_with_chunking_matches_full_sequence():
    torch.manual_seed(2)
    rnn = torch.nn.GRU(input_size=3, hidden_size=4, num_layers=2, batch_first=True)
    features = torch.randn(2, 17, 3)
    full_out, full_hidden = rnn(features)
    chunked_out, chunked_hidden = run_rnn_with_chunking(
        rnn=rnn,
        features=features,
        context_size=4,
        chunk_len=5,
    )
    torch.testing.assert_close(chunked_out, full_out)
    torch.testing.assert_close(chunked_hidden, full_hidden)


def test_legacy_tpp_import_paths_remain_available():
    import importlib

    old_to_new = {
        "src.models.tpp.inter_time_decoders": "src.models.tpp.common.inter_time_decoding",
        "src.models.tpp.oracle_components": "src.models.tpp.common.oracle_blocks",
        "src.models.tpp.recurrent_components": "src.models.tpp.common.recurrent_blocks",
        "src.models.tpp.recurrent_v2": "src.models.tpp.recurrent.model_v2",
        "src.models.tpp.recurrent_sampling": "src.models.tpp.recurrent.sampling",
    }

    for old_name, new_name in old_to_new.items():
        old_mod = importlib.import_module(old_name)
        new_mod = importlib.import_module(new_name)
        assert old_mod is new_mod


def test_legacy_updater_factory_aliases_are_callable():
    from src.models.updaters import bayesian_b_updater

    updater = bayesian_b_updater(
        Mc=2.0,
        delta=0.98,
        a0=10.0,
        init_b_target=1.0,
        mag_key="mag",
        write_back=False,
    )
    assert updater.__class__.__name__ == "BayesianGRBUpdater"


def test_build_sample_batch_adds_survival_slot_when_event_slots_are_full():
    inter_times = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    magnitudes = torch.tensor([[2.5, 3.0]], dtype=torch.float32)

    batch = build_sample_batch(
        inter_times=inter_times,
        t_start=0.0,
        t_end=5.0,
        device=torch.device("cpu"),
        magnitudes=magnitudes,
        clamp_last_surv_time=True,
    )

    assert batch.inter_times.shape == (1, 3)
    assert int(batch.end_idx.item()) == 2
    torch.testing.assert_close(batch.inter_times[0], torch.tensor([1.0, 1.0, 3.0]))
    assert batch.mag.shape == (1, 3)

    seq = batch.to_list()[0]
    torch.testing.assert_close(seq.inter_times, torch.tensor([1.0, 1.0, 3.0]))
    torch.testing.assert_close(seq.mag, torch.tensor([2.5, 3.0]))


def test_build_sample_batch_treats_zero_waits_as_padding():
    batch = build_sample_batch(
        inter_times=torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.float32),
        t_start=0.0,
        t_end=5.0,
        device=torch.device("cpu"),
        clamp_last_surv_time=True,
    )

    assert int(batch.end_idx.item()) == 1
    torch.testing.assert_close(batch.inter_times[0, :2], torch.tensor([1.0, 4.0]))

    seq = batch.to_list()[0]
    assert len(seq) == 1
    torch.testing.assert_close(seq.inter_times, torch.tensor([1.0, 4.0]))
    assert not bool((seq.inter_times[:-1] == 0).any().item())
