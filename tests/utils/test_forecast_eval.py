from __future__ import annotations

from argparse import Namespace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.utils.forecast_eval import (
    SlidingWindowEvaluationRange,
    SlidingWindowForecastConfig,
    _build_sliding_window_bounds,
    _build_sliding_window_starts,
    _limit_range_to_background_cache,
    build_default_sliding_cache_filename,
    build_sliding_cache_metadata,
    evaluate_sliding_window_forecast_plots,
    load_sliding_window_cache_if_compatible,
    resolve_sliding_window_evaluation_range,
    run_sliding_window_forecast,
    save_sliding_window_cache,
)
from src.utils.forecast_eval_helpers import (
    run_sliding_window_forecast as run_legacy_sliding_window_forecast,
)

_CLI_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_sliding_window_forecast.py"
)
_CLI_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "run_sliding_window_forecast_test_module",
    _CLI_SCRIPT_PATH,
)
assert _CLI_SCRIPT_SPEC is not None
assert _CLI_SCRIPT_SPEC.loader is not None
_CLI_SCRIPT_MODULE = importlib.util.module_from_spec(_CLI_SCRIPT_SPEC)
_CLI_SCRIPT_SPEC.loader.exec_module(_CLI_SCRIPT_MODULE)
build_cli_default_cache_filename = _CLI_SCRIPT_MODULE.build_default_cache_filename
resolve_evaluation_range = _CLI_SCRIPT_MODULE.resolve_evaluation_range


class _SampleModel:
    def __init__(self):
        self.sample_durations: list[float] = []
        self.past_sequence_ends: list[float] = []

    def eval(self):
        return self

    def sample(self, *, batch_size, duration, past_seq, return_sequences):
        self.sample_durations.append(float(duration))
        self.past_sequence_ends.append(float(past_seq.t_end))
        assert return_sequences is True
        return [[0] for _ in range(batch_size)]


class _ForecastWithMagnitude:
    def __init__(self):
        self.mag = torch.tensor([1.0], dtype=torch.float64)

    def __len__(self):
        return 1


class _MagnitudeSampleModel(_SampleModel):
    def sample(self, *, batch_size, duration, past_seq, return_sequences):
        self.sample_durations.append(float(duration))
        assert return_sequences is True
        return [_ForecastWithMagnitude() for _ in range(batch_size)]


class _Subsequence:
    def __init__(self, count: int, *, t_end: float):
        self.count = count
        self.t_end = float(t_end)

    def __len__(self):
        return self.count

    def to(self, device):
        return self


class _SequenceStub:
    def __init__(
        self,
        arrival_times,
        *,
        t_start=0.0,
        t_end=None,
        subsequence_t_end_offset=0.0,
    ):
        self.arrival_times = torch.as_tensor(arrival_times, dtype=torch.float64)
        self.t_start = float(t_start)
        self.t_end = float(
            self.arrival_times[-1].item() if t_end is None else t_end
        )
        self.subsequence_t_end_offset = float(subsequence_t_end_offset)
        self.subsequence_bounds: list[tuple[float, float]] = []

    def get_subsequence(self, start, end, reset_t_nll_to_end=False):
        self.subsequence_bounds.append((float(start), float(end)))
        mask = (self.arrival_times >= start) & (self.arrival_times <= end)
        return _Subsequence(
            int(mask.sum().item()),
            t_end=float(end) + self.subsequence_t_end_offset,
        )


def _sequence_with_last_arrival_at_95() -> _SequenceStub:
    return _SequenceStub(
        np.r_[np.arange(1.0, 92.0, 10.0), 95.0],
        t_end=100.0,
    )


def test_sliding_window_starts_can_include_tail_grid_points():
    without_tail = _build_sliding_window_starts(
        0.0,
        91.0,
        10.0,
        10.0,
    )
    with_tail = _build_sliding_window_starts(
        0.0,
        91.0,
        10.0,
        10.0,
        include_truncated_final_window=True,
    )

    np.testing.assert_allclose(without_tail, np.arange(10.0, 81.0, 10.0))
    np.testing.assert_allclose(with_tail, np.arange(10.0, 91.0, 10.0))

    np.testing.assert_allclose(
        _build_sliding_window_starts(
            0.0,
            90.0,
            10.0,
            10.0,
            include_truncated_final_window=True,
        ),
        np.arange(10.0, 90.0, 10.0),
    )


def test_truncated_final_window_uses_shorter_sampling_duration():
    seq = _sequence_with_last_arrival_at_95()
    model = _SampleModel()

    result = run_sliding_window_forecast(
        model,
        seq,
        torch.device("cpu"),
        config=SlidingWindowForecastConfig(
            duration=10.0,
            slide_step=10.0,
            include_truncated_final_window=True,
            samples_per_batch=2,
        ),
    )

    np.testing.assert_allclose(result.t_forecast, np.arange(11.0, 95.0, 10.0))
    np.testing.assert_allclose(
        result.t_window_end,
        np.r_[np.arange(21.0, 92.0, 10.0), 95.0],
    )
    assert model.sample_durations[:-1] == [10.0] * 8
    assert model.sample_durations[-1] == 4.0
    np.testing.assert_allclose(model.past_sequence_ends, result.t_forecast)
    assert seq.subsequence_bounds[-1] == (91.0, 95.0)


def test_past_sequence_end_is_aligned_to_forecast_grid():
    seq = _SequenceStub(
        np.r_[np.arange(1.0, 92.0, 10.0), 95.0],
        t_end=100.0,
        subsequence_t_end_offset=1e-4,
    )
    model = _SampleModel()

    result = run_sliding_window_forecast(
        model,
        seq,
        torch.device("cpu"),
        config=SlidingWindowForecastConfig(
            duration=10.0,
            slide_step=10.0,
            samples_per_batch=2,
        ),
    )

    np.testing.assert_allclose(model.past_sequence_ends, result.t_forecast)


def test_sliding_cache_separates_truncated_window_policy(tmp_path):
    seq = _sequence_with_last_arrival_at_95()
    common = {
        "duration": 10.0,
        "slide_step": 10.0,
        "quantiles": (2.5, 97.5),
        "samples_per_batch": 2,
    }
    metadata_without_tail = build_sliding_cache_metadata(
        seq,
        **common,
        include_truncated_final_window=False,
    )
    metadata_with_tail = build_sliding_cache_metadata(
        seq,
        **common,
        include_truncated_final_window=True,
    )
    cache_path = tmp_path / "sliding.npz"
    save_sliding_window_cache(
        cache_path,
        metadata=metadata_with_tail,
        t_forecast_list=np.array([11.0, 21.0]),
        counts_list=np.array([1, 2]),
        q_list=np.array([[0.0, 2.0], [0.0, 3.0]]),
        mean_list=np.array([1.0, 2.0]),
        sim_count_matrix=np.array([[1, 1], [2, 2]], dtype=np.int32),
    )

    assert load_sliding_window_cache_if_compatible(
        cache_path,
        metadata=metadata_with_tail,
    ) is not None
    assert load_sliding_window_cache_if_compatible(
        cache_path,
        metadata=metadata_without_tail,
    ) is None


def test_evaluation_range_limits_targets_but_retains_full_history():
    seq = _sequence_with_last_arrival_at_95()
    model = _SampleModel()
    evaluation_range = SlidingWindowEvaluationRange(
        name="val",
        start=50.0,
        end=70.0,
    )

    result = run_sliding_window_forecast(
        model,
        seq,
        torch.device("cpu"),
        config=SlidingWindowForecastConfig(
            duration=10.0,
            slide_step=10.0,
            samples_per_batch=2,
            evaluation_range=evaluation_range,
        ),
    )

    np.testing.assert_allclose(result.t_forecast, [50.0, 60.0])
    np.testing.assert_allclose(result.t_window_end, [60.0, 70.0])
    assert model.sample_durations == [10.0, 10.0]
    assert seq.subsequence_bounds == [
        (0.0, 50.0),
        (50.0, 60.0),
        (0.0, 60.0),
        (60.0, 70.0),
    ]


def test_range_tail_can_be_truncated_at_explicit_end():
    seq = _sequence_with_last_arrival_at_95()
    evaluation_range = SlidingWindowEvaluationRange(
        name="custom",
        start=50.0,
        end=67.0,
    )

    full_starts, full_ends = _build_sliding_window_bounds(
        seq,
        duration=10.0,
        slide_step=10.0,
        evaluation_range=evaluation_range,
    )
    truncated_starts, truncated_ends = _build_sliding_window_bounds(
        seq,
        duration=10.0,
        slide_step=10.0,
        evaluation_range=evaluation_range,
        include_truncated_final_window=True,
    )

    np.testing.assert_allclose(full_starts, [50.0])
    np.testing.assert_allclose(full_ends, [60.0])
    np.testing.assert_allclose(truncated_starts, [50.0, 60.0])
    np.testing.assert_allclose(truncated_ends, [60.0, 67.0])


def test_range_metadata_and_default_cache_filename_do_not_collide(tmp_path):
    seq = _sequence_with_last_arrival_at_95()
    common = {
        "duration": 10.0,
        "slide_step": 10.0,
        "quantiles": (2.5, 97.5),
        "samples_per_batch": 2,
        "sampling_seed": 7,
    }
    val_metadata = build_sliding_cache_metadata(
        seq,
        **common,
        evaluation_range=SlidingWindowEvaluationRange(
            name="val",
            start=50.0,
            end=70.0,
        ),
    )
    test_metadata = build_sliding_cache_metadata(
        seq,
        **common,
        evaluation_range=SlidingWindowEvaluationRange(
            name="test",
            start=70.0,
            end=100.0,
        ),
    )
    cache_path = tmp_path / build_default_sliding_cache_filename(val_metadata)
    save_sliding_window_cache(
        cache_path,
        metadata=val_metadata,
        t_forecast_list=np.array([50.0, 60.0]),
        counts_list=np.array([1, 2]),
        q_list=np.array([[0.0, 2.0], [0.0, 3.0]]),
        mean_list=np.array([1.0, 2.0]),
        sim_count_matrix=np.array([[1, 1], [2, 2]], dtype=np.int32),
    )

    assert build_default_sliding_cache_filename(
        val_metadata
    ) != build_default_sliding_cache_filename(test_metadata)
    assert load_sliding_window_cache_if_compatible(
        cache_path,
        metadata=val_metadata,
    ) is not None
    assert load_sliding_window_cache_if_compatible(
        cache_path,
        metadata=test_metadata,
    ) is None

    resolved_test_range = resolve_sliding_window_evaluation_range(
        seq,
        SlidingWindowEvaluationRange(
            name="test",
            start=70.0,
            end=100.0,
        ),
    )
    assert resolved_test_range.end == 100.0


def test_range_bounds_tolerate_timestamp_conversion_rounding():
    seq = _sequence_with_last_arrival_at_95()

    resolved = resolve_sliding_window_evaluation_range(
        seq,
        SlidingWindowEvaluationRange(
            name="test",
            start=-1e-11,
            end=100.0 + 1e-11,
        ),
    )

    assert resolved.start == 0.0
    assert resolved.end == 100.0
    assert resolved.grid_anchor == 0.0


def test_background_cache_endpoint_float_rounding_is_clamped():
    cached_times = torch.tensor(
        [46.74661145, 46.74761145, 46.74861145],
        dtype=torch.float32,
    )
    model = SimpleNamespace(
        bg_model=SimpleNamespace(
            ts_batch_cache=SimpleNamespace(time_series_times=cached_times),
        )
    )
    cache_end = float(cached_times[-1])
    evaluation_range = SlidingWindowEvaluationRange(
        name="test",
        start=float(cached_times[0]),
        end=cache_end + 3.8e-6,
        grid_anchor=float(cached_times[0]),
    )

    with pytest.warns(RuntimeWarning, match="Clamped sliding evaluation range"):
        limited = _limit_range_to_background_cache(model, evaluation_range)

    assert limited.start == float(cached_times[0])
    assert limited.end == cache_end
    assert limited.grid_anchor == float(cached_times[0])


def test_background_cache_endpoint_within_one_grid_step_is_clamped():
    cached_times = torch.tensor(
        [1.998, 1.999, 2.0],
        dtype=torch.float32,
    )
    model = SimpleNamespace(
        bg_model=SimpleNamespace(
            ts_batch_cache=SimpleNamespace(time_series_times=cached_times),
        )
    )
    evaluation_range = SlidingWindowEvaluationRange(
        name="test",
        start=float(cached_times[0]),
        end=float(cached_times[-1]) + 5e-4,
        grid_anchor=float(cached_times[0]),
    )

    with pytest.warns(RuntimeWarning, match="Clamped sliding evaluation range"):
        limited = _limit_range_to_background_cache(model, evaluation_range)

    assert limited.end == float(cached_times[-1])


def test_background_cache_endpoint_material_mismatch_still_raises():
    cached_times = torch.tensor(
        [46.74661145, 46.74761145, 46.74861145],
        dtype=torch.float32,
    )
    model = SimpleNamespace(
        bg_model=SimpleNamespace(
            ts_batch_cache=SimpleNamespace(time_series_times=cached_times),
        )
    )
    evaluation_range = SlidingWindowEvaluationRange(
        name="test",
        start=float(cached_times[0]),
        end=float(cached_times[-1]) + 1e-2,
        grid_anchor=float(cached_times[0]),
    )

    with pytest.raises(
        ValueError,
        match="ends after cached background range",
    ):
        _limit_range_to_background_cache(model, evaluation_range)


def test_legacy_helper_uses_shared_range_implementation():
    seq = _sequence_with_last_arrival_at_95()
    model = _SampleModel()

    t_forecast, counts, quantiles, mean = run_legacy_sliding_window_forecast(
        model,
        seq,
        torch.device("cpu"),
        duration=10.0,
        slide_step=10.0,
        samples_per_batch=2,
        evaluation_range=SlidingWindowEvaluationRange(
            name="val",
            start=50.0,
            end=67.0,
        ),
        include_truncated_final_window=True,
    )

    np.testing.assert_allclose(t_forecast, [50.0, 60.0])
    np.testing.assert_array_equal(counts, [1, 1])
    np.testing.assert_allclose(quantiles, [[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(mean, [1.0, 1.0])


def test_evaluator_cache_preserves_truncated_window_endpoints(tmp_path):
    seq = _sequence_with_last_arrival_at_95()
    seq.mag = torch.linspace(1.0, 2.0, len(seq.arrival_times))
    model = _MagnitudeSampleModel()
    catalog_ds = type(
        "_CatalogStub",
        (),
        {
            "metadata": {
                "start_ts": "2020-01-01T00:00:00",
                "freq": "1D",
            }
        },
    )()
    evaluation_range = SlidingWindowEvaluationRange(
        name="custom",
        start=50.0,
        end=67.0,
    )
    kwargs = {
        "model": model,
        "seq": seq,
        "device": torch.device("cpu"),
        "catalog_ds": catalog_ds,
        "checkpoint_dir": tmp_path,
        "sliding_duration": 10.0,
        "sliding_step": 10.0,
        "sliding_quantiles": (2.5, 97.5),
        "samples_per_batch": 2,
        "include_truncated_final_window": True,
        "evaluation_range": evaluation_range,
        "load_sliding_cache": True,
        "sampling_seed": 7,
        "save_plots": False,
    }

    computed = evaluate_sliding_window_forecast_plots(
        **kwargs,
        force_recompute_sliding=True,
    )
    loaded = evaluate_sliding_window_forecast_plots(
        **kwargs,
        force_recompute_sliding=False,
    )

    assert computed["sliding_loaded_from_cache"] is False
    assert loaded["sliding_loaded_from_cache"] is True
    np.testing.assert_allclose(computed["t_forecast_list"], [50.0, 60.0])
    np.testing.assert_allclose(computed["t_window_end_list"], [60.0, 67.0])
    np.testing.assert_allclose(loaded["t_window_end_list"], [60.0, 67.0])
    assert model.sample_durations == [10.0, 7.0]


def test_cli_resolves_validation_and_test_ranges_and_uses_isolated_names():
    seq = _SequenceStub([1.0, 11.0, 21.0, 31.0], t_end=40.0)
    catalog_ds = type(
        "_CatalogStub",
        (),
        {
            "metadata": {
                "start_ts": "2020-01-01T00:00:00",
                "freq": "1D",
                "val_start_ts": "2020-01-11T00:00:00",
                "test_start_ts": "2020-01-21T00:00:00",
            }
        },
    )()

    val_range = resolve_evaluation_range(
        Namespace(eval_range="val", eval_start=None, eval_end=None),
        catalog_ds=catalog_ds,
        seq=seq,
        catalog_cfg={},
    )
    test_range = resolve_evaluation_range(
        Namespace(eval_range="test", eval_start=None, eval_end=None),
        catalog_ds=catalog_ds,
        seq=seq,
        catalog_cfg={},
    )

    assert val_range == SlidingWindowEvaluationRange(
        name="val",
        start=10.0,
        end=20.0,
    )
    assert test_range == SlidingWindowEvaluationRange(
        name="test",
        start=20.0,
        end=40.0,
    )

    common = {
        "duration": 10.0,
        "slide_step": 10.0,
        "quantiles": (2.5, 97.5),
        "samples_per_batch": 2,
        "sampling_seed": 7,
    }
    val_metadata = build_sliding_cache_metadata(
        seq,
        **common,
        evaluation_range=val_range,
    )
    test_metadata = build_sliding_cache_metadata(
        seq,
        **common,
        evaluation_range=test_range,
    )
    filename_args = Namespace(
        forecast_b_sampling="model",
        updater_name="bayesian_gr",
    )

    assert build_cli_default_cache_filename(
        filename_args,
        val_metadata,
    ) != build_cli_default_cache_filename(
        filename_args,
        test_metadata,
    )
