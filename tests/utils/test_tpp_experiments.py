from __future__ import annotations

import pytest

from src.utils.tpp_experiments import (
    load_tpp_catalog,
    resolve_registered_catalog_class,
    sample_tpp_forecasts,
)


class _CatalogStub:
    init_calls: list[dict] = []

    def __init__(self, root_dir, data_dir=None, freq="1h"):
        call = {"root_dir": root_dir, "data_dir": data_dir, "freq": freq}
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.freq = freq
        type(self).init_calls.append(call)


class _SampleModel:
    def __init__(self):
        self.eval_called = False
        self.calls: list[dict] = []

    def eval(self):
        self.eval_called = True

    def sample(self, batch_size, duration, past_seq=None, return_sequences=False):
        self.calls.append(
            {
                "batch_size": batch_size,
                "duration": duration,
                "past_seq": past_seq,
                "return_sequences": return_sequences,
            }
        )
        return [f"fc-{len(self.calls)}-{i}" for i in range(batch_size)]


class _FlexibleSampleModel:
    def __init__(self):
        self.eval_called = False
        self.calls: list[dict] = []

    def eval(self):
        self.eval_called = True

    def sample(
        self,
        batch_size,
        duration,
        past_seq=None,
        return_sequences=False,
        max_length=None,
        predict_b=None,
        verbose=None,
        random_state=None,
    ):
        self.calls.append(
            {
                "batch_size": batch_size,
                "duration": duration,
                "past_seq": past_seq,
                "return_sequences": return_sequences,
                "max_length": max_length,
                "predict_b": predict_b,
                "verbose": verbose,
                "random_state": random_state,
            }
        )
        return [f"flex-{len(self.calls)}-{i}" for i in range(batch_size)]


def test_resolve_registered_catalog_class_tries_standard_first(monkeypatch):
    def _by_name(name):
        if name == "Demo-Standard":
            return _CatalogStub
        raise RuntimeError(name)

    monkeypatch.setattr("src.data.catalog.Catalog.by_name", _by_name)

    registry_name, cls = resolve_registered_catalog_class("Demo")

    assert registry_name == "Demo-Standard"
    assert cls is _CatalogStub


def test_load_tpp_catalog_builds_catalog_with_filtered_kwargs(tmp_path, monkeypatch):
    _CatalogStub.init_calls.clear()
    (tmp_path / "Demo").mkdir()

    def _by_name(name):
        if name == "Demo-Standard":
            return _CatalogStub
        raise RuntimeError(name)

    monkeypatch.setattr("src.data.catalog.Catalog.by_name", _by_name)

    catalog_obj, registry_name, init_kwargs = load_tpp_catalog(
        "Demo",
        base_dir=tmp_path / "Demo",
        catalog_cfg={"freq": "1D", "unused": "ignored"},
    )

    assert registry_name == "Demo-Standard"
    assert catalog_obj.freq == "1D"
    assert init_kwargs == {
        "root_dir": str((tmp_path / "Demo" / "catalogs").resolve()),
        "data_dir": str((tmp_path / "Demo").resolve()),
        "freq": "1D",
    }
    assert _CatalogStub.init_calls == [init_kwargs]


def test_sample_tpp_forecasts_batches_requests_and_returns_sequences():
    model = _SampleModel()

    forecasts = sample_tpp_forecasts(
        model,
        "past",
        duration=5.0,
        num_samples=5,
        samples_per_batch=2,
    )

    assert model.eval_called is True
    assert [call["batch_size"] for call in model.calls] == [2, 2, 1]
    assert all(call["return_sequences"] is True for call in model.calls)
    assert forecasts == ["fc-1-0", "fc-1-1", "fc-2-0", "fc-2-1", "fc-3-0"]


def test_sample_tpp_forecasts_only_passes_supported_optional_kwargs():
    model = _FlexibleSampleModel()

    forecasts = sample_tpp_forecasts(
        model,
        "past",
        duration=7.0,
        num_samples=3,
        samples_per_batch=3,
        seed=17,
        sample_max_length=99,
        predict_b=False,
        verbose=False,
    )

    assert model.eval_called is True
    assert forecasts == ["flex-1-0", "flex-1-1", "flex-1-2"]
    assert model.calls == [
        {
            "batch_size": 3,
            "duration": 7.0,
            "past_seq": "past",
            "return_sequences": True,
            "max_length": 99,
            "predict_b": False,
            "verbose": False,
            "random_state": 17,
        }
    ]


def test_sample_tpp_forecasts_validates_inputs():
    model = _SampleModel()

    with pytest.raises(ValueError, match="non-negative"):
        sample_tpp_forecasts(
            model,
            "past",
            duration=1.0,
            num_samples=-1,
            samples_per_batch=1,
        )

    with pytest.raises(ValueError, match="positive"):
        sample_tpp_forecasts(
            model,
            "past",
            duration=1.0,
            num_samples=1,
            samples_per_batch=0,
        )
