import json

import pandas as pd
import torch

from src.catalogs.induced_triplet_base import InducedTripletBase


def _write_triplet_dataset(
    root_dir,
    *,
    dataset_name: str = "MiniField",
    eq_offsets_hours=(0.5, 2.0, 4.5, 5.5),
    inj_offsets_hours=(0, 1, 2, 3, 4, 5, 6),
    inj_rates=(10.0, 20.0, 40.0, 30.0, 15.0, 25.0, 35.0),
):
    dataset_dir = root_dir / dataset_name
    processed_dir = dataset_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp("2020-01-01T00:00:00")
    end_ts = pd.Timestamp("2020-01-01T06:00:00")

    eq_df = pd.DataFrame(
        {
            "time_iso": [start_ts + pd.Timedelta(hours=float(offset)) for offset in eq_offsets_hours],
            "magnitude": [2.1, 2.4, 2.6, 2.8][: len(eq_offsets_hours)],
        }
    )
    eq_df.to_csv(processed_dir / f"{dataset_name}_eq_processed.csv", index=False)

    inj_df = pd.DataFrame(
        {
            "time_iso": [start_ts + pd.Timedelta(hours=float(offset)) for offset in inj_offsets_hours],
            "inj_rate_m3_min": list(inj_rates),
        }
    )
    inj_df.to_csv(
        processed_dir / f"{dataset_name}_inj_60min_processed.csv",
        index=False,
    )

    summary = {
        "resample_freq_min": 60,
        "mc": 2.0,
        "inj_fill_policy": "ffill",
        "is_upsample": False,
        "start_time_iso": start_ts.isoformat(),
        "end_time_iso": end_ts.isoformat(),
    }
    with open(processed_dir / f"{dataset_name}_summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj)

    return dataset_dir


def test_induced_triplet_base_normalizes_injection_time_series(tmp_path):
    dataset_dir = _write_triplet_dataset(tmp_path)

    catalog_ds = InducedTripletBase(
        dataset_name="MiniField",
        root_dir=dataset_dir / "catalogs",
        data_dir=dataset_dir,
        normalize=True,
    )

    seq = catalog_ds.full_sequence
    expected = torch.tensor(
        [0.0, 1.0 / 3.0, 1.0, 2.0 / 3.0, 1.0 / 6.0, 0.5, 5.0 / 6.0],
        dtype=torch.float32,
    )

    assert "inj_rate" in catalog_ds.norm_stats
    assert torch.allclose(seq.time_series.squeeze(-1), expected, atol=1e-6)
    assert torch.allclose(
        seq.time_series_times,
        torch.arange(7, dtype=torch.float32),
    )


def test_induced_triplet_base_keeps_raw_injection_time_series_when_disabled(tmp_path):
    dataset_dir = _write_triplet_dataset(tmp_path)

    catalog_ds = InducedTripletBase(
        dataset_name="MiniField",
        root_dir=dataset_dir / "catalogs",
        data_dir=dataset_dir,
        normalize=False,
    )

    seq = catalog_ds.full_sequence
    expected = torch.tensor([10.0, 20.0, 40.0, 30.0, 15.0, 25.0, 35.0], dtype=torch.float32)

    assert "inj_rate" not in catalog_ds.norm_stats
    assert torch.allclose(seq.time_series.squeeze(-1), expected, atol=1e-6)


def test_induced_triplet_base_constant_injection_series_normalizes_safely(tmp_path):
    dataset_dir = _write_triplet_dataset(
        tmp_path,
        dataset_name="FlatField",
        inj_rates=(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0),
    )

    catalog_ds = InducedTripletBase(
        dataset_name="FlatField",
        root_dir=dataset_dir / "catalogs",
        data_dir=dataset_dir,
        normalize=False,
        normalize_time_series=True,
    )

    seq = catalog_ds.full_sequence

    assert "inj_rate" in catalog_ds.norm_stats
    assert torch.isfinite(seq.time_series).all()
    assert torch.allclose(seq.time_series, torch.zeros_like(seq.time_series))


def test_induced_triplet_base_resamples_injection_time_series_to_requested_freq(tmp_path):
    dataset_dir = _write_triplet_dataset(tmp_path)

    catalog_ds = InducedTripletBase(
        dataset_name="MiniField",
        root_dir=dataset_dir / "catalogs",
        data_dir=dataset_dir,
        normalize=False,
        resample_freq_min=30,
    )

    seq = catalog_ds.full_sequence
    expected_times = torch.arange(0.0, 6.0 + 0.5, 0.5, dtype=torch.float32)
    expected_rates = torch.tensor(
        [10.0, 10.0, 20.0, 20.0, 40.0, 40.0, 30.0, 30.0, 15.0, 15.0, 25.0, 25.0, 35.0],
        dtype=torch.float32,
    )

    assert catalog_ds.metadata["freq_min"] == 30
    assert catalog_ds.metadata["source_freq_min"] == 60
    assert catalog_ds.metadata["inj_source_freq_min"] == 60
    assert catalog_ds.metadata["inj_resampled"] is True
    assert torch.allclose(seq.time_series_times, expected_times, atol=1e-6)
    assert torch.allclose(seq.time_series.squeeze(-1), expected_rates, atol=1e-6)
