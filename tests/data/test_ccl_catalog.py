from pathlib import Path

import torch

from src.data.catalog import Catalog
from src.utils.tpp_experiments import load_tpp_catalog


def test_ccl_catalog_registered():
    assert "CCL-Standard" in Catalog.list_available()


def test_ccl_catalog_loads_from_processed_data(tmp_path):
    dataset_dir = Path("data") / "CCL"
    catalog_ds, registry_name, init_kwargs = load_tpp_catalog(
        "CCL",
        base_dir=dataset_dir,
        catalog_cfg={
            "root_dir": tmp_path / "catalogs",
            "normalize": True,
        },
        candidates=["CCL-Standard", "CCL"],
    )

    assert registry_name == "CCL-Standard"
    assert Path(init_kwargs["data_dir"]).resolve() == dataset_dir.resolve()
    assert catalog_ds.root_dir.exists()
    assert (catalog_ds.root_dir / "full_sequence.pt").exists()
    assert len(catalog_ds.full_sequence) > 0
    assert len(catalog_ds.train) == 1
    assert len(catalog_ds.val) == 1
    assert len(catalog_ds.test) == 1
    assert catalog_ds.full_sequence.time_series.shape[0] > 0


def test_ccl_catalog_supports_runtime_resample_freq_min(tmp_path):
    dataset_dir = Path("data") / "CCL"
    catalog_ds, registry_name, init_kwargs = load_tpp_catalog(
        "CCL",
        base_dir=dataset_dir,
        catalog_cfg={
            "root_dir": tmp_path / "catalogs",
            "normalize": True,
            "resample_freq_min": 5,
        },
        candidates=["CCL-Standard", "CCL"],
    )

    assert registry_name == "CCL-Standard"
    assert init_kwargs["resample_freq_min"] == 5
    assert catalog_ds.metadata["freq_min"] == 5
    assert catalog_ds.metadata["inj_source_freq_min"] == 1
    assert catalog_ds.metadata["inj_resampled"] is True

    time_series_times = catalog_ds.full_sequence.time_series_times
    assert time_series_times.shape[0] > 2
    first_diffs = torch.diff(time_series_times[:5])
    assert torch.allclose(first_diffs, torch.full_like(first_diffs, 5.0 / 60.0), atol=1e-6)
