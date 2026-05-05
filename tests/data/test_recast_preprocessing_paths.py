from pathlib import Path

import pandas as pd
import pytest
import torch

from src.data.preprocessing import (
    _resolve_recast_csv_path,
    _resolve_recast_metadata_path,
    process_recast_catalog,
)


def _write_catalog_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "time": [
                "1981-01-01 00:10:00",
                "1981-01-01 00:20:00",
            ],
            "magnitude": [2.0, 2.5],
            "longitude": [-117.0, -117.1],
            "latitude": [34.0, 34.1],
            "depth": [5.0, 6.0],
        }
    )
    df.to_csv(path, index=False)


def test_resolve_recast_csv_path_prefers_non_catalog_case_insensitive(tmp_path):
    raw_dir = tmp_path / "SCEDC" / "raw"
    raw_dir.mkdir(parents=True)
    _write_catalog_csv(raw_dir / "scedc.csv")
    _write_catalog_csv(raw_dir / "catalog.csv")

    resolved = _resolve_recast_csv_path(raw_dir / "SCEDC.csv")

    assert resolved == raw_dir / "scedc.csv"


def test_resolve_recast_metadata_path_uses_hashed_catalog_metadata(tmp_path):
    raw_dir = tmp_path / "SCEDC" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "scedc.csv"
    _write_catalog_csv(csv_path)

    metadata_path = tmp_path / "SCEDC" / "catalogs" / "abc12345" / "metadata.pt"
    metadata_path.parent.mkdir(parents=True)
    torch.save({"start_ts": pd.Timestamp("1981-01-01 00:00:00")}, metadata_path)

    resolved = _resolve_recast_metadata_path(raw_dir / "metadata.pt", csv_path)

    assert resolved == metadata_path


def test_process_recast_catalog_handles_missing_expected_paths(tmp_path):
    raw_dir = tmp_path / "SCEDC" / "raw"
    raw_dir.mkdir(parents=True)
    _write_catalog_csv(raw_dir / "scedc.csv")
    _write_catalog_csv(raw_dir / "catalog.csv")

    metadata_path = tmp_path / "SCEDC" / "catalogs" / "abc12345" / "metadata.pt"
    metadata_path.parent.mkdir(parents=True)
    torch.save({"start_ts": pd.Timestamp("1981-01-01 00:00:00")}, metadata_path)

    df = process_recast_catalog(
        str(raw_dir / "SCEDC.csv"),
        str(raw_dir / "metadata.pt"),
    )

    expected_columns = ["t", "Magnitude", "Latitude", "Longitude", "Depth", "dt", "ts"]
    assert list(df.columns) == expected_columns
    assert df["dt"].iloc[0] == 0.0
    assert (raw_dir / "processed_scedc.csv").exists()


def test_resolve_recast_metadata_path_raises_when_multiple_hashed_metadata(tmp_path):
    raw_dir = tmp_path / "White" / "raw"
    raw_dir.mkdir(parents=True)
    csv_path = raw_dir / "white.csv"
    _write_catalog_csv(csv_path)

    metadata_a = tmp_path / "White" / "catalogs" / "aaa11111" / "metadata.pt"
    metadata_b = tmp_path / "White" / "catalogs" / "bbb22222" / "metadata.pt"
    metadata_a.parent.mkdir(parents=True)
    metadata_b.parent.mkdir(parents=True)
    torch.save({"start_ts": pd.Timestamp("2008-01-01 00:00:00")}, metadata_a)
    torch.save({"start_ts": pd.Timestamp("2008-01-01 00:00:00")}, metadata_b)

    with pytest.raises(FileNotFoundError, match="Multiple metadata\\.pt files found"):
        _resolve_recast_metadata_path(raw_dir / "metadata.pt", csv_path)


def test_process_recast_catalog_infers_start_ts_when_metadata_missing(tmp_path):
    raw_dir = tmp_path / "QTMSanJacinto" / "raw"
    raw_dir.mkdir(parents=True)
    _write_catalog_csv(raw_dir / "san_jacinto.csv")
    _write_catalog_csv(raw_dir / "catalog_san_jacinto.csv")

    df = process_recast_catalog(
        str(raw_dir / "SanJacinto.csv"),
        str(raw_dir / "metadata.pt"),
    )

    assert df["t"].iloc[0] > 0.0
    assert df["dt"].iloc[0] == 0.0
    assert (raw_dir / "processed_san_jacinto.csv").exists()
