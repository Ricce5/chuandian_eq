from pathlib import Path

from src.utils.file_utils import find_latest_model_path


def _write_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "last_model_1.pth").write_bytes(b"checkpoint")


def test_find_latest_model_path_matches_model_prefix_exactly(tmp_path):
    _write_checkpoint(tmp_path / "etas_20260610-100000")
    _write_checkpoint(tmp_path / "netas_20260610-220952")

    latest_path = find_latest_model_path("etas", checkpoint_root=str(tmp_path))

    assert Path(latest_path).name == "etas_20260610-100000"


def test_find_latest_model_path_uses_latest_exact_model_directory(tmp_path):
    _write_checkpoint(tmp_path / "etas_20260609-100000")
    _write_checkpoint(tmp_path / "etas_20260610-100000")
    _write_checkpoint(tmp_path / "netas_20260610-220952")

    latest_path = find_latest_model_path("etas", checkpoint_root=str(tmp_path))

    assert Path(latest_path).name == "etas_20260610-100000"
