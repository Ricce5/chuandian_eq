"""Loader for virtual induced-ETAS artifacts.

Artifacts are created with
``src.utils.induced_virtual_experiment.save_virtual_induced_catalog`` and can
then be trained through the normal TPP CLI using
``dataset: VirtualInducedETAS`` and
``catalog_cfg.artifact_dir: /path/to/artifact``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from src.data import Catalog, TppDataset


@Catalog.register(name="VirtualInducedETAS-Standard")
class VirtualInducedETASStandard(Catalog):
    """Read-only catalog wrapper for a saved virtual induced-ETAS experiment."""

    def __init__(
        self,
        root_dir: Union[str, Path],
        artifact_dir: Union[str, Path, None] = None,
    ):
        self.root_dir = Path(artifact_dir if artifact_dir is not None else root_dir).expanduser().resolve()
        metadata_path = self.root_dir / "metadata.pt"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Virtual induced catalog metadata not found: {metadata_path}. "
                "Create it with save_virtual_induced_catalog() first."
            )
        metadata = torch.load(metadata_path, weights_only=False)
        if not isinstance(metadata, dict):
            raise ValueError(f"Expected dict metadata in {metadata_path}.")
        metadata.setdefault("name", "VirtualInducedETAS")
        metadata.setdefault("mag_completeness", 0.0)
        metadata.setdefault("mag_roundoff_error", 0.0)
        metadata.setdefault("freq", "1D")
        super().__init__(root_dir=self.root_dir, metadata=metadata)

        self.full_sequence = TppDataset.load_from_disk(self.root_dir / "full_sequence.pt")[0]
        self.train = TppDataset.load_from_disk(self.root_dir / "train.pt")
        self.val = TppDataset.load_from_disk(self.root_dir / "val.pt")
        self.test = TppDataset.load_from_disk(self.root_dir / "test.pt")

    @property
    def required_files(self):
        return ["full_sequence.pt", "train.pt", "val.pt", "test.pt", "metadata.pt"]

    def generate_catalog(self):
        raise RuntimeError(
            "VirtualInducedETASStandard is read-only. "
            "Use save_virtual_induced_catalog() to create its artifacts."
        )
