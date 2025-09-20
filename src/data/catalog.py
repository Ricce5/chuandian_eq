from pathlib import Path
from typing import Any, Dict, Union
from src.utils.registrable import Registrable
import numpy as np
import torch


default_catalogs_dir = Path(__file__).parents[2] / "data"


class Catalog(Registrable):
    """Earthquake catalog that consists of multiple datasets.

    Args:
        root_dir: Directory where all files of the dataset should be stored.
        metadata: Information about the catalog.
    """

    def __init__(self, root_dir: Union[str, Path], metadata: Dict[str, Any]): # Union: 可以是多种类型中的一种
        self.norm_stats = {}
        # self.root_dir = Path(root_dir)
        norm_path = self.root_dir / "norm_stats.pt"
        if norm_path.exists():
            self.norm_stats = torch.load(norm_path,weights_only=False)


        self.root_dir = Path(root_dir).expanduser().resolve() # expanduser波浪号 (~) 扩展为当前用户的主目录路径 resolve：返回绝对路径
        self.metadata = metadata
        metadata_path = self.root_dir / "metadata.pt"
        if metadata_path.exists():
            existing_metadata = torch.load(metadata_path, weights_only=False)
            if existing_metadata == self.metadata and self.all_paths_exist():
                print(f"Loading existing catalog from {self.root_dir}.")
            else:
                raise FileExistsError(
                    f"A different catalog already exists in {self.root_dir}. "
                    f"There are two ways to fix this problem:\n"
                    f"  - Specify a different location as `root_dir`.\n"
                    f"  - Remove the existing catalog with\n     rm -rf {self.root_dir}"
                )
        else:
            print("Generating the catalog...")
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self.generate_catalog()
            torch.save(self.metadata, metadata_path)
            print(f"Catalog saved to {self.root_dir}")
            if not self.all_paths_exist():
                missing_paths = "\n  - ".join(
                    str(path.name) for path in self.required_paths if not path.exists()
                )
                raise RuntimeError(
                    "Catalog generation finished, but following required files haven't "
                    "been generated:\n  - "
                    f"{missing_paths}"
                    "\nOne of the methods `generate_catalog` or `required_files` "
                    "isn't implemented correctly."
                )
    
    def set_b_updater(self, b_updater):
        self.b_updater = b_updater

    def estimate_gr_b(self):
        if self.b_updater is not None:
            self.b_updater.fit(self.full_sequence)
        self._split_datasets()

    def _split_datasets(self):
        raise NotImplemented
        

    def generate_catalog(self):
        """Create the dataset from scratch and save it to self.root_dir.""" 
        raise NotImplementedError
    
        
    def normalize(self, name: str, data: torch.Tensor) -> torch.Tensor:
        if name not in self.norm_stats:
            raise ValueError(f"No normalization stats for '{name}'")
        min_val = self.norm_stats[name]["min"]
        max_val = self.norm_stats[name]["max"]
        range_val = max_val - min_val if (max_val - min_val) > 0 else 1e-8
        return (data - min_val) / range_val
    
    def denormalize(self, name: str, data: torch.Tensor) -> torch.Tensor:
        """Min-Max 反归一化"""
        if name not in self.norm_stats:
            raise ValueError(f"No normalization stats for '{name}'")
        min_val = self.norm_stats[name]["min"]
        max_val = self.norm_stats[name]["max"]
        return data * (max_val - min_val ) + min_val

    def normalize_fields(self, data_dict: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        对传入的每个字段计算 min-max 并执行归一化。

        Args:
            data_dict: 字段名到原始值 (np.ndarray 或 torch.Tensor) 的映射

        Returns:
            一个新的 dict，字段名到归一化后的 torch.Tensor 的映射
        """
        normed_data = {}
        for key, data in data_dict.items():
            # 转为 Tensor
            if isinstance(data, np.ndarray):
                data = torch.tensor(data, dtype=torch.float32)
            elif not isinstance(data, torch.Tensor):
                raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(data)} for key {key}")

            self.norm_stats[key] = {
                "min": float(data.min()),
                "max": float(data.max())
            }

            # 归一化
            normed_data[key] = (data - self.norm_stats[key]["min"]) / (self.norm_stats[key]["max"] - self.norm_stats[key]["min"] )

        torch.save(self.norm_stats, self.root_dir / "norm_stats.pt")
        return normed_data


    @property
    def required_files(self):
        """Names of files that the dataset consists of."""
        raise NotImplementedError

    @property
    def required_paths(self):
        """Full paths to the files that the dataset consists of."""
        return [self.root_dir / file for file in self.required_files]

    def all_paths_exist(self):
        """Check if all required files exist."""
        return all([path.exists() for path in self.required_paths])

    def __repr__(self):
        return f"{self.__class__.__name__}(root_dir={self.root_dir})"
