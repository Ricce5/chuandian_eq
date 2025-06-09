import os
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
import torch

from src.data import Catalog,TppDataset , Sequence, default_catalogs_dir
from src.catalogs.catalog_utils import train_val_test_split_sequence


def trim(x_min, x_max, p=0.05):
    length = x_max - x_min
    return x_min + length * p, x_min + length * (1 - p)


class CD(Catalog):
    def __init__(
        self,
        root_dir: Union[str, Path] = default_catalogs_dir / "CD",
        catalog_file: Union[str, Path] = None,
        mag_completeness: float = 3.0,
        train_start_ts: pd.Timestamp = pd.Timestamp("1975-01-01"),
        val_start_ts: pd.Timestamp = pd.Timestamp("1995-01-01"),
        test_start_ts: pd.Timestamp = pd.Timestamp("2007-01-01"),
    ):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.metadata = {
            "name": "CD",
            "freq": "1D",
            "mag_roundoff_error": 0.01,
            "mag_completeness": mag_completeness,
            "start_ts": pd.Timestamp("1970-01-01"),
            "end_ts": pd.Timestamp("2023-08-14"),
        }

        # 若 metadata.pt 不存在，生成 full_sequence.pt 和 metadata.pt
        if not (self.root_dir / "metadata.pt").exists():
            assert catalog_file is not None, "Must provide catalog_file to generate dataset"
            self.generate_catalog(catalog_file)
            torch.save(self.metadata, self.root_dir / "metadata.pt")

        # 调用父类（此时 metadata.pt 已存在）
        super().__init__(root_dir=self.root_dir, metadata=self.metadata)

        # 加入训练/验证/测试起始时间
        self.metadata["train_start_ts"] = pd.Timestamp(train_start_ts)
        self.metadata["val_start_ts"] = pd.Timestamp(val_start_ts)
        self.metadata["test_start_ts"] = pd.Timestamp(test_start_ts)

        # 加载 full_sequence
        self.full_sequence =TppDataset.load_from_disk(self.root_dir / "full_sequence.pt")[0]

        # 划分子集
        seq_train, seq_val, seq_test = train_val_test_split_sequence(
            seq=self.full_sequence,
            start_ts=self.metadata["start_ts"],
            train_start_ts=self.metadata["train_start_ts"],
            val_start_ts=self.metadata["val_start_ts"],
            test_start_ts=self.metadata["test_start_ts"],
        )

        self.train =TppDataset([seq_train])
        self.val =TppDataset([seq_val])
        self.test =TppDataset([seq_test])

    @property
    def required_files(self):
        return ["full_sequence.pt", "metadata.pt"]

    def generate_catalog(self, catalog_file: Union[str, Path]):
        column_names = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',
                        'Latitude', 'Longitude', 'Depth', 'Magnitude']
        df = pd.read_csv(
            catalog_file, header=None, names=column_names, sep=r"\s+"
        )

        # 构建时间戳列
        df['time'] = pd.to_datetime(
            df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce'
        ) + pd.to_timedelta(df['Second'], unit='s')
        df = df[['time', 'Magnitude', 'Latitude', 'Longitude', 'Depth']]

        # 筛选事件
        df = df[df["Magnitude"] > self.metadata["mag_completeness"]].copy()
        df.sort_values("time", inplace=True)
        df["time_diff"] = df["time"].diff().dt.total_seconds()
        ###############################
        df["time_diff"] = df["time"].diff().dt.total_seconds()
        df = df[df["time_diff"] > 0].copy()
        ################################
        # 时间范围计算
        start_ts = self.metadata["start_ts"]
        end_ts = self.metadata["end_ts"]
        t_start = 0.0
        t_end = (end_ts - start_ts) / pd.Timedelta("1D")

        arrival_times = ((df["time"] - start_ts) / pd.Timedelta("1D")).values
        inter_times = np.diff(arrival_times, prepend=[t_start], append=[t_end])
        mag = df["Magnitude"].values

        seq = Sequence(
            inter_times=torch.tensor(inter_times, dtype=torch.float32),
            t_start=t_start,
            mag=torch.tensor(mag, dtype=torch.float32),
        )

        dataset =TppDataset([seq])
        dataset.save_to_disk(self.root_dir / "full_sequence.pt")


cd = CD(
    root_dir="/root/autodl-tmp/chuandian_eq/data/CD2021/raw",
    catalog_file="/root/autodl-tmp/chuandian_eq/data/CD2021/raw/CD2021.dat"
)
