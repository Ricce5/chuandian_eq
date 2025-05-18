import torch
from torch.utils.data import TensorDataset, Subset, DataLoader
from .data_utils import get_split_indices  # 使用统一的划分函数
from .preprocessing import map_region,process_df


def split_dataset(f, mag, y, split_by_time=False, batch_size=64, seed=70, train_ratio=0.7, val_ratio=0.15):
    dataset = TensorDataset(
        torch.tensor(f, dtype=torch.float32),
        torch.tensor(mag, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32)
    )

    total = len(dataset)
    train_idx, val_idx, test_idx = get_split_indices(
        total_length=total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        by_time=split_by_time
    )

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)

    def count_labels(subset):
        labels = torch.stack([subset[i][2] for i in range(len(subset))])
        positive = (labels > 0).sum().item()
        negative = (labels <= 0).sum().item()
        return positive, negative

    train_pos, train_neg = count_labels(train_ds)
    val_pos, val_neg = count_labels(val_ds)
    test_pos, test_neg = count_labels(test_ds)

    print(f"Train set: {train_pos} positive, {train_neg} negative")
    print(f"Validation set: {val_pos} positive, {val_neg} negative")
    print(f"Test set: {test_pos} positive, {test_neg} negative")

    return {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }
