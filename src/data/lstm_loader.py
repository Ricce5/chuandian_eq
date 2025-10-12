import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
plt.rcParams['axes.unicode_minus'] = False
from torch.utils.data import Subset, Dataset
from src.data.utils import get_split_indices

class LSTMDataset(Dataset):
    def __init__(self, X, y, scalars=None):
        """
        Initialize the custom dataset.
        :param X: Input feature data.
        :param y: Target label data.
        :param scalars: Dictionary of scalers used for normalization (e.g., MinMaxScaler).
        """
        self.X = X
        self.y = y
        self.scalars = scalars 
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    def inverse_normalize_label(self, norm_value):
        """
        Perform inverse normalization of the label using the provided scaler.
        Supports input shapes of (B, F) or (B,), and the output shape matches the input.
        :param norm_value: Normalized value with shape (B, F) or (B,)
        :return: Inverse normalized value with the same shape as norm_value
        """
        if 'Mag_max_obs' in self.scalars:
            scalar = self.scalars['Mag_max_obs']
            norm_value_np = np.array(norm_value)
            orig_shape = norm_value_np.shape
            norm_value_2d = norm_value_np.reshape(-1, 1) if norm_value_np.ndim == 1 else norm_value_np
            value = scalar.inverse_transform(norm_value_2d)
            value = value.reshape(orig_shape)
            return torch.tensor(value, dtype=torch.float32)
        else:
            raise ValueError("Scaler for the label column not found.")


def normalize_df(df):
    scalars = {}
    df_nl = df.copy()
    for col in df_nl.columns:
        if col not in ['time']:
            scalar = MinMaxScaler()
            df_nl[col] = scalar.fit_transform(df_nl[col].values.reshape(-1, 1))
            scalars[col] = scalar
    return df_nl, scalars



def clean_data(X, y, nan_value_for_x=0.0, verbose=True):
    """
    Clean the input features X and labels y:
    - Remove samples where y is NaN (and the corresponding X)
    - Replace NaN values in X with a specified value (default is 0.0)
    
    Parameters:
        X (np.ndarray): Input features, a 2D array (n_samples, n_features)
        y (np.ndarray): Target variable, a 1D or 2D array
        nan_value_for_x (float): Value to replace NaN in X, default is 0.0
        verbose (bool): Whether to print processing logs

    Returns:
        X_clean (np.ndarray): Cleaned X
        y_clean (np.ndarray): Cleaned y
    """
    X = np.array(X)
    y = np.array(y)

    valid_mask = ~np.isnan(y).flatten()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]

    X_clean = np.nan_to_num(X_clean, nan=nan_value_for_x)
    if verbose:
        print(f"   Original number of samples: {len(y)}, number of samples after cleaning: {len(y_clean)}")
        print(f"   Replaced NaN values in X with {nan_value_for_x}")

    return X_clean, y_clean


def split_dataset(X, y, by_time=False, batch_size=64, seed=0, train_ratio=0.7, val_ratio=0.15, time_order=('train','val', 'test'),scalars=None):
    dataset = LSTMDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        scalars=scalars
    )

    total = len(dataset)
    train_idx, val_idx, test_idx = get_split_indices(
        total_length=total,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        by_time=by_time,
        time_order= time_order
    )


    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds = Subset(dataset, test_idx)


    print(f"Train set: {len(train_ds)} samples")
    print(f"Validation set: {len(val_ds)} samples")
    print(f"Test set: {len(test_ds)} samples")


    data_loaders = {
        'train': DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        'val': DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    }

    return dataset, data_loaders
