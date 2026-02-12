from typing import Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    return_X: bool = False,
):
    """
    Run the model across a DataLoader and collect predictions, labels,
    and optionally pooled features via `model.get_pooled_representation()`.

    Returns
    -------
    y_pred_all: np.ndarray
        Model predictions of shape (N,) or (N, C)
    y_all: np.ndarray
        Ground truth labels of shape (N,) or (N, C)
    X_all: Optional[np.ndarray]
        Pooled features if `return_X=True`, else None
    """
    y_pred_list = []
    y_list = []
    X_list = [] if return_X else None

    model.eval()
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        if return_X:
            X = model.get_pooled_representation(x)
            X_list.append(X.detach().cpu())
        y_pred = model(x)
        y_pred_list.append(y_pred.detach().cpu())
        y_list.append(y.detach().cpu())

    y_pred_all = torch.cat(y_pred_list, dim=0).cpu().numpy()
    # flatten if shape is (N,1)
    if y_pred_all.ndim == 2 and y_pred_all.shape[1] == 1:
        y_pred_all = y_pred_all.squeeze(1)
    y_all = torch.cat(y_list, dim=0).cpu().numpy()

    if return_X:
        X_all = torch.cat(X_list, dim=0).cpu().numpy()
        return y_pred_all, y_all, X_all
    return y_pred_all, y_all, None


def tsne_scatter(
    X_all: np.ndarray,
    y_all: np.ndarray,
    y_pred_all: np.ndarray,
    perplexity: float = 30,
    learning_rate: float = 200,
    n_iter: int = 1000,
    title_left: str = "t-SNE colored by true label (y_all)",
    title_right: str = "t-SNE colored by model score (y_pred_all)",
    show: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features, run t-SNE, and plot two panels colored by
    truth and normalized prediction score. Returns standardized features,
    2D embeddings, and normalized prediction scores.
    """
    X = X_all.astype(float)
    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=n_iter,
        random_state=0,
    )
    X_tsne = tsne.fit_transform(X_scaled)

    y_min = float(np.min(y_pred_all))
    y_max = float(np.max(y_pred_all))
    y_score = (y_pred_all - y_min) / (y_max - y_min + 1e-8)

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # Left panel: true labels
    if y_all is not None:
        sc0 = axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_all, cmap="coolwarm", s=8, alpha=0.8)
        axs[0].set_title(title_left)
        plt.colorbar(sc0, ax=axs[0])
    else:
        axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], s=8, alpha=0.8)
        axs[0].set_title("t-SNE (no label)")

    # Right panel: normalized prediction score
    if y_score is not None:
        sc1 = axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_score, cmap="viridis", s=8, alpha=0.8)
        axs[1].set_title(title_right)
        plt.colorbar(sc1, ax=axs[1])
    else:
        axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], s=8, alpha=0.8)
        axs[1].set_title("t-SNE (no score)")

    plt.tight_layout()
    if show:
        plt.show()

    return X_scaled, X_tsne, y_score
