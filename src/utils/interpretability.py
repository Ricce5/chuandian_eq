from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler



def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a torch tensor safely and move to CPU."""
    return tensor.detach().cpu().numpy()


def plot_sequence(
    x: torch.Tensor,
    sample_idx: int = -1,
    feature_idx: int = 2,
    t: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Plot a single feature sequence for a given sample.

    Parameters
    ----------
    x: shape (batch, tokens, features)
    sample_idx: which sample in batch to visualize (default last)
    feature_idx: which feature/channel to visualize
    t: optional token index array; if None, uses np.arange
    title: optional plot title
    """
    s = to_numpy(x[sample_idx, :, feature_idx])
    idx = t if (t is not None and len(t) == len(s)) else np.arange(len(s))

    # determine valid points: prefer provided mask, otherwise infer non-zero tokens
    if mask is not None:
        m = to_numpy(mask[sample_idx]).astype(bool)
    else:
        # infer from last dimension non-zero across features
        inferred = (x[sample_idx].abs().sum(dim=-1) > 0)
        m = to_numpy(inferred).astype(bool)

    # filter out masked positions
    idx_plot = idx[m]
    s_plot = s[m]

    plt.figure(figsize=(10, 4))
    plt.plot(idx_plot, s_plot, color="tab:blue", lw=1)
    plt.scatter(idx_plot, s_plot, s=8, color="tab:blue", alpha=0.8)
    plt.title(title or f"Sample {sample_idx}, feature {feature_idx}")
    plt.xlabel("Token index")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def get_attn_token_weights(model: torch.nn.Module, x: torch.Tensor):
    """
    Extract per-token attention weights and mask from model's attention pooling.

    Expects model to provide:
      - `model.base_model(x)` -> (enc_out [B, L, D], non_pad_mask [B, L, 1], ...)
      - `model.base_model.input_adapter.get_extra_inputs(x)` containing 'event_time'
      - `model.extractor.pool(pooling_input, mask)` returning (pooled, attn_weights [B, L])
    """
    enc_out, non_pad_mask, _ = model.base_model(x)  # [B, L, D], [B, L, 1]
    mask = non_pad_mask.squeeze(-1)  # [B, L]

    extra = {}
    if hasattr(model.base_model.input_adapter, "get_extra_inputs"):
        extra = model.base_model.input_adapter.get_extra_inputs(x)
    if "event_time" not in extra:
        raise RuntimeError("Extractor requires 'event_time' in extra inputs")

    event_time = extra["event_time"].unsqueeze(-1)  # [B, L, 1]
    pooling_input = torch.cat([enc_out, event_time], dim=-1)  # [B, L, D+1]
    _, attn_weights = model.extractor.pool(pooling_input, mask)  # [B, D+1], [B, L]
    return attn_weights, mask


def integrated_gradients_token_importance(
    model: torch.nn.Module,
    x: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 32,
):
    """
    Integrated Gradients for per-token importance; returns tensor [B, L].
    Aggregates absolute attributions across feature channels.
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    baseline = torch.zeros_like(x) if baseline is None else baseline.to(device)
    delta = x - baseline
    total_grad = torch.zeros_like(x)

    alphas = torch.linspace(0.0, 1.0, steps=n_steps, device=device)
    for alpha in alphas:
        xa = (baseline + alpha * delta).detach().requires_grad_(True)
        with torch.enable_grad():
            y = model(xa)
            if y.ndim > 0:
                y = y.sum()
            model.zero_grad(set_to_none=True)
            y.backward()
            ga = xa.grad
            total_grad = total_grad + ga

    avg_grad = total_grad / n_steps
    attributions = avg_grad * delta
    token_scores = attributions.abs().sum(dim=-1)  # [B, L]
    return token_scores



def compute_token_importance(
    model: torch.nn.Module,
    x: torch.Tensor,
    method: str = "ig",
    n_steps: int = 32,
):
    """
    Unified interface to compute per-token importance and optional mask.

    Returns
    -------
    scores: torch.Tensor [B, L]
    mask_plot: Optional[torch.Tensor] [B, L]
    label: str
    """
    label = ""
    if method == "attention":
        scores, mask_plot = get_attn_token_weights(model, x)
        label = "Attention weight"
    elif method == "ig":
        scores = integrated_gradients_token_importance(model, x, baseline=None, n_steps=n_steps)
        label = "IG importance"
    elif method == "grad":
        scores = gradient_saliency_token_importance(model, x)
        label = "Gradient saliency"
    else:
        raise ValueError(f"Unknown method: {method}")
    return scores, label


def gradient_saliency_token_importance(model: torch.nn.Module, x: torch.Tensor):
    """Vanilla gradient saliency aggregated over feature channels; returns [B, L]."""
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device).detach().requires_grad_(True)
    with torch.enable_grad():
        y = model(x)
        if y.ndim > 0:
            y = y.sum()
        model.zero_grad(set_to_none=True)
        y.backward()
        sal = x.grad.abs().sum(dim=-1)
    return sal



def plot_token_importance(
    scores: torch.Tensor,
    sample_idx: int = 0,
    label: str = "IG importance",
    mask: Optional[torch.Tensor] = None,
    color: str = "tab:red",
    sequence: Optional[torch.Tensor] = None,
    sequence_feature_idx: int = 2,
    sequence_label: str = "magnitude",
    sequence_color: str = "tab:blue",
) -> None:
    """Plot token importance for one sample, optionally overlay a sequence (e.g. magnitude).

    Both curves are drawn on the same figure using a secondary y-axis. Positions where
    `mask` is False will be omitted from both plots.
    """
    s = to_numpy(scores[sample_idx])
    L = len(s)
    t = np.arange(L)

    # compute mask: True == valid (plot), False == masked (skip)
    if mask is not None:
        m = to_numpy(mask[sample_idx]).astype(bool)
    else:
        m = np.ones(L, dtype=bool)

    t_plot = t[m]
    s_plot = s[m]

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t_plot, s_plot, label=label, color=color)
    ax1.scatter(t_plot, s_plot, s=10, color=color)
    ax1.set_ylabel(label, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    # optionally plot provided sequence (torch tensor of shape [B,L] or array)
    if sequence is not None:
        if isinstance(sequence, torch.Tensor):
            seq = to_numpy(sequence[sample_idx])
        else:
            seq = np.asarray(sequence)
        # if sequence has extra dims (e.g., [B,L,features]), pick feature idx
        if seq.ndim == 2:
            seq_vals = seq[sample_idx] if seq.shape[0] != L else seq[0]
        elif seq.ndim == 1 and len(seq) == L:
            seq_vals = seq
        elif seq.ndim == 2 and seq.shape[1] == L:
            seq_vals = seq[:, sequence_feature_idx] if seq.shape[0] == L else seq[:, sequence_feature_idx]
        elif seq.ndim == 3:
            seq_vals = seq[sample_idx, :, sequence_feature_idx]
        else:
            # best-effort flatten
            seq_vals = seq.reshape(-1)[:L]

        seq_vals = np.asarray(seq_vals)
        seq_plot = seq_vals[m]
        ax2 = ax1.twinx()
        ax2.plot(t_plot, seq_plot, label=sequence_label, color=sequence_color, lw=1)
        ax2.scatter(t_plot, seq_plot, s=8, color=sequence_color, alpha=0.9)
        ax2.set_ylabel(sequence_label, color=sequence_color)
        ax2.tick_params(axis="y", labelcolor=sequence_color)

    plt.title(f"Token Importance & {sequence_label} (sample {sample_idx})")
    # combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = (ax2.get_legend_handles_labels() if sequence is not None else ([], []))
    if handles1 or handles2:
        ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    plt.tight_layout()
    plt.show()


def rank_tokens_numeric(
    token_scores: torch.Tensor,
    sample_idx: int = 0,
    valid_mask: Optional[torch.Tensor] = None,
    topn: int = 20,
) -> Dict:
    """
    Rank tokens by importance for a given sample.

    Parameters
    ----------
    token_scores: [B, L]
    valid_mask: optional [L] boolean mask (True = valid token)
    """
    s = token_scores[sample_idx].detach().cpu()
    L = s.shape[0]

    if valid_mask is not None:
        vm = valid_mask.detach().cpu().bool()
        s_rank = s.clone()
        s_rank[~vm] = float("-inf")
    else:
        s_rank = s

    indices = torch.arange(L)
    # sort descending
    sorted_idx = torch.argsort(s_rank, descending=True)
    top = [(int(i.item()), float(s[int(i.item())].item())) for i in sorted_idx[:topn]]
    # sort ascending for least important
    least_sorted_idx = torch.argsort(s_rank, descending=False)
    least = [(int(i.item()), float(s[int(i.item())].item())) for i in least_sorted_idx[:topn]]

    return {
        "top": top,  # list of (token_idx, score)
        "least": least,  # list of (token_idx, score)
        "sorted_idx": sorted_idx.tolist(),
        "least_sorted_idx": least_sorted_idx.tolist(),
    }


def _compute_baseline_vector(
    x: torch.Tensor,
    sample_idx: int,
    strategy: str = "zeros",  # 'zeros' | 'mean_sample' | 'mean_batch'
) -> torch.Tensor:
    """Compute a baseline feature vector of shape [D] according to the strategy."""
    if strategy == "zeros":
        return torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
    elif strategy == "mean_sample":
        return x[sample_idx].mean(dim=0)
    elif strategy == "mean_batch":
        return x.mean(dim=(0, 1))
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")


def deletion_validation_numeric(
    model: torch.nn.Module,
    x: torch.Tensor,  # [B, L, D]
    token_scores: torch.Tensor,  # [B, L]
    sample_idx: int = 0,
    topk_list: Tuple[int, ...] = (1, 2, 5, 10),
    deletion_mode: str = "zero",  # 'zero' | 'baseline' | 'noise'
    baseline_strategy: str = "zeros",
    valid_mask: Optional[torch.Tensor] = None,
    return_details: bool = True,
) -> Dict:
    """
    Deletion validation on numeric tokens.

    - For each k in topk_list, deletes top-k important tokens and measures drop in prediction.
    - Also computes least-k deletion as sanity.
    - Deletion is applied only to valid tokens if valid_mask is provided.
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        y_base = model(x)
    # squeeze to scalar per sample if needed
    y_base_sample = y_base[sample_idx]
    if y_base_sample.ndim > 0:
        y_base_val = float(y_base_sample.squeeze().item())
    else:
        y_base_val = float(y_base_sample.item())

    # ranking
    ranks = rank_tokens_numeric(token_scores, sample_idx=sample_idx, valid_mask=valid_mask, topn=max(topk_list))
    top_indices = [i for i, _ in ranks["top"]]  # already sorted desc
    least_indices = [i for i, _ in ranks["least"]]  # sorted asc

    # baseline vector for 'baseline' mode
    baseline_vec = _compute_baseline_vector(x, sample_idx, baseline_strategy)

    def _apply_del(x_in: torch.Tensor, del_indices: List[int]) -> torch.Tensor:
        x_mod = x_in.clone()
        for di in del_indices:
            # skip invalid tokens
            if valid_mask is not None:
                if not bool(valid_mask[di].item()):
                    continue
            if deletion_mode == "zero":
                x_mod[sample_idx, di, :] = 0.0
            elif deletion_mode == "baseline":
                x_mod[sample_idx, di, :] = baseline_vec
            elif deletion_mode == "noise":
                noise = torch.randn_like(x_mod[sample_idx, di, :]) * x_mod.std()
                x_mod[sample_idx, di, :] = noise
            else:
                raise ValueError(f"Unknown deletion_mode: {deletion_mode}")
        return x_mod

    results = {"topk": [], "leastk": []}
    details = {"topk": [], "leastk": []} if return_details else None

    # evaluate top-k deletions
    for k in topk_list:
        del_idx = top_indices[:k]
        x_top = _apply_del(x, del_idx)
        with torch.no_grad():
            y_top = model(x_top)[sample_idx]
        y_top_val = float(y_top.squeeze().item()) if y_top.ndim > 0 else float(y_top.item())
        results["topk"].append({"k": k, "score": y_top_val, "drop": y_base_val - y_top_val})
        if return_details:
            details["topk"].append({"k": k, "deleted_indices": del_idx})

    # evaluate least-k deletions
    for k in topk_list:
        del_idx = least_indices[:k]
        x_least = _apply_del(x, del_idx)
        with torch.no_grad():
            y_least = model(x_least)[sample_idx]
        y_least_val = float(y_least.squeeze().item()) if y_least.ndim > 0 else float(y_least.item())
        results["leastk"].append({"k": k, "score": y_least_val, "drop": y_base_val - y_least_val})
        if return_details:
            details["leastk"].append({"k": k, "deleted_indices": del_idx})

    return {"baseline": y_base_val, "results": results, "details": details}

