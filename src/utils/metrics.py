import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binom
from tqdm import tqdm
import os
from scipy.special import expit 

def classification_metrics(targets, preds, threshold=None, optimize_metric="f1",apply_sigmoid=True):
    preds = np.array(preds)
    targets = np.array(targets)

    if apply_sigmoid:
        preds = expit(preds)  # sigmoid 函数
    # 自动选阈值（默认基于 F1）
    if threshold is None:
        fpr, tpr, thresholds = roc_curve(targets, preds)
        best_thresh = 0.5
        best_score = -1

        for th in thresholds:
            preds_bin = (preds > th).astype(int)
            if optimize_metric == "f1":
                score = f1_score(targets, preds_bin, zero_division=0)
            elif optimize_metric == "precision":
                score = precision_score(targets, preds_bin, zero_division=0)
            elif optimize_metric == "recall":
                score = recall_score(targets, preds_bin, zero_division=0)
            elif optimize_metric == "youden": # Youden's J statistic
                TP = np.sum((preds_bin == 1) & (targets == 1))
                FP = np.sum((preds_bin == 1) & (targets == 0))
                FN = np.sum((preds_bin == 0) & (targets == 1))
                TN = np.sum((preds_bin == 0) & (targets == 0))
                TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
                FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
                score = TPR - FPR
            else:
                raise ValueError(f"Unknown optimize_metric: {optimize_metric}")

            if score > best_score:
                best_score = score
                best_thresh = th

        threshold = best_thresh

    # 应用最终 threshold 进行评估
    preds_bin = (preds > threshold).astype(int)
    precision = precision_score(targets, preds_bin, zero_division=0)
    recall = recall_score(targets, preds_bin, zero_division=0)
    f1 = f1_score(targets, preds_bin, zero_division=0)
    auc = roc_auc_score(targets, preds)

    TP = np.sum((preds_bin == 1) & (targets == 1))
    FP = np.sum((preds_bin == 1) & (targets == 0))
    FN = np.sum((preds_bin == 0) & (targets == 1))
    TN = np.sum((preds_bin == 0) & (targets == 0))
    FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
    R = TPR - FPR

    total = len(targets)
    correct = np.sum(preds_bin == targets)
    conf = compute_confidence(total, correct, FPR)

    # 转换所有值为原生 Python 类型，防止 json.dump 报错
    metrics_dict = {
        k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64))
        else int(v) if isinstance(v, (np.integer,))
        else v)
        for k, v in {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auc": auc,
            "fpr": FPR,
            "tpr": TPR,
            "R": R,
            "conf": conf,
            "threshold": threshold
        }.items()
    }

    return metrics_dict


def log_metrics(metrics, prefix=""):
    log_str = f"{prefix}".ljust(12)
    for k, v in metrics.items():
        if isinstance(v, float):
            log_str += f"{k}: {v:.4f} | "
    tqdm.write(log_str.strip(" | "))



def compute_confidence(N, m, b):
    """二项分布置信度计算"""
    return binom.cdf(m - 1, N, b)

def plot_and_save_roc_curve(targets, preds, auc_value, save_dir, filename="roc_curve.png"):
    """
    绘制并保存 ROC 曲线图像。

    参数:
    - targets: numpy 数组，真实标签
    - preds: numpy 数组，预测值（概率）
    - auc_value: float，AUC 指标值
    - save_dir: str，保存目录
    - filename: str，图像文件名
    """
    try:
        fpr, tpr, _ = roc_curve(targets, preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_value:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
        plt.show()
    except Exception as e:
        print("Failed to plot ROC curve:", e)

def plot_classification_distribution(preds, labels, title="Prediction Distribution (Training Set)", save_path=None):
    preds = np.array(preds)
    labels = np.array(labels)

    pos_preds = preds[labels == 1]
    neg_preds = preds[labels == 0]

    plt.figure(figsize=(8, 5))
    sns.histplot(pos_preds, bins=30, color='blue', label='Positive', stat='density', kde=True, alpha=0.6)
    sns.histplot(neg_preds, bins=30, color='yellow', label='Negative', stat='density', kde=True, alpha=0.6)

    plt.title(title)
    plt.xlabel("Predicted Probability (after Sigmoid)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path,dpi=100)
    plt.show()

def regression_metrics(y_true, y_pred):
    import numpy as np
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    RMSE = np.sqrt(np.mean(np.square(y_true - y_pred)))
    MAE = np.mean(np.abs(y_true - y_pred))
    MSE = np.mean(np.square(y_true - y_pred))

    # 避免除以0的问题
    nonzero_real = y_true != 0
    if np.any(nonzero_real):
        MAPE = np.mean(np.abs((y_true[nonzero_real] - y_pred[nonzero_real]) / y_true[nonzero_real])) * 100
    else:
        MAPE = np.nan

    SS_res = np.sum((y_true - y_pred) ** 2)
    SS_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    R2 = 1 - SS_res / SS_tot if SS_tot != 0 else np.nan


    metrics_dict = {
        k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64))
        else int(v) if isinstance(v, (np.integer,))
        else v)
        for k, v in {
        "RMSE": RMSE,
        "MAE": MAE,
        "MSE": MSE,
        "MAPE": MAPE,
        "R2": R2    
      }.items()
    }

    return metrics_dict

def count_metrics(y_true, y_pred):
    y_pred = np.round(np.exp(y_pred))  # 从 log(λ) 转换为 λ，并四舍五入
    y_true = np.round(y_true)

    mae = np.mean(np.abs(y_true - y_pred))
    acc = np.mean(y_true == y_pred)
    return {'MAE': mae, 'ExactMatchAcc': acc}



def plot_regression_scatter(data_dict, save_path=None,verbose=False):
    """
    Scatter plot of predicted vs true values for regression.
    data_dict: {
        "Train": (true_values, pred_values),
        "Validation": (true_values, pred_values),
        "Test": (true_values, pred_values)
    }
    """
    colors = {"Train": "blue", "Validation": "green", "Test": "black"}
    markers = {"Train": "x", "Validation": "^", "Test": "o"}

    plt.figure(figsize=(10, 8))

    for label, (true_vals, pred_vals) in data_dict.items():
        plt.scatter(true_vals, pred_vals, 
                    color=colors.get(label, "gray"), 
                    marker=markers.get(label, "o"), 
                    s=15, label=f"{label} Set")

    # 参考线
    x = np.linspace(4, 8.5, 100)
    plt.plot(x, x, 'r-', label='Ideal: y = x')
    plt.plot(x, x + 0.5, 'gray', linestyle='--', label='y = x + 0.5')
    plt.plot(x, x - 0.5, 'gray', linestyle='--', label='y = x - 0.5')

    plt.xlim(4, 8.5)
    plt.ylim(4, 8.5)
    plt.xlabel("True Magnitude", fontsize=14)
    plt.ylabel("Predicted Magnitude", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        if verbose:
            print(f"[✔] Saved scatter plot to {save_path}")
    plt.show()


def plot_regression_series(data_dict, save_path=None, verbose=False):
    """
    Line plot of true vs predicted values across time/index.
    """
    plt.figure(figsize=(16, 8))
    idx_start = 0

    for label, (true_vals, pred_vals) in data_dict.items():
        n = len(true_vals)
        idx_range = np.arange(idx_start, idx_start + n)

        plt.plot(idx_range, true_vals, label=f'{label} - True')
        plt.plot(idx_range, pred_vals, label=f'{label} - Predicted', linestyle='--')
        idx_start += n

    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Magnitude", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        if verbose:
            print(f"[✔] Saved series plot to {save_path}")
    plt.show()

def plot_count_scatter(data_dict, save_path=None, verbose=False):
    """
    Scatter plot of predicted vs true values for count data.
    data_dict: {
        "Train": (true_values, pred_values),
        "Validation": (true_values, pred_values),
        "Test": (true_values, pred_values)    
    }   
    """
    colors = {"Train": "blue", "Validation": "green", "Test": "black"}
    markers = {"Train": "x", "Validation": "^", "Test": "o"}

    plt.figure(figsize=(10, 8))

    for label, (true_vals, pred_vals) in data_dict.items():
        plt.scatter(true_vals, pred_vals, 
                    color=colors.get(label, "gray"), 
                    marker=markers.get(label, "o"), 
                    s=15, label=f"{label} Set")

    # 参考线
    x = np.linspace(0, 20, 100)
    plt.plot(x, x, 'r-', label='Ideal: y = x')
    plt.plot(x, x + 1, 'gray', linestyle='--', label='y = x + 1')
    plt.plot(x, x - 1, 'gray', linestyle='--', label='y = x - 1')

    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.xlabel("True Count", fontsize=14)
    plt.ylabel("Predicted Count", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        if verbose:
            print(f"[✔] Saved scatter plot to {save_path}")
    plt.show()


def plot_count_series(data_dict, save_path=None, verbose=False):
    """
    Line plot of true vs predicted values across time/index for count data.
    """
    plt.figure(figsize=(16, 8))
    idx_start = 0

    for label, (true_vals, pred_vals) in data_dict.items():
        n = len(true_vals)
        idx_range = np.arange(idx_start, idx_start + n)

        plt.plot(idx_range, true_vals, label=f'{label} - True')
        plt.plot(idx_range, pred_vals, label=f'{label} - Predicted', linestyle='--')
        idx_start += n

    plt.xlabel("Sample Index", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        if verbose:
            print(f"[✔] Saved series plot to {save_path}")
    plt.show()