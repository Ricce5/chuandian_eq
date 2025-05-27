import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from scipy.stats import binom
from tqdm import tqdm
import os

def compute_metrics(targets, preds, threshold=None, optimize_metric="f1"):
    preds = np.array(preds)
    targets = np.array(targets)

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
            elif optimize_metric == "youden":
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