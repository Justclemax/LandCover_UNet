"""Segmentation metrics from a confusion matrix (classes 1..K, 0 ignored)."""

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    """(K, K) counts, rows = reference, columns = prediction. Pixels with y_true == 0 are skipped."""
    t, p = y_true.ravel().astype(np.int64), y_pred.ravel().astype(np.int64)
    keep = (t > 0) & (p > 0)
    return np.bincount((t[keep] - 1) * k + (p[keep] - 1), minlength=k * k).reshape(k, k)


def scores(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    ref, pred, n = cm.sum(axis=1), cm.sum(axis=0), cm.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(ref > 0, tp / ref, np.nan)
        precision = np.where(pred > 0, tp / pred, np.nan)
        f1 = np.where(ref + pred > 0, 2 * tp / (ref + pred), np.nan)
        iou = np.where(ref + pred - tp > 0, tp / (ref + pred - tp), np.nan)
    oa = tp.sum() / n
    pe = (ref * pred).sum() / n**2
    present = ref > 0
    return {
        "pixels": int(n),
        "overall_accuracy": float(oa),
        "kappa": float((oa - pe) / (1 - pe)),
        "mean_iou": float(np.nanmean(iou[present])),
        "macro_f1": float(np.nanmean(f1[present])),
        "balanced_accuracy": float(np.nanmean(recall[present])),
        "per_class": {"iou": iou.tolist(), "f1": f1.tolist(), "precision": precision.tolist(),
                      "recall": recall.tolist(), "support": ref.tolist()},
    }
