"""
metrics.py
----------
Metrik klasifikasi biner (murni numpy, tanpa torch/sklearn) agar mudah diaudit.

Konvensi: kelas positif = Dog (label 1), negatif = Cat (label 0).
Confusion matrix disusun dengan BARIS = label aktual, KOLOM = prediksi:

                 pred Cat   pred Dog
    aktual Cat     TN         FP
    aktual Dog     FN         TP
"""

from __future__ import annotations

import numpy as np


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _f1(p: float, r: float) -> float:
    return _safe_div(2 * p * r, p + r)


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Rank 1..n dengan rata-rata untuk nilai kembar (tie)."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    pos = y_true == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(y_prob.astype(float))
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """y_true: array {0,1}; y_prob: P(Dog) dalam [0,1]. Threshold TETAP 0.5 (tidak di-tuning)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if y_true.shape != y_prob.shape:
        raise ValueError(f"Shape tidak sama: {y_true.shape} vs {y_prob.shape}")
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n = len(y_true)

    prec_dog, rec_dog = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
    prec_cat, rec_cat = _safe_div(tn, tn + fn), _safe_div(tn, tn + fp)
    f1_dog, f1_cat = _f1(prec_dog, rec_dog), _f1(prec_cat, rec_cat)

    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    bce = float(-(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean())

    return {
        "n": n, "threshold": threshold,
        "class_counts": {"Cat": int((y_true == 0).sum()), "Dog": int((y_true == 1).sum())},
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": _safe_div(tp + tn, n),
        "balanced_accuracy": (rec_dog + rec_cat) / 2,
        "positive_class": "Dog",
        "precision_dog": prec_dog, "recall_dog": rec_dog, "f1_dog": f1_dog,
        "precision_cat": prec_cat, "recall_cat": rec_cat, "f1_cat": f1_cat,
        "macro_precision": (prec_dog + prec_cat) / 2,
        "macro_recall": (rec_dog + rec_cat) / 2,
        "macro_f1": (f1_dog + f1_cat) / 2,
        "bce_loss": bce,
        "roc_auc": roc_auc(y_true, y_prob),
    }
