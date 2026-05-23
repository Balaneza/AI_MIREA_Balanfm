from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def cost_from_confusion(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
    false_positive_cost: int = 10,
    false_negative_cost: int = 500,
) -> int:
    del tn, tp
    return int(fp * false_positive_cost + fn * false_negative_cost)


def classification_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    false_positive_cost: int = 10,
    false_negative_cost: int = 500,
) -> int:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return cost_from_confusion(tn, fp, fn, tp, false_positive_cost, false_negative_cost)


def binary_report(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    threshold: float,
    false_positive_cost: int = 10,
    false_negative_cost: int = 500,
) -> dict[str, Any]:
    y_pred = (y_probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    report = {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_probability)),
        "total_cost": cost_from_confusion(
            int(tn),
            int(fp),
            int(fn),
            int(tp),
            false_positive_cost,
            false_negative_cost,
        ),
    }
    try:
        report["roc_auc"] = float(roc_auc_score(y_true, y_probability))
    except ValueError:
        report["roc_auc"] = None
    return report


def find_best_cost_threshold(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    false_positive_cost: int = 10,
    false_negative_cost: int = 500,
) -> tuple[float, dict[str, Any]]:
    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.0, 0.5, 1.0]),
                np.quantile(y_probability, np.linspace(0.0, 1.0, 501)),
                np.linspace(0.001, 0.999, 250),
            ]
        )
    )
    best_threshold = 0.5
    best_report: dict[str, Any] | None = None
    for threshold in thresholds:
        report = binary_report(
            y_true,
            y_probability,
            float(threshold),
            false_positive_cost,
            false_negative_cost,
        )
        if best_report is None:
            best_report = report
            best_threshold = float(threshold)
            continue
        current_key = (
            report["total_cost"],
            -report["recall"],
            -report["f1"],
            -report["precision"],
        )
        best_key = (
            best_report["total_cost"],
            -best_report["recall"],
            -best_report["f1"],
            -best_report["precision"],
        )
        if current_key < best_key:
            best_report = report
            best_threshold = float(threshold)
    if best_report is None:
        raise ValueError("Could not choose a threshold")
    return best_threshold, best_report
