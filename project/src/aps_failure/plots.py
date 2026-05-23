from __future__ import annotations

import os
from pathlib import Path

from .config import PROJECT_ROOT

_matplotlib_cache = PROJECT_ROOT / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay


def _save_current(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_class_balance(y: pd.Series, path: Path) -> None:
    counts = y.map({0: "neg", 1: "pos"}).value_counts().reindex(["neg", "pos"])
    plt.figure(figsize=(6, 4))
    plt.bar(counts.index, counts.values, color=["#506784", "#d1495b"])
    plt.title("Class balance in training data")
    plt.ylabel("Rows")
    for idx, value in enumerate(counts.values):
        plt.text(idx, value, str(int(value)), ha="center", va="bottom")
    _save_current(path)


def plot_missingness(x: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    missing = x.isna().mean().sort_values(ascending=False).head(top_n).sort_values()
    plt.figure(figsize=(8, 6))
    plt.barh(missing.index, missing.values, color="#4c956c")
    plt.title(f"Top {top_n} features by missing-value ratio")
    plt.xlabel("Missing ratio")
    _save_current(path)


def plot_precision_recall(y_true, y_probability, path: Path, title: str) -> None:
    plt.figure(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_probability)
    plt.title(title)
    _save_current(path)


def plot_confusion_matrix(metrics: dict, path: Path, title: str) -> None:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xticks([0, 1], ["pred neg", "pred pos"])
    plt.yticks([0, 1], ["true neg", "true pos"])
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(matrix[row, col]), ha="center", va="center")
    plt.colorbar(fraction=0.046, pad=0.04)
    _save_current(path)


def plot_feature_importance(importance: list[dict], path: Path, top_n: int = 20) -> None:
    top = list(reversed(importance[:top_n]))
    names = [item["feature"] for item in top]
    values = [item["importance"] for item in top]
    plt.figure(figsize=(8, 6))
    plt.barh(names, values, color="#2a9d8f")
    plt.title(f"Top {top_n} permutation importances")
    plt.xlabel("Average precision drop")
    _save_current(path)
