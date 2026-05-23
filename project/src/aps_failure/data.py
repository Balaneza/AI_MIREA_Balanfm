from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


UCI_METADATA_ROWS = 20
TARGET_COLUMN = "class"
POSITIVE_LABEL = "pos"
NEGATIVE_LABEL = "neg"


def load_aps_dataframe(path: str | Path) -> pd.DataFrame:
    """Read the original UCI APS file with metadata header and `na` missing values."""
    df = pd.read_csv(
        path,
        skiprows=UCI_METADATA_ROWS,
        na_values=["na", "NA", ""],
        low_memory=False,
    )
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Column {TARGET_COLUMN!r} was not found in {path}")
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = df[TARGET_COLUMN].map({NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1})
    if target.isna().any():
        bad_labels = sorted(df.loc[target.isna(), TARGET_COLUMN].dropna().unique())
        raise ValueError(f"Unexpected target labels: {bad_labels}")

    features = df.drop(columns=[TARGET_COLUMN]).apply(pd.to_numeric, errors="coerce")
    return features, target.astype(int)


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(load_aps_dataframe(path))


def make_data_quality_report(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    missing_ratio = x_train.isna().mean().sort_values(ascending=False)
    return {
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(x_train.shape[1]),
        "train_positive": int(y_train.sum()),
        "train_negative": int((y_train == 0).sum()),
        "test_positive": int(y_test.sum()),
        "test_negative": int((y_test == 0).sum()),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "missing_cells_train_ratio": float(x_train.isna().mean().mean()),
        "top_missing_features": {
            name: float(value) for name, value in missing_ratio.head(20).items()
        },
    }


def normalize_sensor_value(value: Any) -> float:
    if value is None:
        return np.nan
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "na", "nan", "none", "null"}:
            return np.nan
        value = stripped
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def build_input_frame(
    features: dict[str, Any],
    feature_names: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    unknown_features = sorted(set(features) - set(feature_names))
    missing_features = [name for name in feature_names if name not in features]
    row = {name: normalize_sensor_value(features.get(name)) for name in feature_names}
    return pd.DataFrame([row], columns=feature_names), missing_features, unknown_features
