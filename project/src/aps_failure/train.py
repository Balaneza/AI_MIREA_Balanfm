from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .config import PROJECT_ROOT, load_config
from .data import load_dataset, make_data_quality_report
from .metrics import binary_report, find_best_cost_threshold
from .modeling import build_model_candidates
from .plots import (
    plot_class_balance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_missingness,
    plot_precision_recall,
)


LOGGER = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=_json_default)


def _write_runs_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "threshold",
        "val_average_precision",
        "val_recall",
        "val_f1",
        "val_total_cost",
        "test_average_precision",
        "test_recall",
        "test_f1",
        "test_total_cost",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row["model"],
                    "threshold": row["threshold"],
                    "val_average_precision": row["validation"]["average_precision"],
                    "val_recall": row["validation"]["recall"],
                    "val_f1": row["validation"]["f1"],
                    "val_total_cost": row["validation"]["total_cost"],
                    "test_average_precision": row["test"]["average_precision"],
                    "test_recall": row["test"]["recall"],
                    "test_f1": row["test"]["f1"],
                    "test_total_cost": row["test"]["total_cost"],
                }
            )


def _make_demo_payload(x_test, y_test, output_path: Path) -> None:
    positive_indexes = np.flatnonzero(y_test.to_numpy() == 1)
    row_index = int(positive_indexes[0] if len(positive_indexes) else 0)
    features = {
        column: (None if x_test.iloc[row_index][column] != x_test.iloc[row_index][column] else float(x_test.iloc[row_index][column]))
        for column in x_test.columns
    }
    _write_json(
        output_path,
        {
            "request_id": "demo-positive-scania-aps",
            "features": features,
        },
    )


def _compute_permutation_importance(
    model: object,
    x_val,
    y_val,
    feature_names: list[str],
    n_repeats: int,
    max_samples: int,
    random_state: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    LOGGER.info("Computing permutation importance for the selected model")
    result = permutation_importance(
        model,
        x_val,
        y_val,
        scoring="average_precision",
        n_repeats=n_repeats,
        max_samples=min(max_samples, len(x_val)),
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rows = []
    for name, mean, std in zip(feature_names, result.importances_mean, result.importances_std):
        rows.append(
            {
                "feature": name,
                "importance": float(max(mean, 0.0)),
                "std": float(std),
            }
        )
    return sorted(rows, key=lambda item: item["importance"], reverse=True)


def run_training(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    artifacts_dir = PROJECT_ROOT / "artifacts"
    figures_dir = artifacts_dir / "figures"

    LOGGER.info("Loading APS data")
    x_train_full, y_train_full = load_dataset(config.train_path)
    x_test, y_test = load_dataset(config.test_path)
    feature_names = list(x_train_full.columns)

    quality_report = make_data_quality_report(x_train_full, y_train_full, x_test, y_test)
    _write_json(artifacts_dir / "data_quality.json", quality_report)
    _write_json(
        artifacts_dir / "feature_schema.json",
        {
            "target": "class",
            "positive_label": "pos",
            "negative_label": "neg",
            "feature_count": len(feature_names),
            "features": feature_names,
        },
    )
    _make_demo_payload(x_test, y_test, PROJECT_ROOT / "data" / "demo_payload.json")

    plot_class_balance(y_train_full, figures_dir / "class_balance.png")
    plot_missingness(x_train_full, figures_dir / "missingness_top20.png")

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=config.validation_size,
        stratify=y_train_full,
        random_state=config.random_state,
    )

    candidates = build_model_candidates(config.random_state, config.n_jobs)
    runs: list[dict[str, Any]] = []
    fitted_candidates: dict[str, object] = {}

    for model_name, model in candidates.items():
        LOGGER.info("Training %s", model_name)
        model.fit(x_train, y_train)
        fitted_candidates[model_name] = model
        val_probability = model.predict_proba(x_val)[:, 1]
        threshold, val_report = find_best_cost_threshold(
            y_val.to_numpy(),
            val_probability,
            config.false_positive_cost,
            config.false_negative_cost,
        )
        test_probability = model.predict_proba(x_test)[:, 1]
        test_report = binary_report(
            y_test.to_numpy(),
            test_probability,
            threshold,
            config.false_positive_cost,
            config.false_negative_cost,
        )
        runs.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "validation": val_report,
                "test": test_report,
            }
        )
        LOGGER.info(
            "%s: val_cost=%s test_cost=%s test_pr_auc=%.4f",
            model_name,
            val_report["total_cost"],
            test_report["total_cost"],
            test_report["average_precision"],
        )

    selected = sorted(
        runs,
        key=lambda row: (
            row["validation"]["total_cost"],
            -row["validation"]["average_precision"],
            -row["validation"]["recall"],
        ),
    )[0]
    selected_name = selected["model"]
    selected_model = fitted_candidates[selected_name]
    selected_threshold = selected["threshold"]

    val_probability = selected_model.predict_proba(x_val)[:, 1]
    test_probability = selected_model.predict_proba(x_test)[:, 1]
    feature_importance = _compute_permutation_importance(
        selected_model,
        x_val,
        y_val,
        feature_names,
        config.permutation_repeats,
        config.permutation_max_samples,
        config.random_state,
        config.n_jobs,
    )

    plot_precision_recall(
        y_test,
        test_probability,
        figures_dir / "precision_recall_best_test.png",
        f"Precision-Recall: {selected_name}",
    )
    plot_confusion_matrix(
        selected["test"],
        figures_dir / "confusion_matrix_best_test.png",
        f"Confusion matrix: {selected_name}",
    )
    plot_feature_importance(feature_importance, figures_dir / "feature_importance_top20.png")

    LOGGER.info("Refitting final %s on the full training set", selected_name)
    final_model = build_model_candidates(config.random_state, config.n_jobs)[selected_name]
    final_model.fit(x_train_full, y_train_full)

    trained_at = datetime.now(timezone.utc).isoformat()
    medians = x_train_full.median(numeric_only=True).replace({np.nan: None}).to_dict()
    artifact = {
        "model": final_model,
        "model_name": selected_name,
        "model_version": f"{selected_name.lower()}-{trained_at}",
        "trained_at": trained_at,
        "feature_names": feature_names,
        "feature_medians": medians,
        "threshold": float(selected_threshold),
        "cost_policy": {
            "false_positive": config.false_positive_cost,
            "false_negative": config.false_negative_cost,
        },
        "feature_importance": feature_importance,
        "evaluation": {
            "selection_protocol": "Model selected by validation total cost; official UCI test is reported once for comparison.",
            "selected_run": selected,
            "runs": runs,
            "data_quality": quality_report,
        },
    }
    joblib.dump(artifact, artifacts_dir / "best_model.joblib")

    summary = {
        "trained_at": trained_at,
        "selected_model": selected_name,
        "selected_threshold": selected_threshold,
        "runs": runs,
        "data_quality": quality_report,
        "top_features": feature_importance[:20],
        "figures": {
            "class_balance": "artifacts/figures/class_balance.png",
            "missingness": "artifacts/figures/missingness_top20.png",
            "precision_recall": "artifacts/figures/precision_recall_best_test.png",
            "confusion_matrix": "artifacts/figures/confusion_matrix_best_test.png",
            "feature_importance": "artifacts/figures/feature_importance_top20.png",
        },
    }
    _write_json(artifacts_dir / "metrics.json", summary)
    _write_runs_csv(artifacts_dir / "runs.csv", runs)
    LOGGER.info("Saved final artifact to %s", artifacts_dir / "best_model.joblib")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train APS failure prediction models.")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config. Defaults to APS_CONFIG_PATH or configs/default.yaml.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args()
    summary = run_training(args.config)
    LOGGER.info(
        "Done. Selected %s at threshold %.6f",
        summary["selected_model"],
        summary["selected_threshold"],
    )


if __name__ == "__main__":
    main()
