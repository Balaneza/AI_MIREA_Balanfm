from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    random_state: int
    train_path: Path
    test_path: Path
    description_path: Path
    target_column: str
    positive_label: str
    negative_label: str
    validation_size: float
    false_positive_cost: int
    false_negative_cost: int
    n_jobs: int
    permutation_repeats: int
    permutation_max_samples: int
    default_model_path: Path
    host: str
    port: int


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config_path = Path(
        path
        or os.getenv("APS_CONFIG_PATH")
        or PROJECT_ROOT / "configs" / "default.yaml"
    )
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    project = raw["project"]
    data = raw["data"]
    costs = raw["costs"]
    training = raw["training"]
    service = raw["service"]

    return ProjectConfig(
        name=str(project["name"]),
        random_state=int(project["random_state"]),
        train_path=_resolve_project_path(data["train_path"]),
        test_path=_resolve_project_path(data["test_path"]),
        description_path=_resolve_project_path(data["description_path"]),
        target_column=str(data["target_column"]),
        positive_label=str(data["positive_label"]),
        negative_label=str(data["negative_label"]),
        validation_size=float(data["validation_size"]),
        false_positive_cost=int(costs["false_positive"]),
        false_negative_cost=int(costs["false_negative"]),
        n_jobs=int(training["n_jobs"]),
        permutation_repeats=int(training["permutation_repeats"]),
        permutation_max_samples=int(training["permutation_max_samples"]),
        default_model_path=_resolve_project_path(
            os.getenv("APS_MODEL_PATH") or service["default_model_path"]
        ),
        host=os.getenv("APS_SERVICE_HOST", str(service["host"])),
        port=int(os.getenv("APS_SERVICE_PORT", service["port"])),
    )
