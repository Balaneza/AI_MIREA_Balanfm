from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .config import load_config
from .data import build_input_frame, normalize_sensor_value


LOGGER = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    request_id: str | None = Field(default=None)
    features: dict[str, int | float | str | None] = Field(
        ...,
        description="Dictionary with APS sensor features, for example aa_000, ac_000, ee_005.",
    )


class FeatureExplanation(BaseModel):
    feature: str
    value: float | None
    training_median: float | None
    importance: float
    direction_vs_median: str
    missing_in_request: bool


class PredictionResponse(BaseModel):
    request_id: str | None
    failure_probability: float
    predicted_class: str
    risk_level: str
    decision_threshold: float
    cost_policy: dict[str, int]
    model_name: str
    model_version: str
    missing_feature_count: int
    unknown_features: list[str]
    top_features: list[FeatureExplanation]


def _load_artifact(model_path: Path) -> dict[str, Any] | None:
    if not model_path.exists():
        LOGGER.warning("Model artifact not found at %s", model_path)
        return None
    LOGGER.info("Loading model artifact from %s", model_path)
    return joblib.load(model_path)


def _risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "high"
    if probability >= max(threshold * 0.5, 0.05):
        return "medium"
    return "low"


def _feature_explanations(
    raw_features: dict[str, Any],
    feature_importance: list[dict[str, Any]],
    medians: dict[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    ranked = []
    for item in feature_importance[:40]:
        feature = item["feature"]
        importance = float(item.get("importance", 0.0))
        value = normalize_sensor_value(raw_features.get(feature))
        median_value = medians.get(feature)
        median = None if median_value is None else float(median_value)
        missing = feature not in raw_features or not math.isfinite(value)

        if missing or median is None or not math.isfinite(median):
            direction = "missing_or_unknown"
            local_weight = importance * 0.25
            clean_value = None
        else:
            clean_value = float(value)
            if clean_value > median:
                direction = "above_training_median"
            elif clean_value < median:
                direction = "below_training_median"
            else:
                direction = "near_training_median"
            local_weight = importance * abs(np.log1p(max(clean_value, 0.0)) - np.log1p(max(median, 0.0)))

        ranked.append(
            {
                "feature": feature,
                "value": clean_value,
                "training_median": median,
                "importance": importance,
                "direction_vs_median": direction,
                "missing_in_request": bool(missing),
                "_rank": float(local_weight),
            }
        )

    ranked.sort(key=lambda row: (row["_rank"], row["importance"]), reverse=True)
    if not ranked:
        return []
    if all(row["_rank"] == 0 for row in ranked):
        ranked.sort(key=lambda row: row["importance"], reverse=True)
    return [{key: value for key, value in row.items() if key != "_rank"} for row in ranked[:limit]]


def create_app() -> FastAPI:
    logging.basicConfig(
        level=os.getenv("APS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config()
    artifact = _load_artifact(config.default_model_path)

    app = FastAPI(
        title="APS Failure at Scania Trucks API",
        description="Predicts whether a truck failure is related to the Air Pressure System.",
        version="1.0.0",
    )
    app.state.artifact = artifact
    app.state.started_at = time.time()
    app.state.metrics = {
        "requests_total": 0,
        "predictions_total": 0,
        "errors_total": 0,
        "prediction_latency_seconds_sum": 0.0,
    }

    @app.get("/health")
    def health() -> dict[str, Any]:
        app.state.metrics["requests_total"] += 1
        artifact_loaded = app.state.artifact is not None
        LOGGER.info("GET /health model_loaded=%s", artifact_loaded)
        return {
            "status": "ok" if artifact_loaded else "degraded",
            "model_loaded": artifact_loaded,
            "model_path": str(config.default_model_path),
            "uptime_seconds": round(time.time() - app.state.started_at, 3),
            "feature_count": len(app.state.artifact["feature_names"]) if artifact_loaded else 0,
        }

    @app.get("/metrics")
    def metrics() -> Response:
        app.state.metrics["requests_total"] += 1
        LOGGER.info("GET /metrics")
        lines = []
        for key, value in app.state.metrics.items():
            lines.append(f"aps_{key} {value}")
        return Response("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")

    @app.post("/predict-failure", response_model=PredictionResponse)
    def predict_failure(payload: PredictionRequest) -> PredictionResponse:
        app.state.metrics["requests_total"] += 1
        if app.state.artifact is None:
            app.state.metrics["errors_total"] += 1
            raise HTTPException(status_code=503, detail="Model artifact is not available. Run python3 -m src.train first.")

        started = time.perf_counter()
        artifact = app.state.artifact
        feature_names = artifact["feature_names"]
        x, missing_features, unknown_features = build_input_frame(payload.features, feature_names)
        probability = float(artifact["model"].predict_proba(x)[0, 1])
        threshold = float(artifact["threshold"])
        predicted_positive = probability >= threshold
        app.state.metrics["predictions_total"] += 1
        app.state.metrics["prediction_latency_seconds_sum"] += time.perf_counter() - started
        LOGGER.info(
            "POST /predict-failure request_id=%s probability=%.6f predicted=%s latency=%.4fs",
            payload.request_id,
            probability,
            "pos" if predicted_positive else "neg",
            time.perf_counter() - started,
        )

        return PredictionResponse(
            request_id=payload.request_id,
            failure_probability=round(probability, 6),
            predicted_class="pos" if predicted_positive else "neg",
            risk_level=_risk_level(probability, threshold),
            decision_threshold=round(threshold, 6),
            cost_policy=artifact["cost_policy"],
            model_name=artifact["model_name"],
            model_version=artifact["model_version"],
            missing_feature_count=len(missing_features),
            unknown_features=unknown_features,
            top_features=_feature_explanations(
                payload.features,
                artifact.get("feature_importance", []),
                artifact.get("feature_medians", {}),
            ),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict_alias(payload: PredictionRequest) -> PredictionResponse:
        return predict_failure(payload)

    return app
