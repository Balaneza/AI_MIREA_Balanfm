import json
from pathlib import Path

import pytest

from src.aps_failure.data import build_input_frame
from src.aps_failure.service import _feature_explanations, _risk_level, create_app


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_service_app_loads_model_artifact():
    app = create_app()

    assert app.title == "APS Failure at Scania Trucks API"
    assert app.state.artifact is not None
    assert len(app.state.artifact["feature_names"]) == 170


def test_model_scores_demo_payload_used_by_api():
    artifact_path = PROJECT_ROOT / "artifacts" / "best_model.joblib"
    if not artifact_path.exists():
        pytest.skip("Model artifact is not trained yet")

    app = create_app()
    artifact = app.state.artifact
    payload = json.loads((PROJECT_ROOT / "data" / "demo_payload.json").read_text(encoding="utf-8"))
    x, missing_features, unknown_features = build_input_frame(payload["features"], artifact["feature_names"])
    probability = float(artifact["model"].predict_proba(x)[0, 1])
    explanations = _feature_explanations(
        payload["features"],
        artifact["feature_importance"],
        artifact["feature_medians"],
    )

    assert 0.0 <= probability <= 1.0
    assert _risk_level(probability, artifact["threshold"]) in {"low", "medium", "high"}
    assert len(missing_features) == 0
    assert unknown_features == []
    assert explanations
