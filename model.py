"""Model loading and inference helpers."""

import os
import json
import numpy as np
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocess import preprocess_single, ATTACK_CATEGORIES, MODELS_DIR

_rf_model = None
_xgb_model = None
_label_encoder = None
_metrics = None


def _load_models():
    global _rf_model, _xgb_model, _label_encoder
    if _rf_model is None:
        _rf_model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    if _xgb_model is None:
        _xgb_model = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
    if _label_encoder is None:
        _label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))


def predict_single(features: dict) -> dict:
    """Run both models on a single sample, return predictions + confidence."""
    _load_models()
    X = preprocess_single(features)

    results = {}
    for name, model in [("random_forest", _rf_model), ("xgboost", _xgb_model)]:
        proba = model.predict_proba(X)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = _label_encoder.inverse_transform([pred_idx])[0]
        results[name] = {
            "prediction": pred_label,
            "confidence": round(float(proba[pred_idx]) * 100, 2),
            "probabilities": {
                _label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2)
                for i, p in enumerate(proba)
            },
        }
    return results


def get_metrics() -> dict:
    """Return cached metrics JSON."""
    global _metrics
    if _metrics is None:
        metrics_path = os.path.join(MODELS_DIR, "metrics.json")
        with open(metrics_path, "r") as f:
            _metrics = json.load(f)
    return _metrics
