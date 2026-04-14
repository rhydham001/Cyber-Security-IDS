"""API route handlers."""

import io
import csv
from fastapi import APIRouter, UploadFile, File, HTTPException
from .schemas import PredictRequest, PredictResponse
from .model import predict_single, get_metrics

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """Predict the attack category for a single network connection."""
    features = req.model_dump()
    results = predict_single(features)
    return results


@router.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    """
    Accept a CSV file with raw NSL-KDD features (header row required).
    Returns a list of predictions for each row.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()
    text = content.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))

    results = []
    for idx, row in enumerate(reader):
        # Convert numeric strings to float/int
        parsed = {}
        for k, v in row.items():
            k = k.strip()
            try:
                parsed[k] = float(v)
            except (ValueError, TypeError):
                parsed[k] = v.strip() if isinstance(v, str) else v
        pred = predict_single(parsed)
        results.append({"row_index": idx, **pred})

    return {"predictions": results, "total": len(results)}


@router.get("/metrics")
async def metrics():
    """Return model performance metrics (accuracy, confusion matrix, ROC, etc.)."""
    return get_metrics()


@router.get("/feature-importance")
async def feature_importance():
    """Return top feature importances for both models."""
    m = get_metrics()
    return m.get("feature_importance", {})


@router.get("/attack-distribution")
async def attack_distribution():
    """Return the distribution of attack types in train/test sets."""
    m = get_metrics()
    return m.get("attack_distribution", {})
