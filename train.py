"""
Train Random Forest & XGBoost classifiers on NSL-KDD dataset.
Generates models, metrics JSON, and evaluation artefacts.
"""

import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import joblib

from preprocess import get_preprocessed_data, ATTACK_CATEGORIES, MODELS_DIR, DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def train_and_evaluate():
    """Full training pipeline."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("  Intelligent IDS — Model Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Preprocess
    # ------------------------------------------------------------------
    print("\n[1/5] Loading & preprocessing NSL-KDD dataset …")
    (X_train, y_train, X_test, y_test, feature_columns), train_df, test_df = (
        get_preprocessed_data()
    )
    print(f"  Train shape: {X_train.shape}  |  Test shape: {X_test.shape}")

    n_classes = len(ATTACK_CATEGORIES)
    metrics: dict = {"attack_categories": ATTACK_CATEGORIES}

    # ------------------------------------------------------------------
    # 2. Attack distribution (for frontend chart)
    # ------------------------------------------------------------------
    print("\n[2/5] Computing attack distribution …")
    train_dist = train_df["attack_category"].value_counts().to_dict()
    test_dist = test_df["attack_category"].value_counts().to_dict()
    metrics["attack_distribution"] = {
        "train": {k: int(v) for k, v in train_dist.items()},
        "test": {k: int(v) for k, v in test_dist.items()},
    }
    print(f"  Train distribution: {train_dist}")

    # ------------------------------------------------------------------
    # 3. Train Random Forest
    # ------------------------------------------------------------------
    print("\n[3/5] Training Random Forest (n_estimators=100) …")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    print(f"  RF Accuracy: {accuracy_score(y_test, rf_preds):.4f}")

    # ------------------------------------------------------------------
    # 4. Train XGBoost
    # ------------------------------------------------------------------
    print("\n[4/5] Training XGBoost …")
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_proba = xgb.predict_proba(X_test)
    joblib.dump(xgb, os.path.join(MODELS_DIR, "xgboost.pkl"))
    print(f"  XGB Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")

    # ------------------------------------------------------------------
    # 5. Collect metrics
    # ------------------------------------------------------------------
    print("\n[5/5] Generating evaluation metrics …")

    for name, preds, proba in [
        ("random_forest", rf_preds, rf_proba),
        ("xgboost", xgb_preds, xgb_proba),
    ]:
        acc = accuracy_score(y_test, preds)
        report = classification_report(
            y_test, preds, target_names=ATTACK_CATEGORIES, output_dict=True
        )
        cm = confusion_matrix(y_test, preds).tolist()

        # ROC curve (one-vs-rest)
        y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
        roc_data = {}
        for i, cat in enumerate(ATTACK_CATEGORIES):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_data[cat] = {
                "fpr": fpr[::max(1, len(fpr) // 100)].tolist(),
                "tpr": tpr[::max(1, len(tpr) // 100)].tolist(),
                "auc": round(roc_auc, 4),
            }

        metrics[name] = {
            "accuracy": round(acc, 4),
            "classification_report": {
                k: {kk: round(vv, 4) for kk, vv in v.items()} if isinstance(v, dict) else round(v, 4)
                for k, v in report.items()
            },
            "confusion_matrix": cm,
            "roc": roc_data,
        }

    # Feature importance (top 30)
    fi_rf = rf.feature_importances_
    fi_xgb = xgb.feature_importances_
    top_n = 30
    rf_top_idx = np.argsort(fi_rf)[::-1][:top_n]
    xgb_top_idx = np.argsort(fi_xgb)[::-1][:top_n]

    metrics["feature_importance"] = {
        "random_forest": [
            {"feature": feature_columns[i], "importance": round(float(fi_rf[i]), 6)}
            for i in rf_top_idx
        ],
        "xgboost": [
            {"feature": feature_columns[i], "importance": round(float(fi_xgb[i]), 6)}
            for i in xgb_top_idx
        ],
    }

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Models saved to:  {MODELS_DIR}")
    print(f"  Metrics saved to: {METRICS_PATH}")
    print(f"  RF Accuracy:  {metrics['random_forest']['accuracy']}")
    print(f"  XGB Accuracy: {metrics['xgboost']['accuracy']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    train_and_evaluate()
