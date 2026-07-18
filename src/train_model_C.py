"""Production training script for workout-only Model C."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sys

import joblib
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_model_C import construct_survival_labels, evaluate_split, time_splits
from src.feature_engineering import (
    PRODUCTION_FEATURES,
    build_model_c_matrix,
    prepare_model_c_frame,
)

MODELS = ROOT / "models"
MODEL_C_DIR = MODELS / "model_C"
REPORTS = ROOT / "reports"
MODEL_C_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

MODEL_VERSION = "C_workout_only_v1"


def train_rsf(x_train, y_train):
    model = RandomSurvivalForest(
        n_estimators=400,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def run() -> None:
    df = prepare_model_c_frame()
    df = construct_survival_labels(df)
    train, val, test = time_splits(df)

    x_train = build_model_c_matrix(train, PRODUCTION_FEATURES)
    feature_columns = list(x_train.columns)
    x_val = build_model_c_matrix(val, PRODUCTION_FEATURES, ref_columns=feature_columns)
    x_test = build_model_c_matrix(test, PRODUCTION_FEATURES, ref_columns=feature_columns)

    y_train = Surv.from_arrays(
        event=train["event_observed"].astype(bool).to_numpy(),
        time=train["sessions_until_next_pr"].astype(float).to_numpy(),
    )

    model = train_rsf(x_train, y_train)

    metrics = {
        "train": evaluate_split(model, x_train, train["event_observed"], train["sessions_until_next_pr"], [5, 10, 20], y_train, "train"),
        "val": evaluate_split(model, x_val, val["event_observed"], val["sessions_until_next_pr"], [5, 10, 20], y_train, "validation"),
        "test": evaluate_split(model, x_test, test["event_observed"], test["sessions_until_next_pr"], [5, 10, 20], y_train, "test"),
    }

    metadata = {
        "model_version": MODEL_VERSION,
        "features": PRODUCTION_FEATURES,
        "training_date": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "c_index": metrics["test"].get("c_index"),
            "ibs": metrics["test"].get("ibs"),
            "mae": metrics["test"].get("mae_uncensored_sessions_diagnostic"),
        },
        "preprocessing_metadata": {
            "source_file": "data/workouts.csv",
            "cutoff_date": "2023-10-01",
            "train_end": "2025-06-30",
            "val_end": "2025-12-31",
            "include_exercise_identity": True,
        },
    }

    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "features": PRODUCTION_FEATURES,
        "metadata": metadata,
    }

    joblib.dump(payload, MODEL_C_DIR / "model_C_rsf_survival.joblib")
    (MODEL_C_DIR / "model_C_metrics.json").write_text(json.dumps(metrics, indent=2))

    print("Saved:")
    print("-", MODEL_C_DIR / "model_C_rsf_survival.joblib")
    print("-", MODEL_C_DIR / "model_C_metrics.json")
    print("Model metadata:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
