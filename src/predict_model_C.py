"""Production inference script for workout-only Model C."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_model_C import expected_sessions_from_survival, first_session_meeting_threshold
from src.feature_engineering import (
    PRODUCTION_FEATURES,
    build_model_c_matrix,
    prepare_model_c_frame,
)

MODELS = ROOT / "models"
MODEL_C_DIR = MODELS / "model_C"
OUTPUTS = ROOT / "outputs"
CURVE_POINTS = [1, 3, 5, 8, 10, 15, 20, 25, 30]


def run() -> None:
    payload = joblib.load(MODEL_C_DIR / "model_C_rsf_survival.joblib")
    model = payload["model"]
    features = payload["features"]
    feature_columns = payload["feature_columns"]

    df = prepare_model_c_frame()
    latest_rows = df.sort_values("date").groupby("exercise_title", as_index=False).tail(1).copy()
    x_latest = build_model_c_matrix(latest_rows, features, ref_columns=feature_columns)
    expected_values = expected_sessions_from_survival(model, x_latest)

    surv_fns = model.predict_survival_function(x_latest, return_array=False)
    forecasts = []
    for index, (_, row) in enumerate(latest_rows.reset_index(drop=True).iterrows()):
        fn = surv_fns[index]
        curve = []
        for session_horizon in CURVE_POINTS:
            curve.append(
                {
                    "sessions_ahead": int(session_horizon),
                    "probability_of_pr": float(1.0 - float(fn(session_horizon))),
                }
            )

        forecasts.append(
            {
                "exercise": str(row["exercise_title"]),
                "current_best_est_1RM": float(row["best_est_1RM"]) if pd.notnull(row.get("best_est_1RM")) else None,
                "expected_sessions_until_pr": float(expected_values[index]),
                "probability_of_pr_within_5_sessions": float(1.0 - float(fn(5))),
                "probability_of_pr_within_10_sessions": float(1.0 - float(fn(10))),
                "probability_of_pr_within_20_sessions": float(1.0 - float(fn(20))),
                "50_percent_probability_window": first_session_meeting_threshold(curve, 0.50),
                "80_percent_probability_window": first_session_meeting_threshold(curve, 0.80),
                "probability_curve": curve,
            }
        )

    forecasts = sorted(
        forecasts,
        key=lambda item: -1.0
        if item["probability_curve"] is None
        else -max([p["probability_of_pr"] for p in item["probability_curve"] if p["sessions_ahead"] == 10] or [0.0]),
    )

    output = {
        "model_version": payload.get("metadata", {}).get("model_version", "unknown"),
        "forecasts": forecasts,
    }
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "pr_forecast_predictions.json").write_text(json.dumps(output, indent=2))

    print("Saved:", OUTPUTS / "pr_forecast_predictions.json")
    if forecasts:
        print("Example:")
        print(json.dumps(forecasts[0], indent=2))


if __name__ == "__main__":
    run()
