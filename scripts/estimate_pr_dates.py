from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

processed = pd.read_csv(DATA / "processed_merged.csv", parse_dates=["date"])
processed = processed.copy()
processed["date"] = pd.to_datetime(processed["date"]).dt.normalize()
processed = processed.sort_values(["exercise_title", "date"]).reset_index(drop=True)
processed["exercise_title"] = processed["exercise_title"].fillna("").astype(str)
processed["exercise_norm"] = processed["exercise_title"].str.lower()

# ensure numeric columns used by the models
for col in [
    "best_est_1RM", "total_volume", "relative_strength", "rolling_best_prev",
    "pr_gap", "distance_to_personal_best", "days_since_last_pr",
    "sessions_since_last_pr", "pr_freq_90d", "training_age_sessions",
    "training_age_days", "days_since_last_workout", "sleep_minutes",
    "sleep_7d_avg", "sleep_dev_from_14d", "resting_hr", "hr_7d_avg",
    "hr_baseline_z", "volume_28d_avg", "volume_56d_avg", "volume_28d_ratio",
    "volume_56d_ratio", "volume_28d_z", "volume_56d_z"
]:
    if col in processed.columns:
        processed[col] = pd.to_numeric(processed[col], errors="coerce").fillna(0)

# derive training-age and workout-gap features in the same spirit as training
processed["training_age_sessions"] = 0
processed["training_age_days"] = 0
for _, g in processed.groupby("exercise_title", sort=False):
    g = g.sort_values("date")
    g["training_age_sessions"] = np.arange(1, len(g) + 1)
    first_date = g["date"].min()
    g["training_age_days"] = (g["date"] - first_date).dt.days
    processed.loc[g.index, ["training_age_sessions", "training_age_days"]] = g[["training_age_sessions", "training_age_days"]].values

processed["days_since_last_workout"] = 9999
for _, g in processed.groupby("exercise_title", sort=False):
    g = g.sort_values("date")
    gaps = g["date"].diff().dt.days.fillna(9999)
    processed.loc[g.index, "days_since_last_workout"] = gaps.values

# create missing columns expected by model A
if "pr_gap" in processed.columns and "rolling_best_prev" in processed.columns:
    processed["pr_gap"] = processed["best_est_1RM"] - processed["rolling_best_prev"]
if "distance_to_personal_best" in processed.columns and "rolling_best_prev" in processed.columns:
    processed["distance_to_personal_best"] = np.where(
        processed["rolling_best_prev"] > 0,
        (processed["rolling_best_prev"] - processed["best_est_1RM"]) / processed["rolling_best_prev"],
        np.nan,
    )

for c in ["days_since_last_pr", "sessions_since_last_pr", "pr_freq_90d"]:
    if c not in processed.columns:
        processed[c] = 0.0

model_a = joblib.load(MODELS / "model_A_workout.joblib")
model_b = joblib.load(MODELS / "model_B_recovery.joblib")
model_c = joblib.load(MODELS / "model_C_stacked.joblib")

feature_cols_a = list(model_a.feature_names_in_)
feature_cols_b = list(model_b.feature_names_in_)
feature_cols_c = list(model_c.feature_names_in_)

exercise_aliases = {
    "Preacher curl": ["preacher curl (barbell)", "preacher hammer curl ", "preacher curl"],
    "Incline dumbbell press": ["incline bench press (dumbbell)", "incline dumbbell press", "incline bench press"],
    "Single arm tricep pushdown": ["single arm triceps pushdown (cable)", "single arm tricep pushdown", "tricep pushdown single arm", "triceps pushdown"],
    "Jefferson curl": ["jefferson curl"],
    "Overhead press": ["overhead press (barbell)", "seated overhead press (barbell)", "overhead press"],
    "Pull up": ["pull up", "pull up (assisted)", "pull up (band)"],
}

for label, aliases in exercise_aliases.items():
    match_df = None
    for alias in aliases:
        cand = processed[processed["exercise_norm"] == alias.lower()]
        if not cand.empty:
            match_df = cand
            break
    if match_df is None or match_df.empty:
        print(f"{label}: no exercise match in processed data")
        continue

    match_df = match_df.sort_values("date")
    latest = match_df.iloc[-1]
    avg_gap = int(match_df["date"].diff().dropna().dt.days.median()) if len(match_df) >= 2 else 14

    row_a = pd.DataFrame([{c: latest.get(c, 0.0) for c in feature_cols_a}]).astype(float)
    row_b = pd.DataFrame([{c: latest.get(c, 0.0) for c in feature_cols_b}]).astype(float)
    row_c = pd.DataFrame([{c: latest.get(c, 0.0) for c in feature_cols_c if c not in ["pr_prob_workout", "pr_prob_recovery"]}]).astype(float)

    p_a = float(model_a.predict_proba(row_a)[0, 1])
    p_b = float(model_b.predict_proba(row_b)[0, 1])
    row_c["pr_prob_workout"] = p_a
    row_c["pr_prob_recovery"] = p_b
    p_c = float(model_c.predict_proba(row_c[feature_cols_c])[0, 1])

    prob = max(min(p_c, 0.99), 0.01)
    sessions_to_pr = max(1, int(np.ceil(1.0 / prob)))
    days_to_pr = max(14, sessions_to_pr * max(avg_gap, 7))
    est_date = latest["date"] + pd.Timedelta(days=days_to_pr)

    print(f"{label}")
    print(f"  matched exercise: {latest['exercise_title']}")
    print(f"  latest session date: {latest['date'].date()}")
    print(f"  latest best estimated 1RM: {latest['best_est_1RM']:.2f}")
    print(f"  stacked model PR probability: {prob:.3f}")
    print(f"  rough estimate: about {sessions_to_pr} more sessions (~{days_to_pr} days) until next PR")
    print(f"  estimated PR date: {est_date.date()}")
    print()
