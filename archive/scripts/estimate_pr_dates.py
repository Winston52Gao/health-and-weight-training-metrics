from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["exercise_title", "date"]).reset_index(drop=True)

    grp = df.groupby("exercise_title")
    df["training_age_sessions"] = grp.cumcount() + 1
    first_date = grp["date"].transform("min")
    df["training_age_days"] = (df["date"] - first_date).dt.days

    df = df.sort_values(["exercise_title", "date"]).set_index("date")
    out = []
    for _, g in df.groupby("exercise_title", sort=False):
        g = g.sort_index()
        g["days_since_last_workout"] = g.index.to_series().diff().dt.days.fillna(9999)
        out.append(g.reset_index())
    df = pd.concat(out, ignore_index=True)

    if "pr_gap_percent" not in df.columns and {"rolling_best_prev", "best_est_1RM"}.issubset(df.columns):
        denom = df["rolling_best_prev"].replace(0, np.nan)
        df["pr_gap_percent"] = (df["rolling_best_prev"] - df["best_est_1RM"]) / denom

    if "volume_ratio_28_56" not in df.columns and {"volume_28d_avg", "volume_56d_avg"}.issubset(df.columns):
        denom = df["volume_56d_avg"].replace(0, np.nan)
        df["volume_ratio_28_56"] = df["volume_28d_avg"] / denom

    for col in [
        "relative_strength", "pr_gap_percent", "rolling_best_prev", "best_est_1RM",
        "volume_28d_avg", "volume_56d_avg", "volume_ratio_28_56",
        "sessions_since_last_pr", "days_since_last_pr", "training_age_sessions",
        "training_age_days", "sleep_minutes", "sleep_7d_avg", "resting_hr",
        "hr_7d_avg", "hr_baseline_z", "steps_7d_avg", "model_a_score", "model_b_score",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def build_rsf_matrix(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    num_cols = [c for c in feature_columns if not c.startswith("ex_")]
    X_num = df[num_cols].copy() if num_cols else pd.DataFrame(index=df.index)
    X_ex = pd.get_dummies(df["exercise_title"].fillna("unknown"), prefix="ex", dtype=float)
    X = pd.concat([X_num, X_ex], axis=1)
    return X.reindex(columns=feature_columns, fill_value=0.0)


def expected_sessions_and_interval(rsf, X_one: pd.DataFrame) -> tuple[float, list[float]]:
    fn = rsf.predict_survival_function(X_one, return_array=False)[0]
    times = rsf.unique_times_
    s = fn(times)
    delta = np.diff(np.r_[0.0, times])
    expected = float(np.sum(s * delta))

    def qtime(target_surv: float) -> float:
        idx = np.where(s <= target_surv)[0]
        if len(idx) == 0:
            return float(times[-1])
        return float(times[idx[0]])

    interval = [qtime(0.75), qtime(0.25)]
    return expected, interval


def main() -> None:
    df = pd.read_csv(DATA / "processed_merged.csv", parse_dates=["date"])
    df = prepare_frame(df)

    model_a = joblib.load(MODELS / "model_A_workout.joblib")
    model_b = joblib.load(MODELS / "model_B_recovery.joblib")
    rsf_bundle = joblib.load(MODELS / "model_C_rsf_survival.joblib")
    rsf = rsf_bundle["model"]
    rsf_feature_columns = rsf_bundle["feature_columns"]

    featsA = list(model_a.feature_names_in_)
    featsB = list(model_b.feature_names_in_)

    # Create model A/B scores for the latest row per exercise
    X_a = df[featsA].fillna(0.0)
    X_b = df[featsB].fillna(0.0)
    df["model_a_score"] = model_a.predict_proba(X_a)[:, 1]
    df["model_b_score"] = model_b.predict_proba(X_b)[:, 1]

    exercise_aliases = {
        "Preacher curl": ["preacher curl (barbell)", "preacher hammer curl ", "preacher curl"],
        "Incline dumbbell press": ["incline bench press (dumbbell)", "incline dumbbell press", "incline bench press"],
        "Single arm tricep pushdown": ["single arm triceps pushdown (cable)", "single arm tricep pushdown", "tricep pushdown single arm", "triceps pushdown"],
        "Jefferson curl": ["jefferson curl"],
        "Overhead press": ["overhead press (barbell)", "seated overhead press (barbell)", "overhead press"],
        "Pull up": ["pull up", "pull up (assisted)", "pull up (band)"],
    }

    results = []
    lower_names = df["exercise_title"].fillna("").str.lower()
    for label, aliases in exercise_aliases.items():
        latest = None
        for alias in aliases:
            cand = df[lower_names == alias.lower()].sort_values("date")
            if not cand.empty:
                latest = cand.iloc[[-1]].copy()
                break
        if latest is None:
            print(f"{label}: no exercise match in processed data")
            continue

        X_one = build_rsf_matrix(latest, rsf_feature_columns)
        expected_sessions, interval = expected_sessions_and_interval(rsf, X_one)

        item = {
            "exercise": str(latest["exercise_title"].iloc[0]),
            "expected_sessions_to_PR": float(expected_sessions),
            "confidence_interval": [float(interval[0]), float(interval[1])],
        }
        results.append(item)

        print(json.dumps(item, indent=2))

    out_path = MODELS / "pr_timing_estimates.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf8")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
