"""Workout-only feature engineering for production Model C.

This module owns the full feature pipeline used by both training and inference:
raw workouts CSV -> session-level aggregation -> progression features -> model
matrix construction.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_WORKOUTS_PATH = DATA / "workouts.csv"
CLEANED_WORKOUTS_PATH = DATA / "cleaned workouts.csv"

WORKOUT_FEATURES = [
    "relative_strength",
    "pr_gap_percent",
    "rolling_best_prev",
    "best_est_1RM",
    "volume_28d_avg",
    "volume_56d_avg",
    "volume_ratio_28_56",
    "sessions_since_last_pr",
    "days_since_last_pr",
    "pr_freq_90d",
    "training_age_sessions",
    "training_age_days",
]

FITBIT_FEATURES = [
    "sleep_minutes",
    "sleep_7d_avg",
    "resting_hr",
    "hr_7d_avg",
    "hr_baseline_z",
    "steps_7d_avg",
]

PRODUCTION_FEATURES = WORKOUT_FEATURES
DEFAULT_CUTOFF_DATE = "2023-10-01"
PULL_UP_BODYWEIGHT_KG = 205.0 * 0.45359237
MIN_EXERCISE_SESSIONS = 10


def parse_start_date(value: str) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    match = re.search(r"([A-Za-z]+\s+\d{1,2},\s*\d{4})", str(value))
    if match:
        try:
            return pd.to_datetime(match.group(1)).normalize()
        except Exception:
            return pd.NaT
    try:
        return pd.to_datetime(value).normalize()
    except Exception:
        return pd.NaT


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load raw workout rows from CSV."""
    source = Path(path) if path is not None else RAW_WORKOUTS_PATH
    return pd.read_csv(source)


def save_cleaned_workouts(workouts: pd.DataFrame, path: str | Path | None = None) -> Path:
    destination = Path(path) if path is not None else CLEANED_WORKOUTS_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    workouts.to_csv(destination, index=False)
    return destination


def standardize_workout_dates(workouts: pd.DataFrame) -> pd.DataFrame:
    workouts = workouts.copy()
    if "exercise_title" not in workouts.columns and "exercise" in workouts.columns:
        workouts = workouts.rename(columns={"exercise": "exercise_title"})
    workouts["date"] = pd.to_datetime(workouts["start_time"].astype(str).apply(parse_start_date)).dt.normalize()
    return workouts


def filter_workouts(workouts: pd.DataFrame, cutoff_date: str = DEFAULT_CUTOFF_DATE) -> pd.DataFrame:
    workouts = workouts.copy()
    if "exercise_title" not in workouts.columns and "exercise" in workouts.columns:
        workouts = workouts.rename(columns={"exercise": "exercise_title"})

    exercise_text = workouts.get("exercise_title", pd.Series([""] * len(workouts))).astype(str).str.lower()
    cardio_mask = exercise_text.str.contains("treadmill") | exercise_text.str.contains(r"air\s*-?bike")
    if cardio_mask.any():
        workouts = workouts.loc[~cardio_mask].copy()

    workouts["date"] = pd.to_datetime(workouts["date"]).dt.normalize()
    cutoff = pd.to_datetime(cutoff_date).normalize()
    workouts = workouts.loc[workouts["date"] >= cutoff].copy()

    weight_source = None
    for candidate in ["weight_kg", "weight", "kg", "weight_lb"]:
        if candidate in workouts.columns:
            weight_source = candidate
            break

    if weight_source is None:
        raw_weight_kg = pd.Series(0.0, index=workouts.index, dtype=float)
    elif weight_source == "weight_lb":
        raw_weight_kg = pd.to_numeric(workouts[weight_source], errors="coerce") * 0.45359237
    else:
        raw_weight_kg = pd.to_numeric(workouts[weight_source], errors="coerce")

    workouts["weight_kg"] = raw_weight_kg.fillna(0.0)

    duration_seconds = pd.to_numeric(workouts.get("duration_seconds", 0), errors="coerce").fillna(0.0)
    distance_km = pd.to_numeric(workouts.get("distance_km", 0), errors="coerce").fillna(0.0)
    reps = pd.to_numeric(workouts.get("reps", 0), errors="coerce").fillna(0.0)

    is_pull_up = exercise_text.str.contains(r"\bpull\s*-?up\b", regex=True)
    is_assisted_pull_up = is_pull_up & exercise_text.str.contains(r"assist|band", regex=True)
    pure_time_or_distance = (duration_seconds > 0) | (distance_km > 0)
    pure_reps_bodyweight = (reps > 0) & (workouts["weight_kg"] <= 0)

    keep_mask = (~pure_time_or_distance) & ((workouts["weight_kg"] > 0) | is_pull_up)
    workouts = workouts.loc[keep_mask].copy()

    pull_up_rows = is_pull_up.loc[workouts.index]
    if pull_up_rows.any():
        assisted_pull_up_rows = is_assisted_pull_up.loc[workouts.index]
        workouts.loc[pull_up_rows & (workouts["weight_kg"] <= 0), "weight_kg"] = PULL_UP_BODYWEIGHT_KG
        assisted_loads = raw_weight_kg.loc[workouts.index].loc[assisted_pull_up_rows]
        effective_assisted_weight = PULL_UP_BODYWEIGHT_KG - assisted_loads.fillna(0.0)
        workouts.loc[assisted_pull_up_rows, "weight_kg"] = effective_assisted_weight.clip(lower=0.0)

    # Drop other bodyweight / reps-only movements while preserving pull-ups.
    bodyweight_non_pull_up = pure_reps_bodyweight.loc[workouts.index] & ~pull_up_rows
    if bodyweight_non_pull_up.any():
        workouts = workouts.loc[~bodyweight_non_pull_up].copy()

    session_counts = workouts.groupby("exercise_title")["date"].nunique()
    valid_exercises = session_counts.loc[session_counts >= MIN_EXERCISE_SESSIONS].index
    workouts = workouts.loc[workouts["exercise_title"].isin(valid_exercises)].copy()
    return workouts


def aggregate_workouts(workouts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate set-level workout rows into per-day/per-exercise sessions."""
    workouts = workouts.copy()

    weight_col = None
    for candidate in ["weight", "weight_kg", "weight_lb", "kg"]:
        if candidate in workouts.columns:
            weight_col = candidate
            break

    if weight_col is None:
        workouts["weight"] = 0.0
    else:
        workouts["weight"] = pd.to_numeric(workouts[weight_col], errors="coerce").fillna(0.0)

    workouts["reps"] = pd.to_numeric(workouts.get("reps", 0), errors="coerce").fillna(0.0)
    workouts["volume"] = workouts["weight"] * workouts["reps"]
    workouts["est_1RM_set"] = workouts["weight"] * (1.0 + workouts["reps"] / 30.0)

    if "sets" in workouts.columns:
        workouts["sets"] = pd.to_numeric(workouts["sets"], errors="coerce").fillna(0.0)

    grouped = workouts.groupby(["date", "exercise_title"], dropna=False)
    session = grouped.agg(
        total_volume=("volume", "sum"),
        avg_weight=("weight", "mean"),
        max_weight=("weight", "max"),
        total_reps=("reps", "sum"),
        best_est_1RM=("est_1RM_set", "max"),
    ).reset_index()

    if "sets" in workouts.columns:
        total_sets = grouped["sets"].sum().reset_index(name="total_sets")
    else:
        total_sets = grouped.size().reset_index(name="total_sets")
    session = session.merge(total_sets, on=["date", "exercise_title"], how="left")

    session["date"] = pd.to_datetime(session["date"]).dt.normalize()
    session = session.sort_values(["exercise_title", "date"]).reset_index(drop=True)
    return session


def add_training_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["exercise_title", "date"]).reset_index(drop=True)
    grouped = df.groupby("exercise_title")
    df["training_age_sessions"] = grouped.cumcount() + 1
    first_date = grouped["date"].transform("min")
    df["training_age_days"] = (
        pd.to_datetime(df["date"]).dt.normalize() - pd.to_datetime(first_date).dt.normalize()
    ).dt.days
    return df


def add_recent_training_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["exercise_title", "date"]).set_index("date")
    out_frames = []
    for _, group in df.groupby("exercise_title"):
        group = group.sort_index()
        group["days_since_last_workout"] = group.index.to_series().diff().dt.days.fillna(9999)
        out_frames.append(group.reset_index())
    return (
        pd.concat(out_frames, ignore_index=True)
        .sort_values(["exercise_title", "date"])
        .reset_index(drop=True)
    )


def _per_exercise_time_features(session_frame: pd.DataFrame) -> pd.DataFrame:
    out_frames = []
    for _, group in session_frame.sort_values(["exercise_title", "date"]).groupby("exercise_title", sort=False):
        group = group.sort_values("date").set_index("date")

        group["rolling_best_prev"] = group["best_est_1RM"].cummax().shift(1)
        group["relative_strength"] = np.where(
            group["rolling_best_prev"] > 0,
            group["best_est_1RM"] / group["rolling_best_prev"],
            np.nan,
        )
        group["is_pr"] = (group["best_est_1RM"] > group["rolling_best_prev"]).astype(int)

        group["volume_28d_avg"] = group["total_volume"].rolling("28D", closed="left").mean()
        group["volume_56d_avg"] = group["total_volume"].rolling("56D", closed="left").mean()
        group["volume_ratio_28_56"] = np.where(
            group["volume_56d_avg"] > 0,
            group["volume_28d_avg"] / group["volume_56d_avg"],
            np.nan,
        )

        last_pr_before = pd.Series(group.index.where(group["is_pr"] == 1), index=group.index).ffill().shift(1)
        group["days_since_last_pr"] = (group.index.to_series() - last_pr_before).dt.days
        group["pr_freq_90d"] = group["is_pr"].rolling("90D", closed="left").sum()

        pr_group_id = group["is_pr"].cumsum()
        group["sessions_since_last_pr"] = group.groupby(pr_group_id).cumcount().shift(1)

        group["pr_gap_percent"] = np.where(
            group["rolling_best_prev"] > 0,
            (group["rolling_best_prev"] - group["best_est_1RM"]) / group["rolling_best_prev"],
            np.nan,
        )

        out_frames.append(group.reset_index())

    return pd.concat(out_frames, axis=0).sort_values(["exercise_title", "date"]).reset_index(drop=True)


def ensure_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the production feature columns exist and are numeric-friendly."""
    df = df.copy()

    if "pr_gap_percent" not in df.columns and {"rolling_best_prev", "best_est_1RM"}.issubset(df.columns):
        denom = df["rolling_best_prev"].replace(0, np.nan)
        df["pr_gap_percent"] = (df["rolling_best_prev"] - df["best_est_1RM"]) / denom
    if "pr_gap_percent" not in df.columns:
        df["pr_gap_percent"] = 0.0

    if "volume_ratio_28_56" not in df.columns and {"volume_28d_avg", "volume_56d_avg"}.issubset(df.columns):
        denom = df["volume_56d_avg"].replace(0, np.nan)
        df["volume_ratio_28_56"] = df["volume_28d_avg"] / denom
    if "volume_ratio_28_56" not in df.columns:
        df["volume_ratio_28_56"] = 0.0

    for col in [
        "relative_strength",
        "rolling_best_prev",
        "best_est_1RM",
        "volume_28d_avg",
        "volume_56d_avg",
        "sessions_since_last_pr",
        "days_since_last_pr",
        "pr_freq_90d",
        "training_age_sessions",
        "training_age_days",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    if "is_pr" not in df.columns:
        if {"rolling_best_prev", "best_est_1RM"}.issubset(df.columns):
            df["is_pr"] = (df["best_est_1RM"] > df["rolling_best_prev"]).astype(int)
        else:
            df["is_pr"] = 0

    return df


def prepare_model_c_frame(
    raw_workouts: pd.DataFrame | None = None,
    cutoff_date: str = DEFAULT_CUTOFF_DATE,
) -> pd.DataFrame:
    """Create the full session-level production feature frame from raw workout CSV rows."""
    workouts = load_data() if raw_workouts is None else raw_workouts.copy()
    workouts = standardize_workout_dates(workouts)
    workouts = filter_workouts(workouts, cutoff_date=cutoff_date)
    if raw_workouts is None:
        save_cleaned_workouts(workouts)
    session_frame = aggregate_workouts(workouts)
    session_frame = _per_exercise_time_features(session_frame)
    session_frame = add_training_age(session_frame)
    session_frame = add_recent_training_context(session_frame)
    session_frame = ensure_model_features(session_frame)
    return session_frame


def build_model_c_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    ref_columns: list[str] | None = None,
    include_exercise_identity: bool = True,
) -> pd.DataFrame:
    """Build the Model C design matrix with stable column order."""
    x_num = df[feature_cols].copy().fillna(0.0)
    if include_exercise_identity:
        x_ex = pd.get_dummies(df["exercise_title"].fillna("unknown"), prefix="ex", dtype=float)
        x = pd.concat([x_num, x_ex], axis=1)
    else:
        x = x_num
    if ref_columns is not None:
        x = x.reindex(columns=ref_columns, fill_value=0.0)
    return x
