"""Prepare merged workout + Fitbit datasets and engineer features for two models.

Outputs (saved to `data/`):
- processed_merged.csv
- features_model1.csv
- features_model2.csv
- column_definitions.md

Usage: run this script from the repository root with Python 3.10+.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def parse_start_date(s: str) -> pd.Timestamp | pd.NaT:
    """Extract the date portion from varied start_time strings and return a Timestamp (date only).

    Examples handled:
    - "May 24, 2026, 8:15p.m"
    - "May 24, 2026, 8:15 p.m."
    - other spacing / punctuation variants
    """
    if pd.isna(s):
        return pd.NaT
    # capture patterns like 'May 24, 2026' at start of string
    m = re.search(r"([A-Za-z]+\s+\d{1,2},\s*\d{4})", str(s))
    if not m:
        try:
            return pd.to_datetime(s).normalize()
        except Exception:
            return pd.NaT
    try:
        return pd.to_datetime(m.group(1)).normalize()
    except Exception:
        return pd.NaT


def load_inputs():
    workouts_path = ROOT / "data" / "workouts.csv"
    fitbit_path = ROOT / "fitbit_merged.csv"
    w = pd.read_csv(workouts_path)
    f = pd.read_csv(fitbit_path)
    return w, f


def standardize_workout_dates(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["parsed_start"] = w["start_time"].astype(str).apply(parse_start_date)
    w["date"] = pd.to_datetime(w["parsed_start"]).dt.date
    w = w.drop(columns=["parsed_start"])
    return w


def aggregate_workouts(w: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to session-level per (date, exercise).

    Assumes each row is a set or an exercise-entry with columns `exercise`, `weight`, `reps`, `sets`.
    If `sets` not present, total_sets will be the count of rows aggregated.
    """
    w = w.copy()
    # coerce numeric
    for col in ["weight", "reps"]:
        if col in w.columns:
            w[col] = pd.to_numeric(w[col], errors="coerce").fillna(0)
        else:
            w[col] = 0

    # estimated 1RM per row (Epley)
    w["est_1RM_set"] = w["weight"] * (1 + w["reps"] / 30)

    # determine sets column
    if "sets" in w.columns:
        w["sets"] = pd.to_numeric(w["sets"], errors="coerce").fillna(0)

    agg_funcs = {
        "weight": ["mean", "max"],
        "reps": "sum",
        "est_1RM_set": "max",
    }
    # total sets: sum if present else count
    if "sets" in w.columns:
        agg_funcs["sets"] = "sum"

    grouped = (
        w.groupby(["date", "exercise"], dropna=False)
        .agg(agg_funcs)
        .reset_index()
    )
    # flatten columns
    grouped.columns = [
        "date",
        "exercise",
        "avg_weight",
        "max_weight",
        "total_reps",
        "best_est_1RM",
    ] + (["total_sets"] if "sets" in w.columns else [])

    # if total_sets not present (no sets column), approximate by counting rows per group
    if "total_sets" not in grouped.columns:
        counts = w.groupby(["date", "exercise"]).size().reset_index(name="total_sets")
        grouped = grouped.merge(counts, on=["date", "exercise"], how="left")

    # total_volume = sum(weight * reps) per group
    w["volume"] = w["weight"] * w["reps"]
    vol = w.groupby(["date", "exercise"])['volume'].sum().reset_index(name='total_volume')
    grouped = grouped.merge(vol, on=["date", "exercise"], how="left")

    # ensure date is datetime
    grouped["date"] = pd.to_datetime(grouped["date"]).dt.normalize()

    # reorder
    cols = ["date", "exercise", "total_volume", "total_sets", "avg_weight", "max_weight", "total_reps", "best_est_1RM"]
    return grouped[cols]


def compute_per_exercise_time_features(sess: pd.DataFrame) -> pd.DataFrame:
    df = sess.copy()
    df = df.sort_values(["exercise", "date"])

    out_frames = []
    for ex, g in df.groupby("exercise", sort=False):
        g = g.sort_values("date").set_index("date")

        # rolling best up to previous session (no leakage)
        g["rolling_best_prev"] = g["best_est_1RM"].cummax().shift(1)

        # relative strength (current best / prior best)
        g["relative_strength"] = g["best_est_1RM"] / g["rolling_best_prev"]

        # target: delta relative strength at next session
        g["next_relative_strength"] = g["relative_strength"].shift(-1)
        g["delta_relative_strength_next_session"] = g["next_relative_strength"] - g["relative_strength"]

        # PR flags (is current a PR compared to prior history)
        g["is_pr"] = (g["best_est_1RM"] > g["rolling_best_prev"]).astype(int)

        # rolling max including current (used to evaluate next-session PR)
        g["rolling_max_including_current"] = g["best_est_1RM"].cummax()
        g["next_best"] = g["best_est_1RM"].shift(-1)
        g["PR_next_session"] = (g["next_best"] > g["rolling_max_including_current"]).astype(int)

        # time-based rolling for volume: weekly (7d), 28d, 56d baseline
        g.index = pd.to_datetime(g.index)
        g = g.sort_index()
        g["weekly_volume"] = g["total_volume"].rolling("7D", closed="left").sum()
        g["volume_28d_avg"] = g["total_volume"].rolling("28D", closed="left").mean()
        g["volume_56d_avg"] = g["total_volume"].rolling("56D", closed="left").mean()
        # avoid leakage: these are already excluding current via closed='left'

        # volume ratio: weekly / 8-week baseline (use 56d mean)
        g["volume_ratio"] = g["weekly_volume"] / g["volume_56d_avg"]

        # normalize weekly_volume per exercise via rolling z-score (no global normalization)
        vol_mean = g["weekly_volume"].rolling("56D", closed="left").mean()
        vol_std = g["weekly_volume"].rolling("56D", closed="left").std(ddof=0)
        g["weekly_volume_z"] = (g["weekly_volume"] - vol_mean) / vol_std

        # days since last PR per exercise
        pr_dates = g.index.where(g["is_pr"] == 1)
        last_pr_before = pr_dates.ffill().shift(1)
        g["days_since_last_pr"] = (g.index.to_series() - last_pr_before).dt.days

        # rolling PR frequency in last 90 days (exclude current)
        g["pr_freq_90d"] = g["is_pr"].rolling("90D", closed="left").sum()

        # sessions since last progression: count sessions since last PR
        # group by cumulative PR count to compute counts since last PR
        grp_id = g["is_pr"].cumsum()
        g["sessions_since_last_pr"] = g.groupby(grp_id).cumcount()
        g["sessions_since_last_pr"] = g["sessions_since_last_pr"].shift(1)

        # distance to personal best (relative)
        g["distance_to_personal_best"] = (g["rolling_best_prev"] - g["best_est_1RM"]) / g["rolling_best_prev"]

        out_frames.append(g.reset_index())

    result = pd.concat(out_frames, axis=0).sort_values(["exercise", "date"]).reset_index(drop=True)
    return result


def compute_fitbit_features(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    # ensure date is datetime
    if "date" not in f.columns:
        raise ValueError("fitbit_merged.csv must contain a 'date' column")
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    f = f.set_index("date").sort_index()

    # sleep features
    if "sleep_minutes" in f.columns:
        f["sleep_7d_avg"] = f["sleep_minutes"].rolling("7D", closed="left").mean().shift(1)
        f["sleep_14d_mean"] = f["sleep_minutes"].rolling("14D", closed="left").mean().shift(1)
        f["sleep_dev_from_14d"] = f["sleep_minutes"] - f["sleep_14d_mean"]
    else:
        f["sleep_minutes"] = np.nan

    # heart rate features
    hr_col = None
    for candidate in ["resting_heart_rate", "resting_hr", "resting_heart"]:
        if candidate in f.columns:
            hr_col = candidate
            break
    if hr_col:
        f["hr_7d_avg"] = f[hr_col].rolling("7D", closed="left").mean().shift(1)
        f["hr_28d_mean"] = f[hr_col].rolling("28D", closed="left").mean().shift(1)
        f["hr_baseline_z"] = (f[hr_col] - f["hr_28d_mean"]) / f[hr_col].rolling("56D", closed="left").std()
        f["resting_hr"] = f[hr_col]
    else:
        f["resting_hr"] = np.nan

    # keep index as column for merging
    out = f.reset_index()
    to_keep = [c for c in ["date", "sleep_minutes", "sleep_7d_avg", "sleep_dev_from_14d", "resting_hr", "hr_7d_avg", "hr_baseline_z"] if c in out.columns]
    return out[to_keep]


def merge_workouts_fitbit(sess_feats: pd.DataFrame, fitbit_feats: pd.DataFrame) -> pd.DataFrame:
    # left join on sessions so that every session is preserved
    merged = sess_feats.merge(fitbit_feats, on="date", how="left")
    return merged


def assemble_feature_sets(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = merged.copy()
    # Model 1 features
    model1_cols = [
        "date",
        "exercise",
        "relative_strength",
        "delta_relative_strength_next_session",
        "sleep_minutes",
        "sleep_7d_avg",
        "sleep_dev_from_14d",
        "resting_hr",
        "hr_7d_avg",
        "hr_baseline_z",
        "weekly_volume",
        "volume_28d_avg",
        "volume_ratio",
        "weekly_volume_z",
        "days_since_last_pr",
        "pr_freq_90d",
        "sessions_since_last_pr",
        "distance_to_personal_best",
    ]

    # Model 2 features (classification for PR_next_session)
    model2_cols = model1_cols + ["PR_next_session", "best_est_1RM"]

    model1 = df[[c for c in model1_cols if c in df.columns]].copy()
    model2 = df[[c for c in model2_cols if c in df.columns]].copy()

    return model1, model2


def save_outputs(merged: pd.DataFrame, model1: pd.DataFrame, model2: pd.DataFrame):
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "processed_merged.csv", index=False)
    model1.to_csv(out_dir / "features_model1.csv", index=False)
    model2.to_csv(out_dir / "features_model2.csv", index=False)

    # column definitions
    defs = {
        "date": "Session date (YYYY-MM-DD)",
        "exercise": "Exercise name",
        "total_volume": "Sum of weight*reps for the session",
        "total_sets": "Total sets in session",
        "avg_weight": "Average weight used in session",
        "max_weight": "Max weight used in session",
        "total_reps": "Sum of reps in session",
        "best_est_1RM": "Best estimated 1RM in the session (Epley per-set, then max)",
        "rolling_best_prev": "Best estimated 1RM prior to the current session (no leakage)",
        "relative_strength": "best_est_1RM / rolling_best_prev",
        "delta_relative_strength_next_session": "Target: change in relative strength at next session",
        "PR_next_session": "Target binary: 1 if next session sets a new best_est_1RM",
        "sleep_minutes": "Sleep minutes on session date (from Fitbit)",
        "sleep_7d_avg": "7-day average sleep minutes (prior days)",
        "sleep_dev_from_14d": "Sleep minus 14-day mean",
        "resting_hr": "Resting heart rate on session date",
        "hr_7d_avg": "7-day average resting HR (prior days)",
        "hr_baseline_z": "Z-score of resting HR relative to 56-day window",
        "weekly_volume": "Sum of volume in previous 7 days (per exercise)",
        "volume_28d_avg": "28-day avg volume (prior days)",
        "volume_56d_avg": "56-day avg volume (prior days)",
        "volume_ratio": "weekly_volume / 56-day avg",
        "weekly_volume_z": "Per-exercise z-score for weekly volume (rolling)",
        "days_since_last_pr": "Days since last per-exercise PR",
        "pr_freq_90d": "Number of PRs in the previous 90 days",
        "sessions_since_last_pr": "Sessions since last PR",
        "distance_to_personal_best": "Relative distance to prior personal best",
    }
    with open(out_dir / "column_definitions.md", "w", encoding="utf8") as fh:
        fh.write("# Column definitions\n\n")
        for k, v in defs.items():
            fh.write(f"- **{k}**: {v}\n")


def main():
    w, f = load_inputs()
    w = standardize_workout_dates(w)
    sess = aggregate_workouts(w)
    sess_feats = compute_per_exercise_time_features(sess)
    fitbit_feats = compute_fitbit_features(f)
    merged = merge_workouts_fitbit(sess_feats, fitbit_feats)
    model1, model2 = assemble_feature_sets(merged)
    save_outputs(merged, model1, model2)
    print("Saved: data/processed_merged.csv, data/features_model1.csv, data/features_model2.csv, data/column_definitions.md")


if __name__ == "__main__":
    main()
