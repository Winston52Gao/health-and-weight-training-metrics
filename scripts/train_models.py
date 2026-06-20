"""Train Models A (workout), B (recovery), and C (stacked final) for PR prediction.

Usage:
    python scripts/train_models.py

Notes:
- Expects `data/processed_merged.csv` produced by `scripts/prepare_features.py`.
- Default time splits: train <= 2025-06-30, val 2025-07-01..2025-12-31, test >= 2026-01-01
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)


def load_data():
    p = DATA / "processed_merged.csv"
    df = pd.read_csv(p, parse_dates=["date"]) 
    return df


def add_training_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["exercise_title", "date"]).reset_index(drop=True)
    grp = df.groupby("exercise_title")
    # training_age_sessions: cumulative count of sessions for exercise (starting at 1)
    df["training_age_sessions"] = grp.cumcount() + 1
    # training_age_days: days since first session
    first_date = grp["date"].transform("min")
    df["training_age_days"] = (pd.to_datetime(df["date"]).dt.normalize() - pd.to_datetime(first_date).dt.normalize()).dt.days
    return df


def add_recent_training_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["exercise_title", "date"]).set_index("date")
    out_frames = []
    for ex, g in df.groupby("exercise_title"):
        g = g.sort_index()
        # days since last session
        g["days_since_last_workout"] = g.index.to_series().diff().dt.days
        # recent_volume = weekly_volume (already computed) fallback to total_volume
        if "weekly_volume" not in g.columns:
            g["weekly_volume"] = g["total_volume"].rolling("7D", closed='left').sum()
        # ensure numeric
        g["days_since_last_workout"] = g["days_since_last_workout"].fillna(9999)
        out_frames.append(g.reset_index())
    return pd.concat(out_frames, ignore_index=True).sort_values(["exercise_title","date"]).reset_index(drop=True)


def time_splits(df: pd.DataFrame, train_end="2025-06-30", val_end="2025-12-31"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]) 
    train = df[df["date"] <= pd.to_datetime(train_end)]
    val = df[(df["date"] > pd.to_datetime(train_end)) & (df["date"] <= pd.to_datetime(val_end))]
    test = df[df["date"] > pd.to_datetime(val_end)]
    return train, val, test


def evaluate_model(clf, X, y):
    probs = clf.predict_proba(X)[:,1]
    auc = roc_auc_score(y, probs)
    ap = average_precision_score(y, probs)
    brier = brier_score_loss(y, probs)
    return {"roc_auc": float(auc), "avg_precision": float(ap), "brier": float(brier)}


def train_xgb(X_train, y_train, X_val, y_val, params=None):
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
    clf = xgb.XGBClassifier(**params)
    clf.fit(X_train, y_train, early_stopping_rounds=20, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def select_features(df, feat_list):
    return [c for c in feat_list if c in df.columns]


def run():
    df = load_data()
    df = add_training_age(df)
    df = add_recent_training_context(df)

    # only keep rows with PR_next_session label
    df = df[df["PR_next_session"].notnull()].copy()

    train, val, test = time_splits(df)
    print('splits:', train.shape, val.shape, test.shape)

    # --- Model A: workout history only ---
    features_A = [
        "relative_strength", "rolling_best_prev", "best_est_1RM", "pr_gap", "distance_to_personal_best",
        "total_volume", "weekly_volume", "volume_28d_avg", "volume_56d_avg", "volume_ratio", "weekly_volume_z",
        "total_sets", "total_reps", "avg_weight", "max_weight",
        "days_since_last_pr", "sessions_since_last_pr", "pr_freq_90d",
        "training_age_sessions", "training_age_days"
    ]
    # compute pr_gap if missing
    if "pr_gap" not in df.columns and "rolling_best_prev" in df.columns:
        df["pr_gap"] = df["rolling_best_prev"] - df["best_est_1RM"]

    featsA = select_features(df, features_A)
    print('Model A features used:', featsA)

    Xtr = train[featsA].fillna(0)
    ytr = train["PR_next_session"].astype(int)
    Xv = val[featsA].fillna(0)
    yv = val["PR_next_session"].astype(int)
    Xt = test[featsA].fillna(0)
    yt = test["PR_next_session"].astype(int)

    clfA = train_xgb(Xtr, ytr, Xv, yv)
    joblib.dump(clfA, OUT / "model_A_workout.joblib")
    metrics_A = {
        "train": evaluate_model(clfA, Xtr, ytr),
        "val": evaluate_model(clfA, Xv, yv),
        "test": evaluate_model(clfA, Xt, yt),
    }
    print('Model A metrics:', metrics_A)
    # feature importance
    fiA = dict(zip(featsA, clfA.feature_importances_.tolist()))
    (OUT / "feature_importance_A.json").write_text(json.dumps(fiA, indent=2))

    # --- Model B: recovery model (Fitbit features + recent training) ---
    # restrict to dates >= 2024-11-01
    dfB = df[pd.to_datetime(df["date"]) >= pd.to_datetime('2024-11-01')].copy()
    trainB, valB, testB = time_splits(dfB)

    features_B = [
        "sleep_minutes", "sleep_7d_avg", "sleep_dev_from_14d",
        "resting_hr", "hr_7d_avg", "hr_baseline_z",
        "steps", "steps_7d_avg",
        "weekly_volume", "days_since_last_workout", "training_age_sessions"
    ]
    featsB = select_features(dfB, features_B)
    print('Model B features used:', featsB)

    Xtr = trainB[featsB].fillna(0)
    ytr = trainB["PR_next_session"].astype(int)
    Xv = valB[featsB].fillna(0)
    yv = valB["PR_next_session"].astype(int)
    Xt = testB[featsB].fillna(0)
    yt = testB["PR_next_session"].astype(int)

    clfB = train_xgb(Xtr, ytr, Xv, yv)
    joblib.dump(clfB, OUT / "model_B_recovery.joblib")
    metrics_B = {"train": evaluate_model(clfB, Xtr, ytr), "val": evaluate_model(clfB, Xv, yv), "test": evaluate_model(clfB, Xt, yt)}
    print('Model B metrics:', metrics_B)
    fiB = dict(zip(featsB, clfB.feature_importances_.tolist()))
    (OUT / "feature_importance_B.json").write_text(json.dumps(fiB, indent=2))

    # --- Model C: stacked final model ---
    # Create stacking features: use predictions from A and B (on corresponding date ranges)
    # Get A predictions for full df (use clfA)
    df_all = df.copy()
    df_all["pr_prob_workout"] = clfA.predict_proba(df_all[featsA].fillna(0))[:,1] if len(featsA)>0 else 0.0
    # For B: predictions only available for dates >= 2024-11-01; set others to NaN->0
    if len(featsB) > 0:
        # need to align features for df_all: if missing columns, fill 0
        dfB_all = df_all.copy()
        for c in featsB:
            if c not in dfB_all.columns:
                dfB_all[c] = 0
        df_all["pr_prob_recovery"] = clfB.predict_proba(dfB_all[featsB].fillna(0))[:,1]
    else:
        df_all["pr_prob_recovery"] = 0.0

    # assemble features for Model C
    features_C = ["pr_prob_workout", "pr_prob_recovery", "relative_strength", "pr_gap", "days_since_last_pr", "training_age_sessions"]
    featsC = select_features(df_all, features_C)
    print('Model C features used:', featsC)

    # Use same time splits as df
    trainC, valC, testC = time_splits(df_all)
    Xtr = trainC[featsC].fillna(0)
    ytr = trainC["PR_next_session"].astype(int)
    Xv = valC[featsC].fillna(0)
    yv = valC["PR_next_session"].astype(int)
    Xt = testC[featsC].fillna(0)
    yt = testC["PR_next_session"].astype(int)

    clfC = train_xgb(Xtr, ytr, Xv, yv)
    joblib.dump(clfC, OUT / "model_C_stacked.joblib")
    metrics_C = {"train": evaluate_model(clfC, Xtr, ytr), "val": evaluate_model(clfC, Xv, yv), "test": evaluate_model(clfC, Xt, yt)}
    print('Model C metrics:', metrics_C)
    fiC = dict(zip(featsC, clfC.feature_importances_.tolist()))
    (OUT / "feature_importance_C.json").write_text(json.dumps(fiC, indent=2))

    # Save metrics summary
    (OUT / "metrics_summary.json").write_text(json.dumps({"A": metrics_A, "B": metrics_B, "C": metrics_C}, indent=2))
    print('Saved models and metrics to', OUT)


if __name__ == '__main__':
    run()
