"""Train Models A/B and a survival-based Model C for PR timing.

Usage:
    python scripts/train_models.py

Model summary:
- Model A: XGBoost classifier (workout progression signal)
- Model B: XGBoost classifier (recovery signal)
- Model C: Random Survival Forest (sessions_until_next_pr)
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import joblib

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    from sksurv.metrics import concordance_index_censored
except ImportError as exc:
    raise ImportError(
        "scikit-survival is required for Model C. Install it with: pip install scikit-survival"
    ) from exc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)


# ------------------------------
# Shared data preparation
# ------------------------------
def load_data() -> pd.DataFrame:
    p = DATA / "processed_merged.csv"
    return pd.read_csv(p, parse_dates=["date"])


def add_training_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["exercise_title", "date"]).reset_index(drop=True)
    grp = df.groupby("exercise_title")
    df["training_age_sessions"] = grp.cumcount() + 1
    first_date = grp["date"].transform("min")
    df["training_age_days"] = (
        pd.to_datetime(df["date"]).dt.normalize() - pd.to_datetime(first_date).dt.normalize()
    ).dt.days
    return df


def add_recent_training_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["exercise_title", "date"]).set_index("date")
    out_frames = []
    for _, g in df.groupby("exercise_title"):
        g = g.sort_index()
        g["days_since_last_workout"] = g.index.to_series().diff().dt.days
        g["days_since_last_workout"] = g["days_since_last_workout"].fillna(9999)
        out_frames.append(g.reset_index())
    return pd.concat(out_frames, ignore_index=True).sort_values(["exercise_title", "date"]).reset_index(drop=True)


def ensure_model_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pr_gap" not in df.columns and "rolling_best_prev" in df.columns and "best_est_1RM" in df.columns:
        df["pr_gap"] = df["best_est_1RM"] - df["rolling_best_prev"]
    if "pr_gap" not in df.columns:
        df["pr_gap"] = 0.0

    if "pr_gap_percent" not in df.columns and "rolling_best_prev" in df.columns and "best_est_1RM" in df.columns:
        denom = df["rolling_best_prev"].replace(0, np.nan)
        df["pr_gap_percent"] = (df["rolling_best_prev"] - df["best_est_1RM"]) / denom
    if "pr_gap_percent" not in df.columns:
        df["pr_gap_percent"] = 0.0

    if "volume_ratio_28_56" not in df.columns and "volume_28d_avg" in df.columns and "volume_56d_avg" in df.columns:
        denom = df["volume_56d_avg"].replace(0, np.nan)
        df["volume_ratio_28_56"] = df["volume_28d_avg"] / denom
    if "volume_ratio_28_56" not in df.columns:
        df["volume_ratio_28_56"] = 0.0

    if "distance_to_personal_best" not in df.columns and "rolling_best_prev" in df.columns and "best_est_1RM" in df.columns:
        denom = df["rolling_best_prev"].replace(0, np.nan)
        df["distance_to_personal_best"] = (df["rolling_best_prev"] - df["best_est_1RM"]) / denom
    if "distance_to_personal_best" not in df.columns:
        df["distance_to_personal_best"] = 0.0

    for col in ["days_since_last_pr", "sessions_since_last_pr", "pr_freq_90d", "steps_7d_avg"]:
        if col not in df.columns:
            df[col] = 0.0

    # robustly create is_pr if missing
    if "is_pr" not in df.columns:
        if "rolling_best_prev" in df.columns and "best_est_1RM" in df.columns:
            df["is_pr"] = (df["best_est_1RM"] > df["rolling_best_prev"]).astype(int)
        else:
            df["is_pr"] = 0

    return df


def time_splits(df: pd.DataFrame, train_end: str = "2025-06-30", val_end: str = "2025-12-31"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[df["date"] <= pd.to_datetime(train_end)].copy()
    val = df[(df["date"] > pd.to_datetime(train_end)) & (df["date"] <= pd.to_datetime(val_end))].copy()
    test = df[df["date"] > pd.to_datetime(val_end)].copy()
    return train, val, test


def select_features(df: pd.DataFrame, feat_list: list[str]) -> list[str]:
    # Preserve order but drop duplicates to avoid duplicated DataFrame columns.
    seen = set()
    selected = []
    for c in feat_list:
        if c in df.columns and c not in seen:
            selected.append(c)
            seen.add(c)
    return selected


# ------------------------------
# Model A/B helpers
# ------------------------------
def evaluate_classifier(clf, X: pd.DataFrame, y: pd.Series) -> dict:
    probs = clf.predict_proba(X)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "avg_precision": float(average_precision_score(y, probs)),
        "brier": float(brier_score_loss(y, probs)),
    }


def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, params: dict | None = None):
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "use_label_encoder": False,
            "eval_metric": "logloss",
        }
    clf = xgb.XGBClassifier(**params)
    try:
        clf.fit(X_train, y_train, early_stopping_rounds=25, eval_set=[(X_val, y_val)], verbose=False)
    except TypeError:
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def expanding_window_oof_scores(df_train: pd.DataFrame, feature_cols: list[str], target_col: str, n_splits: int = 5) -> np.ndarray:
    """Generate out-of-fold, time-respecting scores for stacking features."""
    work = df_train.sort_values("date").reset_index().copy()
    # keep a stable reference to the original df_train row labels
    orig_idx = work["index"].to_numpy()
    y = work[target_col].astype(int).to_numpy()
    X = work[feature_cols].fillna(0)

    idx_blocks = np.array_split(np.arange(len(work)), n_splits)
    oof = np.full(len(work), np.nan)

    for i in range(1, len(idx_blocks)):
        tr_idx = np.concatenate(idx_blocks[:i])
        va_idx = idx_blocks[i]
        if len(np.unique(y[tr_idx])) < 2:
            continue
        clf = train_xgb(X.iloc[tr_idx], pd.Series(y[tr_idx]), X.iloc[va_idx], pd.Series(y[va_idx]))
        oof[va_idx] = clf.predict_proba(X.iloc[va_idx])[:, 1]

    # fill any missing OOF scores with model fitted on the full train split
    missing = np.isnan(oof)
    if missing.any() and len(np.unique(y)) >= 2:
        cutoff = int(len(work) * 0.8)
        clf_full = train_xgb(X.iloc[:cutoff], pd.Series(y[:cutoff]), X.iloc[cutoff:], pd.Series(y[cutoff:]))
        oof[missing] = clf_full.predict_proba(X.iloc[missing])[:, 1]

    # map scores back to the original df_train index ordering
    oof_series = pd.Series(np.nan_to_num(oof, nan=0.0), index=orig_idx)
    return oof_series.reindex(df_train.index).fillna(0.0).to_numpy()


# ------------------------------
# Survival labeling + RSF Model C
# ------------------------------
def construct_survival_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create right-censored session-based time-to-event labels.

    duration: sessions_until_next_pr
    event: next PR observed in future data (1) or censored (0)
    """
    df = df.copy()
    df = df.sort_values(["exercise_title", "date"]).reset_index(drop=True)

    durations = np.zeros(len(df), dtype=float)
    events = np.zeros(len(df), dtype=bool)

    for _, g in df.groupby("exercise_title", sort=False):
        idx = g.index.to_numpy()
        is_pr = g["is_pr"].fillna(0).astype(int).to_numpy()
        pr_positions = np.where(is_pr == 1)[0]
        n = len(g)

        for i in range(n):
            future = pr_positions[pr_positions > i]
            if len(future) > 0:
                durations[idx[i]] = float(future[0] - i)
                events[idx[i]] = True
            else:
                # right-censored at the end of observed sequence
                follow_up = max(1, n - 1 - i)
                durations[idx[i]] = float(follow_up)
                events[idx[i]] = False

    df["sessions_until_next_pr"] = durations
    df["event_observed"] = events.astype(int)
    return df


def build_model_c_matrix(df: pd.DataFrame, feature_cols: list[str], ref_columns: list[str] | None = None):
    """Build Model C matrix with numeric features + one-hot exercise identity."""
    X_num = df[feature_cols].copy().fillna(0.0)
    X_ex = pd.get_dummies(df["exercise_title"].fillna("unknown"), prefix="ex", dtype=float)
    X = pd.concat([X_num, X_ex], axis=1)
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0.0)
    return X


def train_rsf(X_train: pd.DataFrame, y_train_struct):
    rsf = RandomSurvivalForest(
        n_estimators=400,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rsf.fit(X_train, y_train_struct)
    return rsf


def expected_sessions_from_survival(rsf: RandomSurvivalForest, X: pd.DataFrame) -> np.ndarray:
    surv_fns = rsf.predict_survival_function(X, return_array=False)
    times = rsf.unique_times_
    delta = np.diff(np.r_[0.0, times])
    expected = []
    for fn in surv_fns:
        s = fn(times)
        expected.append(float(np.sum(s * delta)))
    return np.array(expected)


def prediction_interval_from_survival(rsf: RandomSurvivalForest, X_one: pd.DataFrame) -> list[float]:
    """Approximate interval using survival quantiles (not a strict statistical CI)."""
    fn = rsf.predict_survival_function(X_one, return_array=False)[0]
    times = rsf.unique_times_
    s = fn(times)

    def quantile_time(target_survival: float) -> float:
        mask = np.where(s <= target_survival)[0]
        if len(mask) == 0:
            return float(times[-1])
        return float(times[mask[0]])

    lower = quantile_time(0.75)
    upper = quantile_time(0.25)
    return [lower, upper]


def evaluate_rsf(rsf: RandomSurvivalForest, X: pd.DataFrame, event: pd.Series, duration: pd.Series) -> dict:
    expected = expected_sessions_from_survival(rsf, X)
    # concordance expects risk score (higher = earlier event), so use negative expected sessions
    c_index = concordance_index_censored(event.astype(bool).to_numpy(), duration.to_numpy(), -expected)[0]
    uncensored = event.astype(bool).to_numpy()
    mae = float(np.mean(np.abs(expected[uncensored] - duration.to_numpy()[uncensored]))) if uncensored.any() else np.nan
    return {
        "c_index": float(c_index),
        "mae_uncensored_sessions": mae,
        "mean_expected_sessions": float(np.mean(expected)),
    }


# ------------------------------
# Training orchestration
# ------------------------------
def run() -> None:
    df = load_data()
    df = add_training_age(df)
    df = add_recent_training_context(df)
    df = ensure_model_features(df)

    # keep rows with known PR classification label for Models A/B
    df = df[df["PR_next_session"].notnull()].copy()
    df = construct_survival_labels(df)

    train, val, test = time_splits(df)
    print("splits:", train.shape, val.shape, test.shape)

    # Model A (workout classifier)
    features_A = [
        "relative_strength", "rolling_best_prev", "best_est_1RM", "pr_gap_percent",
        "total_volume", "volume_28d_avg", "volume_56d_avg", "volume_ratio_28_56", "volume_28d_ratio", "volume_56d_ratio", "volume_28d_z", "volume_56d_z",
        "total_sets", "total_reps", "avg_weight", "max_weight",
        "days_since_last_pr", "sessions_since_last_pr", "pr_freq_90d",
        "training_age_sessions", "training_age_days",
    ]
    featsA = select_features(df, features_A)
    print("Model A features used:", featsA)

    XtrA = train[featsA].fillna(0)
    ytrA = train["PR_next_session"].astype(int)
    XvA = val[featsA].fillna(0)
    yvA = val["PR_next_session"].astype(int)
    XtA = test[featsA].fillna(0)
    ytA = test["PR_next_session"].astype(int)

    clfA = train_xgb(XtrA, ytrA, XvA, yvA)
    joblib.dump(clfA, OUT / "model_A_workout.joblib")
    metrics_A = {
        "train": evaluate_classifier(clfA, XtrA, ytrA),
        "val": evaluate_classifier(clfA, XvA, yvA),
        "test": evaluate_classifier(clfA, XtA, ytA),
    }
    (OUT / "feature_importance_A.json").write_text(json.dumps(dict(zip(featsA, clfA.feature_importances_.tolist())), indent=2))
    print("Model A metrics:", metrics_A)

    # Model B (recovery classifier)
    dfB = df[pd.to_datetime(df["date"]) >= pd.to_datetime("2024-11-01")].copy()
    trainB, valB, testB = time_splits(dfB)

    features_B = [
        "sleep_minutes", "sleep_7d_avg", "sleep_dev_from_14d",
        "resting_hr", "hr_7d_avg", "hr_baseline_z", "steps_7d_avg",
        "volume_28d_avg", "volume_56d_avg", "volume_ratio_28_56",
        "days_since_last_workout", "training_age_sessions",
    ]
    featsB = select_features(dfB, features_B)
    print("Model B features used:", featsB)

    XtrB = trainB[featsB].fillna(0)
    ytrB = trainB["PR_next_session"].astype(int)
    XvB = valB[featsB].fillna(0)
    yvB = valB["PR_next_session"].astype(int)
    XtB = testB[featsB].fillna(0)
    ytB = testB["PR_next_session"].astype(int)

    clfB = train_xgb(XtrB, ytrB, XvB, yvB)
    joblib.dump(clfB, OUT / "model_B_recovery.joblib")
    metrics_B = {
        "train": evaluate_classifier(clfB, XtrB, ytrB),
        "val": evaluate_classifier(clfB, XvB, yvB),
        "test": evaluate_classifier(clfB, XtB, ytB),
    }
    (OUT / "feature_importance_B.json").write_text(json.dumps(dict(zip(featsB, clfB.feature_importances_.tolist())), indent=2))
    print("Model B metrics:", metrics_B)

    # Stacking inputs for Model C (no 1/p inversion; direct survival target)
    # Use time-respecting OOF scores for training rows.
    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["model_a_score"] = expanding_window_oof_scores(train, featsA, "PR_next_session", n_splits=5)
    train["model_b_score"] = 0.0
    train_b_mask = train["date"] >= pd.to_datetime("2024-11-01")
    if train_b_mask.any() and len(np.unique(train.loc[train_b_mask, "PR_next_session"].astype(int))) > 1:
        train.loc[train_b_mask, "model_b_score"] = expanding_window_oof_scores(
            train.loc[train_b_mask].copy(), featsB, "PR_next_session", n_splits=5
        )

    # fit final A/B on train split and score val/test
    clfA_final = train_xgb(XtrA, ytrA, XvA, yvA)
    val["model_a_score"] = clfA_final.predict_proba(XvA)[:, 1]
    test["model_a_score"] = clfA_final.predict_proba(XtA)[:, 1]

    clfB_final = train_xgb(XtrB, ytrB, XvB, yvB)
    val["model_b_score"] = 0.0
    test["model_b_score"] = 0.0
    val_b_mask = val["date"] >= pd.to_datetime("2024-11-01")
    test_b_mask = test["date"] >= pd.to_datetime("2024-11-01")
    if val_b_mask.any():
        val.loc[val_b_mask, "model_b_score"] = clfB_final.predict_proba(val.loc[val_b_mask, featsB].fillna(0))[:, 1]
    if test_b_mask.any():
        test.loc[test_b_mask, "model_b_score"] = clfB_final.predict_proba(test.loc[test_b_mask, featsB].fillna(0))[:, 1]

    # Model C (Random Survival Forest)
    features_C = [
        "relative_strength",
        "pr_gap_percent",
        "rolling_best_prev",
        "best_est_1RM",
        "volume_28d_avg",
        "volume_56d_avg",
        "volume_ratio_28_56",
        "sessions_since_last_pr",
        "days_since_last_pr",
        "training_age_sessions",
        "training_age_days",
        "sleep_minutes",
        "sleep_7d_avg",
        "resting_hr",
        "hr_7d_avg",
        "hr_baseline_z",
        "steps_7d_avg",
        "model_a_score",
        "model_b_score",
    ]
    featsC = select_features(pd.concat([train, val, test], axis=0), features_C)
    print("Model C (RSF) features used:", featsC)

    XtrC = build_model_c_matrix(train, featsC)
    XvC = build_model_c_matrix(val, featsC, ref_columns=list(XtrC.columns))
    XtC = build_model_c_matrix(test, featsC, ref_columns=list(XtrC.columns))

    ytrC = Surv.from_arrays(event=train["event_observed"].astype(bool).to_numpy(), time=train["sessions_until_next_pr"].astype(float).to_numpy())

    rsf = train_rsf(XtrC, ytrC)
    joblib.dump({"model": rsf, "feature_columns": list(XtrC.columns), "base_features": featsC}, OUT / "model_C_rsf_survival.joblib")

    metrics_C = {
        "train": evaluate_rsf(rsf, XtrC, train["event_observed"], train["sessions_until_next_pr"]),
        "val": evaluate_rsf(rsf, XvC, val["event_observed"], val["sessions_until_next_pr"]),
        "test": evaluate_rsf(rsf, XtC, test["event_observed"], test["sessions_until_next_pr"]),
    }
    print("Model C (RSF) metrics:", metrics_C)

    # Save expected sessions for downstream inspection
    out_val = val[["date", "exercise_title", "sessions_until_next_pr", "event_observed"]].copy()
    out_val["expected_sessions_until_pr"] = expected_sessions_from_survival(rsf, XvC)
    out_test = test[["date", "exercise_title", "sessions_until_next_pr", "event_observed"]].copy()
    out_test["expected_sessions_until_pr"] = expected_sessions_from_survival(rsf, XtC)
    out_val.to_csv(OUT / "model_C_val_predictions.csv", index=False)
    out_test.to_csv(OUT / "model_C_test_predictions.csv", index=False)

    # Example output structure
    if len(out_test) > 0:
        ex_row = out_test.sort_values("date").iloc[-1]
        x_one = XtC.loc[[ex_row.name]] if ex_row.name in XtC.index else XtC.iloc[[0]]
        interval = prediction_interval_from_survival(rsf, x_one)
        example = {
            "exercise": str(ex_row["exercise_title"]),
            "expected_sessions_to_PR": float(ex_row["expected_sessions_until_pr"]),
            "confidence_interval": [float(interval[0]), float(interval[1])],
        }
    else:
        example = {
            "exercise": "Bench Press",
            "expected_sessions_to_PR": 2.4,
            "confidence_interval": [1.5, 4.0],
        }
    (OUT / "model_C_example_output.json").write_text(json.dumps(example, indent=2))

    # Consolidated metrics
    all_metrics = {"A": metrics_A, "B": metrics_B, "C_survival": metrics_C}
    (OUT / "metrics_summary.json").write_text(json.dumps(all_metrics, indent=2))
    print("Saved models and metrics to", OUT)


if __name__ == "__main__":
    run()
