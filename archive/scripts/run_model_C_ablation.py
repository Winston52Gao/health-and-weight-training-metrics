"""Run Model C ablation study without modifying production training script.

This script compares feature-group variants for Model C (RSF survival model)
using the same preprocessing, split logic, and survival label construction as
scripts/train_models.py.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sksurv.ensemble import RandomSurvivalForest

try:
    from sksurv.metrics import brier_score as sksurv_brier_score
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    try:
        from sksurv.metrics import integrated_brier_score
    except ImportError:
        integrated_brier_score = None
except ImportError as exc:
    raise ImportError(
        "scikit-survival is required for Model C ablation. Install it with: pip install scikit-survival"
    ) from exc

from src.evaluate_model_C import (  # noqa: E402
    construct_survival_labels,
    evaluate_split,
    time_splits,
)
from src.feature_engineering import (  # noqa: E402
    add_recent_training_context,
    add_training_age,
    build_model_c_matrix,
    ensure_model_features,
    load_data,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MODEL_C_DIR = ROOT / "models" / "model_C"
EXPERIMENTS_DIR = MODEL_C_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

BRIER_HORIZONS = [5, 10, 20]
FITBIT_START_DATE = "2024-11-01"
FITBIT_MISSING_THRESHOLD = 0.80

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


def coverage_report(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for feat in features:
        if feat not in df.columns:
            rows.append(
                {
                    "feature": feat,
                    "non_null_pct": 0.0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "max": np.nan,
                }
            )
            continue

        series = pd.to_numeric(df[feat], errors="coerce")
        non_null = series.notna().sum()
        rows.append(
            {
                "feature": feat,
                "non_null_pct": float(100.0 * non_null / n) if n > 0 else np.nan,
                "mean": float(series.mean()) if non_null > 0 else np.nan,
                "std": float(series.std()) if non_null > 1 else np.nan,
                "min": float(series.min()) if non_null > 0 else np.nan,
                "max": float(series.max()) if non_null > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def select_features(df: pd.DataFrame, feature_pool: list[str]) -> list[str]:
    seen = set()
    selected = []
    for feature in feature_pool:
        if feature in df.columns and feature not in seen:
            selected.append(feature)
            seen.add(feature)
    return selected


def train_rsf(x_train: pd.DataFrame, y_train):
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


def build_eval_grid(train_times: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
    train_times = np.asarray(train_times, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float)
    if train_times.size == 0 or eval_times.size == 0:
        return np.array([], dtype=float)

    lower = max(float(np.min(train_times)), float(np.min(eval_times)))
    upper = min(float(np.max(train_times)), float(np.max(eval_times)))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return np.array([], dtype=float)

    grid = np.unique(train_times[(train_times >= lower) & (train_times <= upper)])
    if len(grid) > 2:
        grid = grid[:-1]
    return grid


def baseline_survival_matrix(median_sessions: float, grid: np.ndarray, n_rows: int) -> np.ndarray:
    if len(grid) == 0:
        return np.empty((n_rows, 0))
    surv = (grid < median_sessions).astype(float)
    return np.tile(surv, (n_rows, 1))


def compute_ibs_generic(
    y_train_struct,
    y_eval_struct,
    surv_matrix: np.ndarray,
    grid: np.ndarray,
) -> float:
    if surv_matrix.size == 0 or len(grid) < 3:
        return np.nan

    try:
        if integrated_brier_score is None:
            raise AttributeError("integrated_brier_score unavailable")
        return float(integrated_brier_score(y_train_struct, y_eval_struct, surv_matrix, grid))
    except Exception:
        try:
            times_out, brier_t = sksurv_brier_score(y_train_struct, y_eval_struct, surv_matrix, grid)
            if len(times_out) < 2 or not np.all(np.isfinite(brier_t)):
                return np.nan
            denom = float(times_out[-1] - times_out[0])
            if denom <= 0:
                return np.nan
            if hasattr(np, "trapezoid"):
                return float(np.trapezoid(brier_t, times_out) / denom)
            return float(np.trapz(brier_t, times_out) / denom)
        except Exception:
            return np.nan


def brier_and_calibration_from_prob(
    probs: np.ndarray,
    event_bool: np.ndarray,
    duration_np: np.ndarray,
    horizon: int,
) -> tuple[float, list[dict]]:
    known_mask = ~((~event_bool) & (duration_np <= horizon))
    if known_mask.sum() < 20:
        return np.nan, []

    y_h = (event_bool & (duration_np <= horizon)).astype(int)
    y_known = y_h[known_mask]
    p_known = probs[known_mask]
    brier = float(brier_score_loss(y_known, p_known))

    cal_df = pd.DataFrame({"y": y_known, "p": p_known})
    cal_df["bin"] = pd.qcut(cal_df["p"], q=min(5, max(2, cal_df["p"].nunique())), duplicates="drop")
    cal = (
        cal_df.groupby("bin", observed=True)
        .agg(predicted_prob=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    return brier, cal.to_dict(orient="records")


def evaluate_baseline_median(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    split_name: str,
) -> dict:
    median_sessions = float(train_df["sessions_until_next_pr"].median())

    event_eval = eval_df["event_observed"].astype(bool).to_numpy()
    duration_eval = eval_df["sessions_until_next_pr"].astype(float).to_numpy()
    expected = np.full(len(eval_df), median_sessions, dtype=float)

    y_train_struct = Surv.from_arrays(
        event=train_df["event_observed"].astype(bool).to_numpy(),
        time=train_df["sessions_until_next_pr"].astype(float).to_numpy(),
    )
    y_eval_struct = Surv.from_arrays(event=event_eval, time=duration_eval)

    train_times = train_df["sessions_until_next_pr"].astype(float).to_numpy()
    grid = build_eval_grid(train_times, duration_eval)
    surv_matrix = baseline_survival_matrix(median_sessions, grid, len(eval_df))

    c_index = concordance_index_censored(event_eval, duration_eval, -expected)[0]
    mae_diag = float(np.mean(np.abs(expected[event_eval] - duration_eval[event_eval]))) if event_eval.any() else np.nan
    ibs = compute_ibs_generic(y_train_struct, y_eval_struct, surv_matrix, grid)

    brier_by_horizon = {}
    calibration_by_horizon = {}
    for h in BRIER_HORIZONS:
        p_h = np.full(len(eval_df), 1.0 if median_sessions <= h else 0.0, dtype=float)
        brier_h, cal_h = brier_and_calibration_from_prob(p_h, event_eval, duration_eval, h)
        brier_by_horizon[str(h)] = brier_h
        calibration_by_horizon[str(h)] = cal_h

    return {
        "split": split_name,
        "c_index": float(c_index),
        "ibs": ibs,
        "mae_uncensored_sessions_diagnostic": mae_diag,
        "mean_expected_sessions": float(np.mean(expected)),
        "brier_by_horizon": brier_by_horizon,
        "calibration_by_horizon": calibration_by_horizon,
        "median_sessions_train": median_sessions,
    }


def run_rsf_experiment(
    name: str,
    feature_pool: list[str],
    include_identity: bool,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    all_df = pd.concat([train_df, val_df, test_df], axis=0)
    feats = select_features(all_df, feature_pool)

    X_train = build_model_c_matrix(train_df, feats, include_exercise_identity=include_identity)
    X_val = build_model_c_matrix(val_df, feats, ref_columns=list(X_train.columns), include_exercise_identity=include_identity)
    X_test = build_model_c_matrix(test_df, feats, ref_columns=list(X_train.columns), include_exercise_identity=include_identity)

    y_train = Surv.from_arrays(
        event=train_df["event_observed"].astype(bool).to_numpy(),
        time=train_df["sessions_until_next_pr"].astype(float).to_numpy(),
    )

    model = train_rsf(X_train, y_train)

    return {
        "name": name,
        "features_used": feats,
        "include_exercise_identity": include_identity,
        "n_train_features": int(X_train.shape[1]),
        "metrics": {
            "train": evaluate_split(model, X_train, train_df["event_observed"], train_df["sessions_until_next_pr"], BRIER_HORIZONS, y_train, "train"),
            "val": evaluate_split(model, X_val, val_df["event_observed"], val_df["sessions_until_next_pr"], BRIER_HORIZONS, y_train, "validation"),
            "test": evaluate_split(model, X_test, test_df["event_observed"], test_df["sessions_until_next_pr"], BRIER_HORIZONS, y_train, "test"),
        },
    }


def row_from_metrics(model_name: str, feature_label: str, metrics_test: dict) -> dict:
    return {
        "Model": model_name,
        "Features": feature_label,
        "C-index": metrics_test.get("c_index", np.nan),
        "IBS": metrics_test.get("ibs", np.nan),
        "MAE": metrics_test.get("mae_uncensored_sessions_diagnostic", np.nan),
        "Brier@5": metrics_test.get("brier_by_horizon", {}).get("5", np.nan),
        "Brier@10": metrics_test.get("brier_by_horizon", {}).get("10", np.nan),
        "Brier@20": metrics_test.get("brier_by_horizon", {}).get("20", np.nan),
    }


def run() -> None:
    df_all = load_data()
    df_all = add_training_age(df_all)
    df_all = add_recent_training_context(df_all)
    df_all = ensure_model_features(df_all)

    df = df_all[df_all["PR_next_session"].notnull()].copy()
    df = construct_survival_labels(df)

    cov_features = WORKOUT_FEATURES + FITBIT_FEATURES
    cov = coverage_report(df, cov_features)
    cov.to_csv(EXPERIMENTS_DIR / "model_C_feature_coverage.csv", index=False)

    fitbit_cov = cov[cov["feature"].isin(FITBIT_FEATURES)].copy()
    fitbit_non_null_rates = fitbit_cov["non_null_pct"] / 100.0 
    run_fitbit_period = bool((fitbit_non_null_rates < FITBIT_MISSING_THRESHOLD).any())

    train, val, test = time_splits(df)

    results = []
    details = {}

    baseline_train = evaluate_baseline_median(train, train, "train")
    baseline_val = evaluate_baseline_median(train, val, "validation")
    baseline_test = evaluate_baseline_median(train, test, "test")
    details["Experiment 0: Baseline median"] = {
        "metrics": {
            "train": baseline_train,
            "val": baseline_val,
            "test": baseline_test,
        }
    }
    results.append(
        row_from_metrics("Baseline median", "none", baseline_test)
    )

    exp1 = run_rsf_experiment(
        name="Experiment 1: Workout only",
        feature_pool=WORKOUT_FEATURES,
        include_identity=False,
        train_df=train,
        val_df=val,
        test_df=test,
    )
    details[exp1["name"]] = exp1
    results.append(
        row_from_metrics("Workout only", "training", exp1["metrics"]["test"])
    )

    exp2 = run_rsf_experiment(
        name="Experiment 2: Workout + exercise identity",
        feature_pool=WORKOUT_FEATURES,
        include_identity=True,
        train_df=train,
        val_df=val,
        test_df=test,
    )
    details[exp2["name"]] = exp2
    results.append(
        row_from_metrics("Workout + exercise", "training + identity", exp2["metrics"]["test"])
    )

    exp3 = run_rsf_experiment(
        name="Experiment 3: Workout + Fitbit",
        feature_pool=WORKOUT_FEATURES + FITBIT_FEATURES,
        include_identity=False,
        train_df=train,
        val_df=val,
        test_df=test,
    )
    details[exp3["name"]] = exp3
    results.append(
        row_from_metrics("Workout + Fitbit", "training + recovery", exp3["metrics"]["test"])
    )

    exp4 = run_rsf_experiment(
        name="Experiment 4: Full Model C",
        feature_pool=WORKOUT_FEATURES + FITBIT_FEATURES,
        include_identity=True,
        train_df=train,
        val_df=val,
        test_df=test,
    )
    details[exp4["name"]] = exp4
    results.append(
        row_from_metrics("Full Model C", "all features", exp4["metrics"]["test"])
    )

    if run_fitbit_period:
        df_fitbit = df[df["date"] >= pd.to_datetime(FITBIT_START_DATE)].copy()
        tr_f, va_f, te_f = time_splits(df_fitbit)

        # Skip experiment 5 if any split is too small for stable RSF training/eval.
        if min(len(tr_f), len(va_f), len(te_f)) >= 30:
            exp5a = run_rsf_experiment(
                name="Experiment 5a: Fitbit period workout only",
                feature_pool=WORKOUT_FEATURES,
                include_identity=False,
                train_df=tr_f,
                val_df=va_f,
                test_df=te_f,
            )
            exp5b = run_rsf_experiment(
                name="Experiment 5b: Fitbit period workout + Fitbit",
                feature_pool=WORKOUT_FEATURES + FITBIT_FEATURES,
                include_identity=False,
                train_df=tr_f,
                val_df=va_f,
                test_df=te_f,
            )
            details[exp5a["name"]] = exp5a
            details[exp5b["name"]] = exp5b

            results.append(
                row_from_metrics("Fitbit period workout only", "training (date>=2024-11-01)", exp5a["metrics"]["test"])
            )
            results.append(
                row_from_metrics("Fitbit period workout + Fitbit", "training + recovery (date>=2024-11-01)", exp5b["metrics"]["test"])
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(EXPERIMENTS_DIR / "model_C_ablation_results.csv", index=False)

    full_row = results_df[results_df["Model"] == "Full Model C"]
    if len(full_row) == 1: 
        c_full = float(full_row.iloc[0]["C-index"])
        ibs_full = float(full_row.iloc[0]["IBS"])
        mae_full = float(full_row.iloc[0]["MAE"])
        results_df["Delta_C-index_vs_Full_Model_C"] = results_df["C-index"] - c_full
        results_df["Delta_IBS_vs_Full_Model_C"] = results_df["IBS"] - ibs_full
        results_df["Delta_MAE_vs_Full_Model_C"] = results_df["MAE"] - mae_full
        results_df.to_csv(EXPERIMENTS_DIR / "model_C_ablation_results.csv", index=False)

    core_models = results_df[results_df["Model"].isin([
        "Baseline median",
        "Workout only",
        "Workout + exercise",
        "Workout + Fitbit",
        "Full Model C",
    ])].copy()

    best_c_row = core_models.loc[core_models["C-index"].idxmax()]
    best_ibs_row = core_models.loc[core_models["IBS"].idxmin()]
    best_mae_row = core_models.loc[core_models["MAE"].idxmin()]

    recommendation = str(best_ibs_row["Model"])
    reasoning = [
        f"Best test C-index: {best_c_row['Model']} ({best_c_row['C-index']:.4f}).",
        f"Best test IBS (lower is better): {best_ibs_row['Model']} ({best_ibs_row['IBS']:.4f}).",
        f"Best test MAE on uncensored events: {best_mae_row['Model']} ({best_mae_row['MAE']:.4f}).",
    ]

    if run_fitbit_period:
        reasoning.append(
            "Fitbit-period-only sensitivity analysis was run because at least one Fitbit feature had non-null coverage below 80%."
        )
    else:
        reasoning.append(
            "Fitbit-period-only sensitivity analysis was skipped because all Fitbit features had at least 80% non-null coverage."
        )

    summary = {
        "best_model_by_c_index": str(best_c_row["Model"]),
        "best_model_by_ibs": str(best_ibs_row["Model"]),
        "best_model_by_mae": str(best_mae_row["Model"]),
        "production_recommendation": recommendation,
        "reasoning": reasoning,
        "fitbit_missing_threshold": FITBIT_MISSING_THRESHOLD,
        "fitbit_period_start": FITBIT_START_DATE,
    }
    (EXPERIMENTS_DIR / "model_C_ablation_summary.json").write_text(json.dumps(summary, indent=2))

    # Save rich per-split metrics, calibration records, and feature sets for auditability.
    (EXPERIMENTS_DIR / "model_C_ablation_detailed_metrics.json").write_text(json.dumps(details, indent=2, default=str))

    print("Saved:")
    print("-", EXPERIMENTS_DIR / "model_C_feature_coverage.csv")
    print("-", EXPERIMENTS_DIR / "model_C_ablation_results.csv")
    print("-", EXPERIMENTS_DIR / "model_C_ablation_summary.json")
    print("-", EXPERIMENTS_DIR / "model_C_ablation_detailed_metrics.json")


if __name__ == "__main__":
    run()
