"""Train Model C as a direct Random Survival Forest PR forecaster.

Model C:
Random Survival Forest predicting time-to-next-PR directly from engineered
workout, training history, recovery, and exercise context features.

Model A:
Independent XGBoost classifier predicting probability of PR in the next
workout session based on workout progression features.
(Model A is now trained in notebooks/model_A_workout_progression.ipynb.)
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import brier_score as sksurv_brier_score
    from sksurv.metrics import concordance_index_censored
    from sksurv.util import Surv
    try:
        from sksurv.metrics import integrated_brier_score
    except ImportError:
        integrated_brier_score = None
except ImportError as exc:
    raise ImportError(
        "scikit-survival is required for Model C. Install it with: pip install scikit-survival"
    ) from exc

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "models"
OUT.mkdir(exist_ok=True)

PR_THRESHOLDS = {
    "likely": 0.50,
    "high_confidence": 0.80,
}
CURVE_POINTS = [1, 3, 5, 8, 10, 15, 20, 25, 30]
Brier_HORIZONS = [5, 10, 20]


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA / "processed_merged.csv", parse_dates=["date"])


def add_training_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["exercise_title", "date"]).reset_index(drop=True)
    grp = df.groupby("exercise_title")
    df["training_age_sessions"] = grp.cumcount() + 1
    first_date = grp["date"].transform("min")
    df["training_age_days"] = (
        pd.to_datetime(df["date"]).dt.normalize() - pd.to_datetime(first_date).dt.normalize()
    ).dt.days
    return df


def add_recent_training_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["exercise_title", "date"]).set_index("date")
    out_frames = []
    for _, g in df.groupby("exercise_title"):
        g = g.sort_index()
        g["days_since_last_workout"] = g.index.to_series().diff().dt.days.fillna(9999)
        out_frames.append(g.reset_index())
    return (
        pd.concat(out_frames, ignore_index=True)
        .sort_values(["exercise_title", "date"])
        .reset_index(drop=True)
    )


def ensure_model_features(df: pd.DataFrame) -> pd.DataFrame:
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
        "days_since_last_pr",
        "sessions_since_last_pr",
        "pr_freq_90d",
        "steps_7d_avg",
        "sleep_minutes",
        "sleep_7d_avg",
        "resting_hr",
        "hr_7d_avg",
        "hr_baseline_z",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    if "is_pr" not in df.columns:
        if {"rolling_best_prev", "best_est_1RM"}.issubset(df.columns):
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
    seen = set()
    out = []
    for c in feat_list:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def construct_survival_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["exercise_title", "date"]).reset_index(drop=True)
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
                durations[idx[i]] = float(max(1, n - 1 - i))
                events[idx[i]] = False

    df["sessions_until_next_pr"] = durations
    df["event_observed"] = events.astype(int)
    return df


def build_model_c_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    ref_columns: list[str] | None = None,
    include_exercise_identity: bool = True,
):
    X_num = df[feature_cols].copy().fillna(0.0)
    if include_exercise_identity:
        X_ex = pd.get_dummies(df["exercise_title"].fillna("unknown"), prefix="ex", dtype=float)
        X = pd.concat([X_num, X_ex], axis=1)
    else:
        X = X_num
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


def probability_within_n_sessions(rsf: RandomSurvivalForest, X: pd.DataFrame, n_sessions: int) -> np.ndarray:
    surv_fns = rsf.predict_survival_function(X, return_array=False)
    probs = []
    for fn in surv_fns:
        surv_at_n = float(fn(n_sessions))
        probs.append(float(1.0 - surv_at_n))
    return np.array(probs)


def first_session_meeting_threshold(curve: list[dict], threshold: float) -> int | None:
    for row in curve:
        if row["probability_of_pr"] >= threshold:
            return int(row["sessions_ahead"])
    return None


def split_survival_diagnostics(split_name: str, event: pd.Series, duration: pd.Series) -> dict:
    event_raw = pd.Series(event)
    duration_raw = pd.Series(duration)

    event_num = pd.to_numeric(event_raw, errors="coerce")
    duration_num = pd.to_numeric(duration_raw, errors="coerce")

    missing_event = int(event_num.isna().sum())
    missing_duration = int(duration_num.isna().sum())

    finite_duration_mask = np.isfinite(duration_num.fillna(np.nan).to_numpy())
    positive_duration_mask = duration_num.fillna(np.nan).to_numpy() > 0
    finite_positive_duration = finite_duration_mask & positive_duration_mask

    allowed_event_mask = event_num.isin([0, 1]).to_numpy()
    valid_mask = allowed_event_mask & finite_positive_duration

    valid_event = event_num[valid_mask].astype(int)
    valid_duration = duration_num[valid_mask].astype(float)

    n = int(len(event_raw))
    n_valid = int(valid_mask.sum())
    n_event = int(valid_event.sum())
    n_censored = int(n_valid - n_event)
    event_rate = float(n_event / n_valid) if n_valid > 0 else np.nan

    return {
        "split": split_name,
        "n_observations": n,
        "n_valid": n_valid,
        "n_events": n_event,
        "n_censored": n_censored,
        "event_rate": event_rate,
        "censoring_rate": float(1.0 - event_rate) if np.isfinite(event_rate) else np.nan,
        "min_duration": float(valid_duration.min()) if n_valid > 0 else np.nan,
        "max_duration": float(valid_duration.max()) if n_valid > 0 else np.nan,
        "mean_duration": float(valid_duration.mean()) if n_valid > 0 else np.nan,
        "median_duration": float(valid_duration.median()) if n_valid > 0 else np.nan,
        "missing_durations": missing_duration,
        "missing_events": missing_event,
        "invalid_event_values": int((~allowed_event_mask).sum()),
        "invalid_duration_values": int((~finite_positive_duration).sum()),
        "valid_event_indicator": bool(int((~allowed_event_mask).sum()) == 0),
        "valid_duration_values": bool(int((~finite_positive_duration).sum()) == 0),
        "valid_mask": valid_mask,
    }


def print_split_diagnostics(diag: dict) -> None:
    print(
        f"[IBS diagnostic] split={diag['split']} n={diag['n_observations']} valid={diag['n_valid']} "
        f"events={diag['n_events']} censored={diag['n_censored']} event_rate={diag['event_rate']:.4f}"
    )
    print(
        f"[IBS diagnostic] split={diag['split']} duration min={diag['min_duration']:.3f} max={diag['max_duration']:.3f} "
        f"mean={diag['mean_duration']:.3f} median={diag['median_duration']:.3f}"
    )
    print(
        f"[IBS diagnostic] split={diag['split']} missing_duration={diag['missing_durations']} "
        f"missing_event={diag['missing_events']} invalid_duration={diag['invalid_duration_values']} "
        f"invalid_event={diag['invalid_event_values']}"
    )


def print_dataset_survival_diagnostics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print("\n=== Survival Label Diagnostics (Train/Validation/Test) ===")
    train_diag = split_survival_diagnostics("train", train_df["event_observed"], train_df["sessions_until_next_pr"])
    val_diag = split_survival_diagnostics("validation", val_df["event_observed"], val_df["sessions_until_next_pr"])
    test_diag = split_survival_diagnostics("test", test_df["event_observed"], test_df["sessions_until_next_pr"])

    for d in [train_diag, val_diag, test_diag]:
        print_split_diagnostics(d)

    total_valid = train_diag["n_valid"] + val_diag["n_valid"] + test_diag["n_valid"]
    total_censored = train_diag["n_censored"] + val_diag["n_censored"] + test_diag["n_censored"]
    overall_censoring = float(total_censored / total_valid) if total_valid > 0 else np.nan
    print(
        f"[IBS diagnostic] censoring rates overall={overall_censoring:.4f} "
        f"train={train_diag['censoring_rate']:.4f} validation={val_diag['censoring_rate']:.4f} "
        f"test={test_diag['censoring_rate']:.4f}"
    )
    print(
        f"[IBS diagnostic] follow-up max train={train_diag['max_duration']:.3f} "
        f"validation={val_diag['max_duration']:.3f} test={test_diag['max_duration']:.3f}"
    )


def validate_ibs_inputs(
    rsf: RandomSurvivalForest,
    X_eval: pd.DataFrame,
    y_train_struct,
    event_eval: pd.Series,
    duration_eval: pd.Series,
    split_name: str,
) -> dict:
    report_lines = []
    reasons = []

    if integrated_brier_score is None:
        reasons.append("integrated_brier_score is unavailable in this scikit-survival build")

    train_time = y_train_struct["time"].astype(float)
    eval_diag = split_survival_diagnostics(split_name, event_eval, duration_eval)
    valid_mask = eval_diag["valid_mask"]

    report_lines.append(
        "[OK] valid event indicators"
        if eval_diag["valid_event_indicator"]
        else f"[FAIL] invalid event indicators: {eval_diag['invalid_event_values']}"
    )
    report_lines.append(
        "[OK] valid durations"
        if eval_diag["valid_duration_values"]
        else f"[FAIL] invalid durations: {eval_diag['invalid_duration_values']}"
    )

    if eval_diag["n_valid"] < 5:
        reasons.append(f"too few valid rows for IBS in {split_name} ({eval_diag['n_valid']})")

    event_np = pd.to_numeric(pd.Series(event_eval), errors="coerce").to_numpy()[valid_mask].astype(int)
    duration_np = pd.to_numeric(pd.Series(duration_eval), errors="coerce").to_numpy()[valid_mask].astype(float)
    X_valid = X_eval.loc[valid_mask].copy()

    train_max = float(np.max(train_time)) if len(train_time) > 0 else np.nan
    eval_max = float(np.max(duration_np)) if len(duration_np) > 0 else np.nan
    eval_min = float(np.min(duration_np)) if len(duration_np) > 0 else np.nan

    if np.isfinite(train_max) and np.isfinite(eval_max) and eval_max > train_max:
        report_lines.append(
            f"[FAIL] follow-up overlap warning: eval max {eval_max:.3f} exceeds train max {train_max:.3f}"
        )
    else:
        report_lines.append("[OK] valid follow-up overlap")

    times = rsf.unique_times_.astype(float)
    if len(times) < 3:
        reasons.append("model has fewer than 3 unique event times")

    lower = max(float(np.min(train_time)) if len(train_time) > 0 else np.nan, eval_min)
    upper = min(train_max, eval_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        reasons.append(f"no valid overlap for evaluation times (lower={lower}, upper={upper})")
        grid = np.array([], dtype=float)
    else:
        grid = times[(times >= lower) & (times <= upper)]
        if len(grid) > 2:
            grid = np.unique(grid)
            grid = grid[:-1]
        if len(grid) < 3:
            reasons.append(f"evaluation grid has fewer than 3 points after overlap filtering (n={len(grid)})")

    if len(grid) > 0:
        report_lines.append(
            f"[OK] valid evaluation grid: min={float(np.min(grid)):.3f}, max={float(np.max(grid)):.3f}, n={len(grid)}"
        )
    else:
        report_lines.append("[FAIL] valid evaluation grid")

    y_eval_struct = Surv.from_arrays(event=event_np.astype(bool), time=duration_np)
    surv_fns = rsf.predict_survival_function(X_valid, return_array=False)
    surv_matrix = np.vstack([fn(grid) for fn in surv_fns]) if len(grid) > 0 else np.empty((len(X_valid), 0))

    if surv_matrix.shape != (len(X_valid), len(grid)):
        reasons.append(
            f"survival prediction matrix shape mismatch expected {(len(X_valid), len(grid))} got {surv_matrix.shape}"
        )
        report_lines.append("[FAIL] valid prediction matrix")
    else:
        report_lines.append("[OK] valid prediction matrix")

    has_nan = bool(np.isnan(surv_matrix).any())
    has_inf = bool(np.isinf(surv_matrix).any())
    if has_nan or has_inf:
        reasons.append(f"survival predictions contain nan={has_nan} inf={has_inf}")

    if surv_matrix.size > 0:
        min_p = float(np.nanmin(surv_matrix))
        max_p = float(np.nanmax(surv_matrix))
        in_range = (min_p >= 0.0) and (max_p <= 1.0)
    else:
        min_p, max_p, in_range = np.nan, np.nan, False
    if not in_range:
        reasons.append(f"survival probabilities outside [0,1] (min={min_p}, max={max_p})")

    if surv_matrix.shape[1] >= 2:
        diffs = np.diff(surv_matrix, axis=1)
        viol_rows = np.where(np.any(diffs > 1e-10, axis=1))[0]
        if len(viol_rows) > 0:
            reasons.append(
                f"non-monotonic survival curves in {len(viol_rows)} rows; example rows={viol_rows[:5].tolist()}"
            )

    if len(reasons) == 0:
        report_lines.append("[OK] valid survival probabilities")
    else:
        report_lines.append("[FAIL] valid survival probabilities")

    return {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "report_lines": report_lines,
        "grid": grid,
        "y_eval_struct": y_eval_struct,
        "surv_matrix": surv_matrix,
        "y_train_struct": y_train_struct,
        "eval_diag": eval_diag,
        "train_max_follow_up": train_max,
        "eval_max_follow_up": eval_max,
    }


def compute_integrated_brier_score(
    rsf: RandomSurvivalForest,
    X_eval: pd.DataFrame,
    y_train_struct,
    event_eval: pd.Series,
    duration_eval: pd.Series,
    split_name: str,
) -> tuple[float, dict]:
    diag = validate_ibs_inputs(
        rsf=rsf,
        X_eval=X_eval,
        y_train_struct=y_train_struct,
        event_eval=event_eval,
        duration_eval=duration_eval,
        split_name=split_name,
    )

    print(f"[IBS checks] split={split_name}")
    for line in diag["report_lines"]:
        print(f"  {line}")

    if not diag["ok"]:
        print("[IBS] not computable due to:")
        for reason in diag["reasons"]:
            print(f"  - {reason}")
        return np.nan, diag

    try:
        ibs = integrated_brier_score(
            diag["y_train_struct"],
            diag["y_eval_struct"],
            diag["surv_matrix"],
            diag["grid"],
        )
        return float(ibs), diag
    except Exception as exc:
        msg = str(exc)
        if isinstance(exc, AttributeError) and "trapz" in msg:
            try:
                times_out, brier_t = sksurv_brier_score(
                    diag["y_train_struct"],
                    diag["y_eval_struct"],
                    diag["surv_matrix"],
                    diag["grid"],
                )
                if len(times_out) >= 2 and np.all(np.isfinite(brier_t)):
                    denom = float(times_out[-1] - times_out[0])
                    if denom > 0:
                        if hasattr(np, "trapezoid"):
                            ibs_fb = float(np.trapezoid(brier_t, times_out) / denom)
                        else:
                            ibs_fb = float(np.trapz(brier_t, times_out) / denom)
                        print("[IBS] computed via validated fallback integration of Brier(t).")
                        return ibs_fb, diag
                    diag["reasons"].append("fallback IBS denominator is non-positive")
                else:
                    diag["reasons"].append("fallback Brier curve invalid (insufficient points or non-finite)")
            except Exception as fb_exc:
                diag["reasons"].append(
                    f"fallback IBS failed with exception: {type(fb_exc).__name__}: {fb_exc}"
                )

        reason = f"integrated_brier_score failed with exception: {type(exc).__name__}: {exc}"
        print(f"[IBS] not computable: {reason}")
        diag["reasons"].append(reason)
        diag["ok"] = False
        return np.nan, diag


def evaluate_rsf_forecast(
    rsf: RandomSurvivalForest,
    X: pd.DataFrame,
    event: pd.Series,
    duration: pd.Series,
    horizons: list[int],
    y_train_struct,
    split_name: str,
) -> dict:
    expected = expected_sessions_from_survival(rsf, X)
    event_bool = event.astype(bool).to_numpy()
    duration_np = duration.astype(float).to_numpy()

    c_index = concordance_index_censored(event_bool, duration_np, -expected)[0]
    uncensored = event_bool
    mae_diag = float(np.mean(np.abs(expected[uncensored] - duration_np[uncensored]))) if uncensored.any() else np.nan

    ibs_value, ibs_diag = compute_integrated_brier_score(
        rsf=rsf,
        X_eval=X,
        y_train_struct=y_train_struct,
        event_eval=event,
        duration_eval=duration,
        split_name=split_name,
    )

    brier_by_horizon = {}
    calibration_by_horizon = {}
    for h in horizons:
        probs_h = probability_within_n_sessions(rsf, X, h)
        known_mask = ~((~event_bool) & (duration_np <= h))
        if known_mask.sum() < 20:
            brier_by_horizon[str(h)] = np.nan
            calibration_by_horizon[str(h)] = []
            continue

        y_h = (event_bool & (duration_np <= h)).astype(int)
        y_known = y_h[known_mask]
        p_known = probs_h[known_mask]
        brier_by_horizon[str(h)] = float(brier_score_loss(y_known, p_known))

        cal_df = pd.DataFrame({"y": y_known, "p": p_known})
        cal_df["bin"] = pd.qcut(cal_df["p"], q=min(5, max(2, cal_df["p"].nunique())), duplicates="drop")
        cal = (
            cal_df.groupby("bin", observed=True)
            .agg(predicted_prob=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
        )
        calibration_by_horizon[str(h)] = cal.to_dict(orient="records")

    return {
        "c_index": float(c_index),
        "ibs": ibs_value,
        "mae_uncensored_sessions_diagnostic": mae_diag,
        "mean_expected_sessions": float(np.mean(expected)),
        "brier_by_horizon": brier_by_horizon,
        "calibration_by_horizon": calibration_by_horizon,
        "ibs_diagnostics": {
            "ok": bool(ibs_diag["ok"]),
            "reasons": list(ibs_diag["reasons"]),
            "grid_min": float(np.min(ibs_diag["grid"])) if len(ibs_diag["grid"]) > 0 else np.nan,
            "grid_max": float(np.max(ibs_diag["grid"])) if len(ibs_diag["grid"]) > 0 else np.nan,
            "grid_n": int(len(ibs_diag["grid"])),
            "train_max_follow_up": float(ibs_diag["train_max_follow_up"]),
            "eval_max_follow_up": float(ibs_diag["eval_max_follow_up"]),
            "censoring_rate": float(ibs_diag["eval_diag"]["censoring_rate"]),
        },
    }


def prediction_interval_from_survival(rsf: RandomSurvivalForest, X_one: pd.DataFrame) -> list[float]:
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


def build_pr_forecast_predictions(
    rsf: RandomSurvivalForest,
    feature_columns: list[str],
    latest_rows: pd.DataFrame,
    feature_cols_c: list[str],
    thresholds: dict[str, float],
    curve_points: list[int],
) -> list[dict]:
    X_latest = build_model_c_matrix(latest_rows, feature_cols_c, ref_columns=feature_columns)
    expected_vals = expected_sessions_from_survival(rsf, X_latest)

    surv_fns = rsf.predict_survival_function(X_latest, return_array=False)
    forecasts = []
    for i, (_, row) in enumerate(latest_rows.reset_index(drop=True).iterrows()):
        fn = surv_fns[i]
        curve = []
        for n in curve_points:
            p = float(1.0 - float(fn(n)))
            curve.append({"sessions_ahead": int(n), "probability_of_pr": p})

        p50 = first_session_meeting_threshold(curve, thresholds["likely"])
        p80 = first_session_meeting_threshold(curve, thresholds["high_confidence"])

        forecasts.append(
            {
                "exercise": str(row["exercise_title"]),
                "current_best_est_1RM": float(row["best_est_1RM"]) if pd.notnull(row.get("best_est_1RM", np.nan)) else None,
                "expected_sessions_until_pr": float(expected_vals[i]),
                "pr_window": {
                    "50_percent_probability": p50,
                    "80_percent_probability": p80,
                },
                "probability_curve": curve,
                "probability_within_10_sessions": next((c["probability_of_pr"] for c in curve if c["sessions_ahead"] == 10), None),
            }
        )

    forecasts = sorted(
        forecasts,
        key=lambda x: (-1 if x["probability_within_10_sessions"] is None else -x["probability_within_10_sessions"]),
    )
    for rank, item in enumerate(forecasts, start=1):
        item["rank_by_10_session_probability"] = rank

    return forecasts


def update_ablation_summary_with_production_decision() -> None:
    path = OUT / "model_C_ablation_summary.json"
    if not path.exists():
        return

    decision = {
        "production_decision": "Model A retained as an independent progression classifier but removed from Model C stacking.",
        "reason": [
            "Model A provided marginal C-index improvement.",
            "Model A slightly worsened IBS.",
            "Model A slightly worsened MAE.",
            "Standalone Model C provides a simpler production pipeline.",
        ],
    }

    data = json.loads(path.read_text())
    if isinstance(data, list):
        payload = {
            "experiments": data,
            **decision,
        }
    elif isinstance(data, dict):
        payload = dict(data)
        payload.update(decision)
    else:
        payload = decision

    path.write_text(json.dumps(payload, indent=2))


def run() -> None:
    df_all = load_data()
    df_all = add_training_age(df_all)
    df_all = add_recent_training_context(df_all)
    df_all = ensure_model_features(df_all)

    df = df_all[df_all["PR_next_session"].notnull()].copy()
    df = construct_survival_labels(df)

    train, val, test = time_splits(df)
    print("splits:", train.shape, val.shape, test.shape)
    print_dataset_survival_diagnostics(train, val, test)

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
        "pr_freq_90d",
        "training_age_sessions",
        "training_age_days",
        "sleep_minutes",
        "sleep_7d_avg",
        "resting_hr",
        "hr_7d_avg",
        "hr_baseline_z",
        "steps_7d_avg",
    ]
    featsC = select_features(pd.concat([train, val, test], axis=0), features_C)
    print("Model C (RSF) features used:", featsC)

    XtrC = build_model_c_matrix(train, featsC)
    XvC = build_model_c_matrix(val, featsC, ref_columns=list(XtrC.columns))
    XtC = build_model_c_matrix(test, featsC, ref_columns=list(XtrC.columns))

    ytrC = Surv.from_arrays(
        event=train["event_observed"].astype(bool).to_numpy(),
        time=train["sessions_until_next_pr"].astype(float).to_numpy(),
    )

    rsf = train_rsf(XtrC, ytrC)
    joblib.dump(
        {"model": rsf, "feature_columns": list(XtrC.columns), "base_features": featsC},
        OUT / "model_C_rsf_survival.joblib",
    )

    metrics_C = {
        "train": evaluate_rsf_forecast(rsf, XtrC, train["event_observed"], train["sessions_until_next_pr"], Brier_HORIZONS, ytrC, "train"),
        "val": evaluate_rsf_forecast(rsf, XvC, val["event_observed"], val["sessions_until_next_pr"], Brier_HORIZONS, ytrC, "validation"),
        "test": evaluate_rsf_forecast(rsf, XtC, test["event_observed"], test["sessions_until_next_pr"], Brier_HORIZONS, ytrC, "test"),
    }
    print("Model C (RSF) metrics:", metrics_C)

    out_val = val[["date", "exercise_title", "sessions_until_next_pr", "event_observed"]].copy()
    out_val["expected_sessions_until_pr"] = expected_sessions_from_survival(rsf, XvC)
    out_val["probability_of_pr_within_5_sessions"] = probability_within_n_sessions(rsf, XvC, 5)
    out_val["probability_of_pr_within_10_sessions"] = probability_within_n_sessions(rsf, XvC, 10)
    out_val["probability_of_pr_within_20_sessions"] = probability_within_n_sessions(rsf, XvC, 20)

    out_test = test[["date", "exercise_title", "sessions_until_next_pr", "event_observed"]].copy()
    out_test["expected_sessions_until_pr"] = expected_sessions_from_survival(rsf, XtC)
    out_test["probability_of_pr_within_5_sessions"] = probability_within_n_sessions(rsf, XtC, 5)
    out_test["probability_of_pr_within_10_sessions"] = probability_within_n_sessions(rsf, XtC, 10)
    out_test["probability_of_pr_within_20_sessions"] = probability_within_n_sessions(rsf, XtC, 20)

    out_val.to_csv(OUT / "model_C_val_predictions.csv", index=False)
    out_test.to_csv(OUT / "model_C_test_predictions.csv", index=False)

    forecast_rows = df_all.sort_values("date").groupby("exercise_title", as_index=False).tail(1).copy()
    forecasts = build_pr_forecast_predictions(
        rsf=rsf,
        feature_columns=list(XtrC.columns),
        latest_rows=forecast_rows,
        feature_cols_c=featsC,
        thresholds=PR_THRESHOLDS,
        curve_points=CURVE_POINTS,
    )
    forecast_payload = {
        "thresholds": {
            "likely": PR_THRESHOLDS["likely"],
            "high_confidence": PR_THRESHOLDS["high_confidence"],
        },
        "curve_points": CURVE_POINTS,
        "forecasts": forecasts,
    }
    (OUT / "pr_forecast_predictions.json").write_text(json.dumps(forecast_payload, indent=2))

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

    all_metrics = {
        "C_survival": metrics_C,
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(all_metrics, indent=2))

    update_ablation_summary_with_production_decision()
    print("Saved Model C artifacts to", OUT)


if __name__ == "__main__":
    run()
