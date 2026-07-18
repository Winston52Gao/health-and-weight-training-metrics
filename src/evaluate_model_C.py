"""Evaluation and survival diagnostics for production Model C."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

try:
    from sksurv.metrics import integrated_brier_score
except ImportError:
    integrated_brier_score = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import (
    build_model_c_matrix,
    prepare_model_c_frame,
    WORKOUT_FEATURES,
)

MODELS = ROOT / "models"
MODEL_C_DIR = MODELS / "model_C"
REPORTS = ROOT / "reports"
MODEL_C_DIR.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(exist_ok=True)

BARRIER_HORIZONS = [5, 10, 20]


def construct_survival_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["exercise_title", "date"]).reset_index(drop=True)
    durations = np.zeros(len(df), dtype=float)
    events = np.zeros(len(df), dtype=bool)

    for _, group in df.groupby("exercise_title", sort=False):
        idx = group.index.to_numpy()
        is_pr = group["is_pr"].fillna(0).astype(int).to_numpy()
        pr_positions = np.where(is_pr == 1)[0]
        n_rows = len(group)
        for i in range(n_rows):
            future = pr_positions[pr_positions > i]
            if len(future) > 0:
                durations[idx[i]] = float(future[0] - i)
                events[idx[i]] = True
            else:
                durations[idx[i]] = float(max(1, n_rows - 1 - i))
                events[idx[i]] = False

    df["sessions_until_next_pr"] = durations
    df["event_observed"] = events.astype(int)
    return df


def time_splits(df: pd.DataFrame, train_end: str = "2025-06-30", val_end: str = "2025-12-31"):
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    train = frame[frame["date"] <= pd.to_datetime(train_end)].copy()
    val = frame[(frame["date"] > pd.to_datetime(train_end)) & (frame["date"] <= pd.to_datetime(val_end))].copy()
    test = frame[frame["date"] > pd.to_datetime(val_end)].copy()
    return train, val, test


def expected_sessions_from_survival(model, x: pd.DataFrame) -> np.ndarray:
    surv_fns = model.predict_survival_function(x, return_array=False)
    times = model.unique_times_
    delta = np.diff(np.r_[0.0, times])
    expected = []
    for fn in surv_fns:
        expected.append(float(np.sum(fn(times) * delta)))
    return np.array(expected)


def probability_within_n_sessions(model, x: pd.DataFrame, n_sessions: int) -> np.ndarray:
    surv_fns = model.predict_survival_function(x, return_array=False)
    return np.array([float(1.0 - float(fn(n_sessions))) for fn in surv_fns])


def first_session_meeting_threshold(curve: list[dict], threshold: float) -> int | None:
    for row in curve:
        if row["probability_of_pr"] >= threshold:
            return int(row["sessions_ahead"])
    return None


def split_survival_diagnostics(split_name: str, event: pd.Series, duration: pd.Series) -> dict:
    event_num = pd.to_numeric(pd.Series(event), errors="coerce")
    duration_num = pd.to_numeric(pd.Series(duration), errors="coerce")

    allowed_event_mask = event_num.isin([0, 1]).to_numpy()
    duration_values = duration_num.fillna(np.nan).to_numpy()
    finite_positive_duration_mask = np.isfinite(duration_values) & (duration_values > 0)
    valid_mask = allowed_event_mask & finite_positive_duration_mask

    valid_event = event_num[valid_mask].astype(int)
    valid_duration = duration_num[valid_mask].astype(float)

    n_valid = int(valid_mask.sum())
    n_events = int(valid_event.sum())
    event_rate = float(n_events / n_valid) if n_valid > 0 else np.nan

    return {
        "split": split_name,
        "n_observations": int(len(event_num)),
        "n_valid": n_valid,
        "n_events": n_events,
        "n_censored": int(n_valid - n_events),
        "event_rate": event_rate,
        "censoring_rate": float(1.0 - event_rate) if np.isfinite(event_rate) else np.nan,
        "min_duration": float(valid_duration.min()) if n_valid > 0 else np.nan,
        "max_duration": float(valid_duration.max()) if n_valid > 0 else np.nan,
        "mean_duration": float(valid_duration.mean()) if n_valid > 0 else np.nan,
        "median_duration": float(valid_duration.median()) if n_valid > 0 else np.nan,
        "missing_durations": int(duration_num.isna().sum()),
        "missing_events": int(event_num.isna().sum()),
        "invalid_event_values": int((~allowed_event_mask).sum()),
        "invalid_duration_values": int((~finite_positive_duration_mask).sum()),
        "valid_event_indicator": bool(int((~allowed_event_mask).sum()) == 0),
        "valid_duration_values": bool(int((~finite_positive_duration_mask).sum()) == 0),
        "valid_mask": valid_mask,
    }


def compute_integrated_brier_score(
    model,
    x_eval: pd.DataFrame,
    y_train_struct,
    event_eval: pd.Series,
    duration_eval: pd.Series,
    split_name: str,
) -> tuple[float, dict]:
    diag = split_survival_diagnostics(split_name, event_eval, duration_eval)
    report_lines = []
    reasons = []

    if integrated_brier_score is None:
        reasons.append("integrated_brier_score is unavailable in this scikit-survival build")

    train_time = y_train_struct["time"].astype(float)
    valid_mask = diag["valid_mask"]
    event_np = pd.to_numeric(pd.Series(event_eval), errors="coerce").to_numpy()[valid_mask].astype(int)
    duration_np = pd.to_numeric(pd.Series(duration_eval), errors="coerce").to_numpy()[valid_mask].astype(float)
    x_valid = x_eval.loc[valid_mask].copy()

    report_lines.append("[OK] valid event indicators" if diag["valid_event_indicator"] else f"[FAIL] invalid event indicators: {diag['invalid_event_values']}")
    report_lines.append("[OK] valid durations" if diag["valid_duration_values"] else f"[FAIL] invalid durations: {diag['invalid_duration_values']}")

    if diag["n_valid"] < 5:
        reasons.append(f"too few valid rows for IBS in {split_name} ({diag['n_valid']})")

    train_max = float(np.max(train_time)) if len(train_time) > 0 else np.nan
    eval_max = float(np.max(duration_np)) if len(duration_np) > 0 else np.nan
    eval_min = float(np.min(duration_np)) if len(duration_np) > 0 else np.nan

    if np.isfinite(train_max) and np.isfinite(eval_max) and eval_max > train_max:
        report_lines.append(f"[FAIL] follow-up overlap warning: eval max {eval_max:.3f} exceeds train max {train_max:.3f}")
    else:
        report_lines.append("[OK] valid follow-up overlap")

    times = model.unique_times_.astype(float)
    lower = max(float(np.min(train_time)) if len(train_time) > 0 else np.nan, eval_min)
    upper = min(train_max, eval_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        reasons.append(f"no valid overlap for evaluation times (lower={lower}, upper={upper})")
        grid = np.array([], dtype=float)
    else:
        grid = times[(times >= lower) & (times <= upper)]
        if len(grid) > 2:
            grid = np.unique(grid)[:-1]
        if len(grid) < 3:
            reasons.append(f"evaluation grid has fewer than 3 points after overlap filtering (n={len(grid)})")

    if len(grid) > 0:
        report_lines.append(f"[OK] valid evaluation grid: min={float(np.min(grid)):.3f}, max={float(np.max(grid)):.3f}, n={len(grid)}")
    else:
        report_lines.append("[FAIL] valid evaluation grid")

    y_eval_struct = Surv.from_arrays(event=event_np.astype(bool), time=duration_np)
    surv_fns = model.predict_survival_function(x_valid, return_array=False)
    surv_matrix = np.vstack([fn(grid) for fn in surv_fns]) if len(grid) > 0 else np.empty((len(x_valid), 0))

    if surv_matrix.shape != (len(x_valid), len(grid)):
        reasons.append(f"survival prediction matrix shape mismatch expected {(len(x_valid), len(grid))} got {surv_matrix.shape}")
        report_lines.append("[FAIL] valid prediction matrix")
    else:
        report_lines.append("[OK] valid prediction matrix")

    if surv_matrix.size > 0:
        min_p = float(np.nanmin(surv_matrix))
        max_p = float(np.nanmax(surv_matrix))
        if min_p < 0.0 or max_p > 1.0:
            reasons.append(f"survival probabilities outside [0,1] (min={min_p}, max={max_p})")
        if surv_matrix.shape[1] >= 2 and np.any(np.diff(surv_matrix, axis=1) > 1e-10):
            reasons.append("non-monotonic survival curves detected")
    else:
        reasons.append("empty survival prediction matrix")

    if len(reasons) == 0:
        report_lines.append("[OK] valid survival probabilities")
    else:
        report_lines.append("[FAIL] valid survival probabilities")

    diag_payload = {
        "ok": len(reasons) == 0,
        "reasons": reasons,
        "report_lines": report_lines,
        "grid": grid,
        "y_eval_struct": y_eval_struct,
        "surv_matrix": surv_matrix,
        "y_train_struct": y_train_struct,
        "eval_diag": {k: v for k, v in diag.items() if k != "valid_mask"},
        "train_max_follow_up": train_max,
        "eval_max_follow_up": eval_max,
    }

    if not diag_payload["ok"]:
        print(f"[IBS] not computable due to: {reasons}")
        return np.nan, diag_payload

    try:
        ibs = float(integrated_brier_score(diag_payload["y_train_struct"], diag_payload["y_eval_struct"], diag_payload["surv_matrix"], diag_payload["grid"]))
        return ibs, diag_payload
    except Exception as exc:
        try:
            times_out, brier_t = sksurv_brier_score(
                diag_payload["y_train_struct"],
                diag_payload["y_eval_struct"],
                diag_payload["surv_matrix"],
                diag_payload["grid"],
            )
            denom = float(times_out[-1] - times_out[0])
            if len(times_out) >= 2 and np.all(np.isfinite(brier_t)) and denom > 0:
                if hasattr(np, "trapezoid"):
                    ibs_fb = float(np.trapezoid(brier_t, times_out) / denom)
                else:
                    ibs_fb = float(np.trapz(brier_t, times_out) / denom)
                print("[IBS] computed via validated fallback integration of Brier(t).")
                return ibs_fb, diag_payload
        except Exception:
            pass
        print(f"[IBS] not computable: {type(exc).__name__}: {exc}")
        diag_payload["ok"] = False
        diag_payload["reasons"].append(f"integrated_brier_score failed with exception: {type(exc).__name__}: {exc}")
        return np.nan, diag_payload


def evaluate_split(model, x: pd.DataFrame, event: pd.Series, duration: pd.Series, horizons: list[int], y_train_struct, split_name: str) -> dict:
    expected = expected_sessions_from_survival(model, x)
    event_bool = event.astype(bool).to_numpy()
    duration_np = duration.astype(float).to_numpy()

    c_index = float(concordance_index_censored(event_bool, duration_np, -expected)[0])
    mae = float(np.mean(np.abs(expected[event_bool] - duration_np[event_bool]))) if event_bool.any() else np.nan
    ibs, ibs_diag = compute_integrated_brier_score(model, x, y_train_struct, event, duration, split_name)

    brier_by_horizon = {}
    calibration_by_horizon = {}
    for horizon in horizons:
        probs_h = probability_within_n_sessions(model, x, horizon)
        known_mask = ~((~event_bool) & (duration_np <= horizon))
        if known_mask.sum() < 20:
            brier_by_horizon[str(horizon)] = np.nan
            calibration_by_horizon[str(horizon)] = []
            continue

        y_h = (event_bool & (duration_np <= horizon)).astype(int)
        y_known = y_h[known_mask]
        p_known = probs_h[known_mask]
        brier_by_horizon[str(horizon)] = float(brier_score_loss(y_known, p_known))

        cal_df = pd.DataFrame({"y": y_known, "p": p_known})
        cal_df["bin"] = pd.qcut(cal_df["p"], q=min(5, max(2, cal_df["p"].nunique())), duplicates="drop")
        cal = (
            cal_df.groupby("bin", observed=True)
            .agg(predicted_prob=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
        )
        calibration_by_horizon[str(horizon)] = cal.to_dict(orient="records")

    survival_diag = split_survival_diagnostics(split_name, event, duration)
    survival_diag.pop("valid_mask", None)

    return {
        "c_index": c_index,
        "ibs": ibs,
        "mae_uncensored_sessions_diagnostic": mae,
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
        "survival_diagnostics": survival_diag,
    }


def run() -> None:
    import joblib

    payload = joblib.load(MODEL_C_DIR / "model_C_rsf_survival.joblib")
    model = payload["model"]
    features = payload["features"]
    feature_columns = payload["feature_columns"]

    df = prepare_model_c_frame()
    df = construct_survival_labels(df)
    train, val, test = time_splits(df)

    x_train = build_model_c_matrix(train, features, ref_columns=feature_columns)
    x_val = build_model_c_matrix(val, features, ref_columns=feature_columns)
    x_test = build_model_c_matrix(test, features, ref_columns=feature_columns)

    y_train = Surv.from_arrays(
        event=train["event_observed"].astype(bool).to_numpy(),
        time=train["sessions_until_next_pr"].astype(float).to_numpy(),
    )

    metrics = {
        "train": evaluate_split(model, x_train, train["event_observed"], train["sessions_until_next_pr"], BARRIER_HORIZONS, y_train, "train"),
        "val": evaluate_split(model, x_val, val["event_observed"], val["sessions_until_next_pr"], BARRIER_HORIZONS, y_train, "validation"),
        "test": evaluate_split(model, x_test, test["event_observed"], test["sessions_until_next_pr"], BARRIER_HORIZONS, y_train, "test"),
    }

    (MODEL_C_DIR / "model_C_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved:", MODEL_C_DIR / "model_C_metrics.json")


if __name__ == "__main__":
    run()
