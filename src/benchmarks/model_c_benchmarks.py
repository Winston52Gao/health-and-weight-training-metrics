"""Model C benchmark experiments.

Benchmarks implemented:
1) Heuristic probability curve (mean gap and median gap variants)
2) Kaplan-Meier baseline
3) Production RSF comparison metrics

Outputs are written to models/benchmarks/.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sksurv.metrics import brier_score as sksurv_brier_score
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

try:
    from sksurv.metrics import integrated_brier_score
except ImportError:
    integrated_brier_score = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate_model_C import (
    construct_survival_labels,
    evaluate_split,
    expected_sessions_from_survival,
    first_session_meeting_threshold,
    probability_within_n_sessions,
    time_splits,
)
from src.feature_engineering import PRODUCTION_FEATURES, build_model_c_matrix, prepare_model_c_frame
from src.train_model_C import train_rsf

MODEL_C_DIR = ROOT / "models" / "model_C"
BENCHMARK_DIR = ROOT / "models" / "benchmarks"
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

CURVE_POINTS = [1, 3, 5, 8, 10, 15, 20, 25, 30]
HORIZONS = [5, 10, 20]


@dataclass
class HeuristicStats:
    exercise: str
    avg_sessions_between_prs: float
    median_sessions_between_prs: float
    n_historical_prs: int
    total_sessions: int


def _safe_float(value, fallback: float) -> float:
    try:
        if value is None or pd.isna(value):
            return float(fallback)
        return float(value)
    except Exception:
        return float(fallback)


def _build_eval_grid(train_times: np.ndarray, eval_times: np.ndarray) -> np.ndarray:
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


def _compute_ibs_generic(y_train_struct, y_eval_struct, surv_matrix: np.ndarray, grid: np.ndarray) -> float:
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


def _compute_pr_gap_stats(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    all_gaps = []

    for exercise, group in train_df.sort_values(["exercise_title", "date"]).groupby("exercise_title", sort=False):
        pr_positions = np.where(group["is_pr"].fillna(0).astype(int).to_numpy() == 1)[0]
        if len(pr_positions) >= 2:
            gaps = np.diff(pr_positions).astype(float)
            avg_gap = float(np.mean(gaps))
            med_gap = float(np.median(gaps))
            all_gaps.extend(gaps.tolist())
        else:
            avg_gap = np.nan
            med_gap = np.nan

        rows.append(
            HeuristicStats(
                exercise=str(exercise),
                avg_sessions_between_prs=avg_gap,
                median_sessions_between_prs=med_gap,
                n_historical_prs=int(len(pr_positions)),
                total_sessions=int(len(group)),
            ).__dict__
        )

    stats_df = pd.DataFrame(rows)
    global_avg = float(np.mean(all_gaps)) if all_gaps else 10.0
    global_med = float(np.median(all_gaps)) if all_gaps else 10.0

    stats_df["avg_sessions_between_prs"] = stats_df["avg_sessions_between_prs"].fillna(global_avg)
    stats_df["median_sessions_between_prs"] = stats_df["median_sessions_between_prs"].fillna(global_med)

    globals_payload = {
        "global_avg_sessions_between_prs": global_avg,
        "global_median_sessions_between_prs": global_med,
    }
    return stats_df, globals_payload


def _with_heuristic_stats(df: pd.DataFrame, stats_df: pd.DataFrame, globals_payload: dict[str, float]) -> pd.DataFrame:
    merged = df.merge(stats_df, left_on="exercise_title", right_on="exercise", how="left")
    merged["avg_sessions_between_prs"] = merged["avg_sessions_between_prs"].fillna(globals_payload["global_avg_sessions_between_prs"])
    merged["median_sessions_between_prs"] = merged["median_sessions_between_prs"].fillna(globals_payload["global_median_sessions_between_prs"])
    merged["sessions_since_last_pr"] = pd.to_numeric(merged.get("sessions_since_last_pr"), errors="coerce").fillna(0.0)
    return merged


def _heuristic_prob_within_n(sessions_since_last_pr: np.ndarray, gap: np.ndarray, n: float) -> np.ndarray:
    gap = np.where(gap <= 0, 1.0, gap)
    probs = (sessions_since_last_pr + n) / gap
    return np.clip(probs, 0.0, 1.0)


def _heuristic_expected_sessions(sessions_since_last_pr: np.ndarray, gap: np.ndarray) -> np.ndarray:
    gap = np.where(gap <= 0, 1.0, gap)
    expected = np.maximum(gap - sessions_since_last_pr, 1.0)
    return expected.astype(float)


def _heuristic_survival_matrix(sessions_since_last_pr: np.ndarray, gap: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if len(grid) == 0:
        return np.empty((len(sessions_since_last_pr), 0))
    probs = np.vstack([_heuristic_prob_within_n(sessions_since_last_pr, gap, t) for t in grid]).T
    return np.clip(1.0 - probs, 0.0, 1.0)


def _horizon_brier_metrics(event_bool: np.ndarray, duration_np: np.ndarray, horizon_probs: dict[int, np.ndarray]) -> tuple[dict[str, float], dict[str, list[dict]]]:
    brier_by_horizon = {}
    calibration_by_horizon = {}

    for h, probs in horizon_probs.items():
        known_mask = ~((~event_bool) & (duration_np <= h))
        if known_mask.sum() < 20:
            brier_by_horizon[str(h)] = np.nan
            calibration_by_horizon[str(h)] = []
            continue

        y_h = (event_bool & (duration_np <= h)).astype(int)
        y_known = y_h[known_mask]
        p_known = probs[known_mask]
        brier_by_horizon[str(h)] = float(brier_score_loss(y_known, p_known))

        cal_df = pd.DataFrame({"y": y_known, "p": p_known})
        cal_df["bin"] = pd.qcut(cal_df["p"], q=min(5, max(2, cal_df["p"].nunique())), duplicates="drop")
        cal = (
            cal_df.groupby("bin", observed=True)
            .agg(predicted_prob=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True)
        )
        calibration_by_horizon[str(h)] = cal.to_dict(orient="records")

    return brier_by_horizon, calibration_by_horizon


def _evaluate_generic_split(
    expected_sessions: np.ndarray,
    horizon_probs: dict[int, np.ndarray],
    surv_matrix: np.ndarray,
    y_train_struct,
    event_eval: pd.Series,
    duration_eval: pd.Series,
) -> dict:
    event_bool = event_eval.astype(bool).to_numpy()
    duration_np = duration_eval.astype(float).to_numpy()

    c_index = float(concordance_index_censored(event_bool, duration_np, -expected_sessions)[0])
    mae = float(np.mean(np.abs(expected_sessions[event_bool] - duration_np[event_bool]))) if event_bool.any() else np.nan

    y_eval_struct = Surv.from_arrays(event=event_bool, time=duration_np)

    grid = _build_eval_grid(y_train_struct["time"], duration_np)
    if len(grid) > 0:
        if surv_matrix.shape[1] != len(grid):
            # Rebuild on compatible grid by interpolation from horizon function if needed.
            # For generic benchmarks this branch should rarely execute.
            surv_matrix_use = np.empty((len(duration_np), len(grid)))
            for i in range(len(duration_np)):
                surv_matrix_use[i, :] = np.clip(surv_matrix[i, :len(grid)], 0.0, 1.0)
        else:
            surv_matrix_use = surv_matrix
    else:
        surv_matrix_use = np.empty((len(duration_np), 0))

    ibs = _compute_ibs_generic(y_train_struct, y_eval_struct, surv_matrix_use, grid)
    brier_by_horizon, calibration_by_horizon = _horizon_brier_metrics(event_bool, duration_np, horizon_probs)

    return {
        "c_index": c_index,
        "ibs": ibs,
        "mae_uncensored_sessions_diagnostic": mae,
        "mean_expected_sessions": float(np.mean(expected_sessions)),
        "brier_by_horizon": brier_by_horizon,
        "calibration_by_horizon": calibration_by_horizon,
    }


def _evaluate_heuristic_variant(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    globals_payload: dict[str, float],
    use_median: bool,
) -> dict:
    y_train_struct = Surv.from_arrays(
        event=train_df["event_observed"].astype(bool).to_numpy(),
        time=train_df["sessions_until_next_pr"].astype(float).to_numpy(),
    )
    eval_enriched = _with_heuristic_stats(eval_df, stats_df, globals_payload)

    gap_col = "median_sessions_between_prs" if use_median else "avg_sessions_between_prs"
    gap = pd.to_numeric(eval_enriched[gap_col], errors="coerce").fillna(
        globals_payload["global_median_sessions_between_prs"] if use_median else globals_payload["global_avg_sessions_between_prs"]
    ).to_numpy(dtype=float)
    sessions_since = pd.to_numeric(eval_enriched["sessions_since_last_pr"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    expected = _heuristic_expected_sessions(sessions_since, gap)
    horizon_probs = {h: _heuristic_prob_within_n(sessions_since, gap, float(h)) for h in HORIZONS}

    grid = _build_eval_grid(
        train_df["sessions_until_next_pr"].astype(float).to_numpy(),
        eval_df["sessions_until_next_pr"].astype(float).to_numpy(),
    )
    surv_matrix = _heuristic_survival_matrix(sessions_since, gap, grid)

    return _evaluate_generic_split(
        expected_sessions=expected,
        horizon_probs=horizon_probs,
        surv_matrix=surv_matrix,
        y_train_struct=y_train_struct,
        event_eval=eval_df["event_observed"],
        duration_eval=eval_df["sessions_until_next_pr"],
    )


def _km_survival_at(time_grid: np.ndarray, surv_probs: np.ndarray, t: float) -> float:
    if len(time_grid) == 0:
        return 1.0
    idx = np.searchsorted(time_grid, t, side="right") - 1
    if idx < 0:
        return 1.0
    return float(surv_probs[idx])


def _evaluate_kaplan_meier(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    train_event = train_df["event_observed"].astype(bool).to_numpy()
    train_time = train_df["sessions_until_next_pr"].astype(float).to_numpy()
    eval_event = eval_df["event_observed"]
    eval_time = eval_df["sessions_until_next_pr"]

    km_times, km_surv = kaplan_meier_estimator(train_event, train_time)
    km_times = np.asarray(km_times, dtype=float)
    km_surv = np.asarray(km_surv, dtype=float)

    if len(km_times) > 0:
        delta = np.diff(np.r_[0.0, km_times])
        expected_single = float(np.sum(km_surv * delta))
    else:
        expected_single = np.nan

    n_eval = len(eval_df)
    expected = np.full(n_eval, expected_single, dtype=float)
    horizon_probs = {h: np.full(n_eval, 1.0 - _km_survival_at(km_times, km_surv, float(h)), dtype=float) for h in HORIZONS}

    grid = _build_eval_grid(train_time, eval_df["sessions_until_next_pr"].astype(float).to_numpy())
    surv_grid = np.array([_km_survival_at(km_times, km_surv, float(t)) for t in grid], dtype=float)
    surv_matrix = np.tile(surv_grid, (n_eval, 1)) if len(grid) > 0 else np.empty((n_eval, 0))

    return _evaluate_generic_split(
        expected_sessions=expected,
        horizon_probs=horizon_probs,
        surv_matrix=surv_matrix,
        y_train_struct=Surv.from_arrays(event=train_event, time=train_time),
        event_eval=eval_event,
        duration_eval=eval_time,
    )


def _build_heuristic_forecasts(
    latest_rows: pd.DataFrame,
    stats_df: pd.DataFrame,
    globals_payload: dict[str, float],
    use_median: bool,
) -> list[dict]:
    enriched = _with_heuristic_stats(latest_rows, stats_df, globals_payload)
    gap_col = "median_sessions_between_prs" if use_median else "avg_sessions_between_prs"

    forecasts = []
    for _, row in enriched.iterrows():
        gap = _safe_float(row.get(gap_col), globals_payload["global_median_sessions_between_prs"] if use_median else globals_payload["global_avg_sessions_between_prs"])
        s_since = _safe_float(row.get("sessions_since_last_pr"), 0.0)

        curve = []
        for n in CURVE_POINTS:
            p = float(_heuristic_prob_within_n(np.array([s_since]), np.array([gap]), float(n))[0])
            curve.append({"sessions_ahead": int(n), "probability_of_pr": p})

        prob5 = float(_heuristic_prob_within_n(np.array([s_since]), np.array([gap]), 5.0)[0])
        prob10 = float(_heuristic_prob_within_n(np.array([s_since]), np.array([gap]), 10.0)[0])
        prob20 = float(_heuristic_prob_within_n(np.array([s_since]), np.array([gap]), 20.0)[0])

        forecasts.append(
            {
                "exercise": str(row["exercise_title"]),
                "current_best_est_1RM": _safe_float(row.get("best_est_1RM"), np.nan),
                "expected_sessions_until_pr": float(_heuristic_expected_sessions(np.array([s_since]), np.array([gap]))[0]),
                "probability_of_pr_within_5_sessions": prob5,
                "probability_of_pr_within_10_sessions": prob10,
                "probability_of_pr_within_20_sessions": prob20,
                "probability_within_10_sessions": prob10,
                "pr_window": {
                    "50_percent_probability": first_session_meeting_threshold(curve, 0.50),
                    "80_percent_probability": first_session_meeting_threshold(curve, 0.80),
                },
                "50_percent_probability_window": first_session_meeting_threshold(curve, 0.50),
                "80_percent_probability_window": first_session_meeting_threshold(curve, 0.80),
                "probability_curve": curve,
                "benchmark_metadata": {
                    "sessions_since_last_pr": float(s_since),
                    "avg_sessions_between_prs": float(_safe_float(row.get("avg_sessions_between_prs"), globals_payload["global_avg_sessions_between_prs"])),
                    "median_sessions_between_prs": float(_safe_float(row.get("median_sessions_between_prs"), globals_payload["global_median_sessions_between_prs"])),
                    "n_historical_prs": int(_safe_float(row.get("n_historical_prs"), 0.0)),
                    "total_workout_sessions": int(_safe_float(row.get("total_sessions"), 0.0)),
                    "gap_mode": "median" if use_median else "mean",
                },
            }
        )

    forecasts = sorted(forecasts, key=lambda item: -item.get("probability_of_pr_within_10_sessions", 0.0))
    for rank, row in enumerate(forecasts, start=1):
        row["rank_by_10_session_probability"] = rank
    return forecasts


def _build_km_forecasts(latest_rows: pd.DataFrame, km_times: np.ndarray, km_surv: np.ndarray) -> list[dict]:
    if len(km_times) > 0:
        delta = np.diff(np.r_[0.0, km_times])
        expected_single = float(np.sum(km_surv * delta))
    else:
        expected_single = np.nan

    forecasts = []
    for _, row in latest_rows.iterrows():
        curve = []
        for n in CURVE_POINTS:
            p = float(1.0 - _km_survival_at(km_times, km_surv, float(n)))
            curve.append({"sessions_ahead": int(n), "probability_of_pr": p})

        prob5 = float(1.0 - _km_survival_at(km_times, km_surv, 5.0))
        prob10 = float(1.0 - _km_survival_at(km_times, km_surv, 10.0))
        prob20 = float(1.0 - _km_survival_at(km_times, km_surv, 20.0))

        forecasts.append(
            {
                "exercise": str(row["exercise_title"]),
                "current_best_est_1RM": _safe_float(row.get("best_est_1RM"), np.nan),
                "expected_sessions_until_pr": expected_single,
                "probability_of_pr_within_5_sessions": prob5,
                "probability_of_pr_within_10_sessions": prob10,
                "probability_of_pr_within_20_sessions": prob20,
                "probability_within_10_sessions": prob10,
                "pr_window": {
                    "50_percent_probability": first_session_meeting_threshold(curve, 0.50),
                    "80_percent_probability": first_session_meeting_threshold(curve, 0.80),
                },
                "50_percent_probability_window": first_session_meeting_threshold(curve, 0.50),
                "80_percent_probability_window": first_session_meeting_threshold(curve, 0.80),
                "probability_curve": curve,
            }
        )

    forecasts = sorted(forecasts, key=lambda item: -item.get("probability_of_pr_within_10_sessions", 0.0))
    for rank, row in enumerate(forecasts, start=1):
        row["rank_by_10_session_probability"] = rank
    return forecasts


def _extract_summary_row(name: str, metrics_test: dict) -> dict:
    return {
        "Model": name,
        "C-index": metrics_test.get("c_index", np.nan),
        "IBS": metrics_test.get("ibs", np.nan),
        "MAE": metrics_test.get("mae_uncensored_sessions_diagnostic", np.nan),
        "Brier@5": metrics_test.get("brier_by_horizon", {}).get("5", np.nan),
        "Brier@10": metrics_test.get("brier_by_horizon", {}).get("10", np.nan),
        "Brier@20": metrics_test.get("brier_by_horizon", {}).get("20", np.nan),
        "Mean predicted sessions until PR": metrics_test.get("mean_expected_sessions", np.nan),
    }


def run() -> None:
    df = prepare_model_c_frame()
    df = construct_survival_labels(df)
    train, val, test = time_splits(df)

    stats_df, global_gap_stats = _compute_pr_gap_stats(train)

    heuristic_mean_metrics = {
        "train": _evaluate_heuristic_variant(train, train, stats_df, global_gap_stats, use_median=False),
        "val": _evaluate_heuristic_variant(train, val, stats_df, global_gap_stats, use_median=False),
        "test": _evaluate_heuristic_variant(train, test, stats_df, global_gap_stats, use_median=False),
    }
    heuristic_median_metrics = {
        "train": _evaluate_heuristic_variant(train, train, stats_df, global_gap_stats, use_median=True),
        "val": _evaluate_heuristic_variant(train, val, stats_df, global_gap_stats, use_median=True),
        "test": _evaluate_heuristic_variant(train, test, stats_df, global_gap_stats, use_median=True),
    }

    km_train_event = train["event_observed"].astype(bool).to_numpy()
    km_train_time = train["sessions_until_next_pr"].astype(float).to_numpy()
    km_times, km_surv = kaplan_meier_estimator(km_train_event, km_train_time)
    km_times = np.asarray(km_times, dtype=float)
    km_surv = np.asarray(km_surv, dtype=float)

    km_metrics = {
        "train": _evaluate_kaplan_meier(train, train),
        "val": _evaluate_kaplan_meier(train, val),
        "test": _evaluate_kaplan_meier(train, test),
    }

    y_train_struct = Surv.from_arrays(
        event=train["event_observed"].astype(bool).to_numpy(),
        time=train["sessions_until_next_pr"].astype(float).to_numpy(),
    )

    rsf_model_path = MODEL_C_DIR / "model_C_rsf_survival.joblib"
    if rsf_model_path.exists():
        rsf_payload = joblib.load(rsf_model_path)
        rsf_model = rsf_payload["model"]
        rsf_features = rsf_payload["features"]
        rsf_feature_columns = rsf_payload["feature_columns"]
        rsf_source = "loaded_production_artifact"
    else:
        rsf_features = PRODUCTION_FEATURES
        x_train_tmp = build_model_c_matrix(train, rsf_features)
        rsf_feature_columns = list(x_train_tmp.columns)
        rsf_model = train_rsf(x_train_tmp, y_train_struct)
        rsf_source = "trained_in_memory_fallback"

    x_train = build_model_c_matrix(train, rsf_features, ref_columns=rsf_feature_columns)
    x_val = build_model_c_matrix(val, rsf_features, ref_columns=rsf_feature_columns)
    x_test = build_model_c_matrix(test, rsf_features, ref_columns=rsf_feature_columns)

    rsf_metrics = {
        "train": evaluate_split(rsf_model, x_train, train["event_observed"], train["sessions_until_next_pr"], HORIZONS, y_train_struct, "train"),
        "val": evaluate_split(rsf_model, x_val, val["event_observed"], val["sessions_until_next_pr"], HORIZONS, y_train_struct, "validation"),
        "test": evaluate_split(rsf_model, x_test, test["event_observed"], test["sessions_until_next_pr"], HORIZONS, y_train_struct, "test"),
    }

    latest_rows = df.sort_values("date").groupby("exercise_title", as_index=False).tail(1).copy()

    heuristic_mean_forecasts = _build_heuristic_forecasts(latest_rows, stats_df, global_gap_stats, use_median=False)
    heuristic_median_forecasts = _build_heuristic_forecasts(latest_rows, stats_df, global_gap_stats, use_median=True)
    km_forecasts = _build_km_forecasts(latest_rows, km_times, km_surv)

    rsf_latest_x = build_model_c_matrix(latest_rows, rsf_features, ref_columns=rsf_feature_columns)
    rsf_expected = expected_sessions_from_survival(rsf_model, rsf_latest_x)
    rsf_surv = rsf_model.predict_survival_function(rsf_latest_x, return_array=False)
    rsf_forecasts = []
    for i, (_, row) in enumerate(latest_rows.reset_index(drop=True).iterrows()):
        fn = rsf_surv[i]
        curve = [{"sessions_ahead": int(n), "probability_of_pr": float(1.0 - float(fn(n)))} for n in CURVE_POINTS]
        p10 = float(probability_within_n_sessions(rsf_model, rsf_latest_x.iloc[[i]], 10)[0])
        rsf_forecasts.append(
            {
                "exercise": str(row["exercise_title"]),
                "current_best_est_1RM": _safe_float(row.get("best_est_1RM"), np.nan),
                "expected_sessions_until_pr": float(rsf_expected[i]),
                "probability_of_pr_within_5_sessions": float(probability_within_n_sessions(rsf_model, rsf_latest_x.iloc[[i]], 5)[0]),
                "probability_of_pr_within_10_sessions": p10,
                "probability_of_pr_within_20_sessions": float(probability_within_n_sessions(rsf_model, rsf_latest_x.iloc[[i]], 20)[0]),
                "probability_within_10_sessions": p10,
                "pr_window": {
                    "50_percent_probability": first_session_meeting_threshold(curve, 0.50),
                    "80_percent_probability": first_session_meeting_threshold(curve, 0.80),
                },
                "50_percent_probability_window": first_session_meeting_threshold(curve, 0.50),
                "80_percent_probability_window": first_session_meeting_threshold(curve, 0.80),
                "probability_curve": curve,
            }
        )

    rsf_forecasts = sorted(rsf_forecasts, key=lambda item: -item.get("probability_of_pr_within_10_sessions", 0.0))
    for rank, row in enumerate(rsf_forecasts, start=1):
        row["rank_by_10_session_probability"] = rank

    heuristic_payload = {
        "model": "Heuristic probability benchmark",
        "curve_points": CURVE_POINTS,
        "global_pr_gap_stats": global_gap_stats,
        "variants": {
            "mean_gap": {
                "forecasts": heuristic_mean_forecasts,
            },
            "median_gap": {
                "forecasts": heuristic_median_forecasts,
            },
        },
    }
    km_payload = {
        "model": "Kaplan-Meier benchmark",
        "curve_points": CURVE_POINTS,
        "forecasts": km_forecasts,
    }

    (BENCHMARK_DIR / "heuristic_probability_predictions.json").write_text(json.dumps(heuristic_payload, indent=2))
    (BENCHMARK_DIR / "kaplan_meier_predictions.json").write_text(json.dumps(km_payload, indent=2))

    benchmark_metrics = {
        "heuristic_mean": heuristic_mean_metrics,
        "heuristic_median": heuristic_median_metrics,
        "kaplan_meier": km_metrics,
        "production_rsf": rsf_metrics,
    }
    (BENCHMARK_DIR / "benchmark_metrics.json").write_text(json.dumps(benchmark_metrics, indent=2))

    summary_rows = [
        _extract_summary_row("Heuristic (mean gap)", heuristic_mean_metrics["test"]),
        _extract_summary_row("Heuristic (median gap)", heuristic_median_metrics["test"]),
        _extract_summary_row("Kaplan-Meier", km_metrics["test"]),
        _extract_summary_row("Production RSF", rsf_metrics["test"]),
    ]

    rsf_test = rsf_metrics["test"]
    comparisons = {
        "vs_heuristic_mean": {
            "rsf_better_c_index": bool(rsf_test["c_index"] > heuristic_mean_metrics["test"]["c_index"]),
            "rsf_better_ibs": bool(rsf_test["ibs"] < heuristic_mean_metrics["test"]["ibs"]),
            "rsf_better_mae": bool(rsf_test["mae_uncensored_sessions_diagnostic"] < heuristic_mean_metrics["test"]["mae_uncensored_sessions_diagnostic"]),
        },
        "vs_heuristic_median": {
            "rsf_better_c_index": bool(rsf_test["c_index"] > heuristic_median_metrics["test"]["c_index"]),
            "rsf_better_ibs": bool(rsf_test["ibs"] < heuristic_median_metrics["test"]["ibs"]),
            "rsf_better_mae": bool(rsf_test["mae_uncensored_sessions_diagnostic"] < heuristic_median_metrics["test"]["mae_uncensored_sessions_diagnostic"]),
        },
        "vs_kaplan_meier": {
            "rsf_better_c_index": bool(rsf_test["c_index"] > km_metrics["test"]["c_index"]),
            "rsf_better_ibs": bool(rsf_test["ibs"] < km_metrics["test"]["ibs"]),
            "rsf_better_mae": bool(rsf_test["mae_uncensored_sessions_diagnostic"] < km_metrics["test"]["mae_uncensored_sessions_diagnostic"]),
        },
    }

    summary_payload = {
        "table": summary_rows,
        "rsf_outperformance": comparisons,
        "artifacts": {
            "heuristic_mean_predictions": "models/benchmarks/heuristic_probability_predictions.json",
            "heuristic_median_predictions": "models/benchmarks/heuristic_probability_predictions.json",
            "kaplan_meier_predictions": "models/benchmarks/kaplan_meier_predictions.json",
            "benchmark_metrics": "models/benchmarks/benchmark_metrics.json",
        },
        "rsf_source": rsf_source,
        "dashboard_schema_compatible_fields": [
            "exercise",
            "expected_sessions_until_pr",
            "probability_within_10_sessions",
            "probability_curve",
        ],
    }
    (BENCHMARK_DIR / "model_C_benchmark_summary.json").write_text(json.dumps(summary_payload, indent=2))

    print("Saved benchmark outputs:")
    print("-", BENCHMARK_DIR / "heuristic_probability_predictions.json")
    print("-", BENCHMARK_DIR / "kaplan_meier_predictions.json")
    print("-", BENCHMARK_DIR / "benchmark_metrics.json")
    print("-", BENCHMARK_DIR / "model_C_benchmark_summary.json")


if __name__ == "__main__":
    run()
