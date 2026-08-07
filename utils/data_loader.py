"""Data loading and normalization helpers for the Streamlit dashboard."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.feature_engineering import CLEANED_WORKOUTS_PATH

ROOT = Path(__file__).resolve().parents[1]
FORECAST_CANDIDATES = [
    ROOT / "outputs" / "pr_forecast_predictions.json",
    ROOT / "models" / "pr_forecast_predictions.json",
    ROOT / "models" / "model_C" / "experiments" / "pr_forecast_predictions.json",
]

HEURISTIC_BENCHMARK_CANDIDATES = [
    ROOT / "models" / "benchmarks" / "heuristic_probability_predictions.json",
]

KAPLAN_MEIER_BENCHMARK_CANDIDATES = [
    ROOT / "models" / "benchmarks" / "kaplan_meier_predictions.json",
]

WORKOUTS_PATH = CLEANED_WORKOUTS_PATH


def resolve_forecast_path() -> Path:
    for candidate in FORECAST_CANDIDATES:
        if candidate.exists():
            return candidate
    return FORECAST_CANDIDATES[0]


def _resolve_first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_heuristic_benchmark_path() -> Path:
    return _resolve_first_existing(HEURISTIC_BENCHMARK_CANDIDATES)


def resolve_kaplan_meier_benchmark_path() -> Path:
    return _resolve_first_existing(KAPLAN_MEIER_BENCHMARK_CANDIDATES)


def resolve_workouts_path() -> Path:
    return WORKOUTS_PATH


def get_file_signature(path: Path) -> tuple[int, int]:
    if not path.exists():
        return (0, 0)
    stat = path.stat()
    return (int(stat.st_mtime_ns), int(stat.st_size))


@st.cache_data(show_spinner=False)
def load_forecast_payload(path_str: str, file_signature: tuple[int, int]) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Forecast file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _extract_window_value(record: dict[str, Any], primary_key: str, nested_key: str | None = None) -> int | None:
    if primary_key in record and record.get(primary_key) is not None:
        return _safe_int(record.get(primary_key))
    window = record.get("pr_window") or {}
    if nested_key and nested_key in window:
        return _safe_int(window.get(nested_key))
    if primary_key in window:
        return _safe_int(window.get(primary_key))
    return None


def _extract_curve_value(curve: list[dict[str, Any]], sessions_ahead: int) -> float | None:
    for point in curve:
        if _safe_int(point.get("sessions_ahead")) == sessions_ahead:
            return _safe_float(point.get("probability_of_pr", point.get("probability")))
    return None


def normalize_forecast_record(record: dict[str, Any]) -> dict[str, Any]:
    raw_curve = record.get("probability_curve") or []
    curve = []
    for point in raw_curve:
        sessions_ahead = _safe_int(point.get("sessions_ahead"))
        probability = _safe_float(point.get("probability_of_pr", point.get("probability")))
        if sessions_ahead is not None and probability is not None:
            curve.append({"sessions_ahead": sessions_ahead, "probability_of_pr": probability})

    current_best = record.get("current_best_est_1RM", record.get("current_best_1RM"))
    current_best = _safe_float(current_best)
    expected_sessions = _safe_float(record.get("expected_sessions_until_pr"))
    prob5 = _safe_float(record.get("probability_of_pr_within_5_sessions"))
    prob10 = _safe_float(record.get("probability_of_pr_within_10_sessions"))
    prob20 = _safe_float(record.get("probability_of_pr_within_20_sessions"))

    if prob5 is None:
        prob5 = _extract_curve_value(curve, 5)
    if prob10 is None:
        prob10 = _extract_curve_value(curve, 10)
    if prob20 is None:
        prob20 = _extract_curve_value(curve, 20)

    normalized = {
        "exercise": str(record.get("exercise", "Unknown")),
        "current_best_est_1RM": current_best,
        "expected_sessions_until_pr": expected_sessions,
        "probability_of_pr_within_5_sessions": prob5,
        "probability_of_pr_within_10_sessions": prob10,
        "probability_of_pr_within_20_sessions": prob20,
        "50_percent_probability_window": _extract_window_value(record, "50_percent_probability_window", "50_percent_probability"),
        "80_percent_probability_window": _extract_window_value(record, "80_percent_probability_window", "80_percent_probability"),
        "rank_by_10_session_probability": _safe_int(record.get("rank_by_10_session_probability")),
        "probability_curve": curve,
    }
    return normalized


def load_forecast_dataframe(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any], Path, tuple[int, int]]:
    forecast_path = path or resolve_forecast_path()
    file_signature = get_file_signature(forecast_path)
    payload = load_forecast_payload(str(forecast_path), file_signature)

    records = [normalize_forecast_record(record) for record in payload.get("forecasts", [])]
    frame = pd.DataFrame(records)
    if not frame.empty and "probability_of_pr_within_10_sessions" in frame.columns:
        frame = frame.sort_values(
            by=["probability_of_pr_within_10_sessions", "expected_sessions_until_pr"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    if not frame.empty:
        frame["display_rank"] = np.arange(1, len(frame) + 1)
    return frame, payload, forecast_path, file_signature


def _records_from_heuristic_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    variants = payload.get("variants")
    if isinstance(variants, dict):
        mean_payload = variants.get("mean_gap") or {}
        records = mean_payload.get("forecasts") or []
        return records if isinstance(records, list) else []
    records = payload.get("forecasts") or []
    return records if isinstance(records, list) else []


def load_heuristic_benchmark_dataframe(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any], Path, tuple[int, int]]:
    benchmark_path = path or resolve_heuristic_benchmark_path()
    file_signature = get_file_signature(benchmark_path)
    payload = load_forecast_payload(str(benchmark_path), file_signature)

    records = [normalize_forecast_record(record) for record in _records_from_heuristic_payload(payload)]
    frame = pd.DataFrame(records)
    if not frame.empty and "probability_of_pr_within_10_sessions" in frame.columns:
        frame = frame.sort_values(
            by=["probability_of_pr_within_10_sessions", "expected_sessions_until_pr"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    if not frame.empty:
        frame["display_rank"] = np.arange(1, len(frame) + 1)

    return frame, payload, benchmark_path, file_signature


def load_kaplan_meier_benchmark_dataframe(path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any], Path, tuple[int, int]]:
    benchmark_path = path or resolve_kaplan_meier_benchmark_path()
    file_signature = get_file_signature(benchmark_path)
    payload = load_forecast_payload(str(benchmark_path), file_signature)

    records = [normalize_forecast_record(record) for record in payload.get("forecasts", [])]
    frame = pd.DataFrame(records)
    if not frame.empty and "probability_of_pr_within_10_sessions" in frame.columns:
        frame = frame.sort_values(
            by=["probability_of_pr_within_10_sessions", "expected_sessions_until_pr"],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)
    if not frame.empty:
        frame["display_rank"] = np.arange(1, len(frame) + 1)

    return frame, payload, benchmark_path, file_signature


@st.cache_data(show_spinner=False)
def load_workout_history_dataframe(path_str: str, file_signature: tuple[int, int]) -> pd.DataFrame:
    path = Path(path_str)
    frame = pd.read_csv(path)
    if frame.empty:
        return frame

    if "exercise_title" in frame.columns:
        frame["exercise_title"] = frame["exercise_title"].astype(str)

    if "start_time" in frame.columns:
        frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    return frame


def build_summary_metrics(frame: pd.DataFrame, soon_threshold: float = 0.80) -> dict[str, Any]:
    if frame.empty:
        return {
            "total_exercises": 0,
            "likely_to_pr_soon": 0,
            "average_sessions_until_pr": None,
            "average_prob10": None,
        }

    prob10 = pd.to_numeric(frame.get("probability_of_pr_within_10_sessions"), errors="coerce")
    expected = pd.to_numeric(frame.get("expected_sessions_until_pr"), errors="coerce")
    likely_soon = int((prob10 >= soon_threshold).fillna(False).sum())

    return {
        "total_exercises": int(len(frame)),
        "likely_to_pr_soon": likely_soon,
        "average_sessions_until_pr": float(expected.mean()) if expected.notna().any() else None,
        "average_prob10": float(prob10.mean()) if prob10.notna().any() else None,
    }


def get_selected_record(frame: pd.DataFrame, selected_exercise: str) -> dict[str, Any] | None:
    if frame.empty:
        return None
    if selected_exercise == "All Exercises":
        summary = {
            "exercise": "All Exercises",
            "current_best_est_1RM": float(pd.to_numeric(frame["current_best_est_1RM"], errors="coerce").median()) if frame["current_best_est_1RM"].notna().any() else None,
            "expected_sessions_until_pr": float(pd.to_numeric(frame["expected_sessions_until_pr"], errors="coerce").mean()) if frame["expected_sessions_until_pr"].notna().any() else None,
            "probability_of_pr_within_5_sessions": float(pd.to_numeric(frame["probability_of_pr_within_5_sessions"], errors="coerce").mean()) if frame["probability_of_pr_within_5_sessions"].notna().any() else None,
            "probability_of_pr_within_10_sessions": float(pd.to_numeric(frame["probability_of_pr_within_10_sessions"], errors="coerce").mean()) if frame["probability_of_pr_within_10_sessions"].notna().any() else None,
            "probability_of_pr_within_20_sessions": float(pd.to_numeric(frame["probability_of_pr_within_20_sessions"], errors="coerce").mean()) if frame["probability_of_pr_within_20_sessions"].notna().any() else None,
            "50_percent_probability_window": float(pd.to_numeric(frame["50_percent_probability_window"], errors="coerce").median()) if frame["50_percent_probability_window"].notna().any() else None,
            "80_percent_probability_window": float(pd.to_numeric(frame["80_percent_probability_window"], errors="coerce").median()) if frame["80_percent_probability_window"].notna().any() else None,
            "probability_curve": [],
        }
        return summary

    match = frame.loc[frame["exercise"] == selected_exercise]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def build_rankings_table(frame: pd.DataFrame, exercise_filter: str = "All Exercises", min_probability: float = 0.0) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    ranked = frame.copy()
    if exercise_filter != "All Exercises":
        ranked = ranked.loc[ranked["exercise"] == exercise_filter].copy()
    ranked = ranked.loc[pd.to_numeric(ranked["probability_of_pr_within_10_sessions"], errors="coerce") >= min_probability].copy()
    ranked = ranked.sort_values(
        by=["probability_of_pr_within_10_sessions", "expected_sessions_until_pr"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return ranked[[
        "exercise",
        "current_best_est_1RM",
        "expected_sessions_until_pr",
        "probability_of_pr_within_10_sessions",
        "50_percent_probability_window",
        "80_percent_probability_window",
    ]]
