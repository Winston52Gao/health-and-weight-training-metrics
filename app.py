"""Streamlit dashboard for Model C PR forecasts."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.data_loader import (
    build_rankings_table,
    build_summary_metrics,
    get_selected_record,
    load_forecast_dataframe,
    load_heuristic_benchmark_dataframe,
    load_kaplan_meier_benchmark_dataframe,
    load_workout_history_dataframe,
    resolve_forecast_path,
    resolve_heuristic_benchmark_path,
    resolve_kaplan_meier_benchmark_path,
    resolve_workouts_path,
)
from utils.visualization import (
    build_average_labeled_curve_dataframe,
    build_best_set_volume_dataframe,
    build_labeled_curve_dataframe,
    build_multi_model_curve_figure,
    build_training_volume_dataframe,
    build_weight_progression_dataframe,
    build_workout_time_series_figure,
)

st.set_page_config(
    page_title="Progressive Overload Forecast Dashboard",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
            color: #0f172a;
        }
        .dashboard-shell {
            padding-top: 0.5rem;
        }
        .section-card {
            background: rgba(255,255,255,0.78);
            border: 1px solid rgba(148,163,184,0.25);
            border-radius: 18px;
            padding: 1rem 1.15rem;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        }
        .section-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .subtle {
            color: #64748b;
            font-size: 0.95rem;
        }
        .stMetric label,
        .stMetric div,
        .stMetric [data-testid="stMetricLabel"],
        .stMetric [data-testid="stMetricValue"],
        .stMarkdown,
        .stSubheader,
        .stCaption {
            color: #000000 !important;
        }
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stSubheader,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] div {
            color: #ffffff !important;
        }
        button[title="Download as CSV"],
        button[aria-label="Download as CSV"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_summary_card(selected_record: dict | None) -> None:
    if not selected_record:
        st.warning("No forecast data found.")
        return

    title = selected_record.get("exercise", "Selected Exercise")
    with st.container(border=True):
        st.markdown(f"### {title}")
        cols = st.columns(4)
        cols[0].metric(
            "Current estimated 1RM",
            f"{selected_record.get('current_best_est_1RM'):.2f}" if selected_record.get("current_best_est_1RM") is not None else "N/A",
        )
        cols[1].metric(
            "Expected sessions until PR",
            f"{selected_record.get('expected_sessions_until_pr'):.2f}" if selected_record.get("expected_sessions_until_pr") is not None else "N/A",
        )
        cols[2].metric(
            "Likely PR window (50%)",
            f"{selected_record.get('50_percent_probability_window')} sessions" if selected_record.get("50_percent_probability_window") is not None else "N/A",
        )
        cols[3].metric(
            "High-confidence window (80%)",
            f"{selected_record.get('80_percent_probability_window')} sessions" if selected_record.get("80_percent_probability_window") is not None else "N/A",
        )

        prob10 = selected_record.get("probability_of_pr_within_10_sessions")
        if prob10 is not None:
            st.caption(f"Probability of PR within 10 sessions: {prob10 * 100:.1f}%")


def _get_exercise_record(frame: pd.DataFrame, selected_exercise: str) -> dict:
    if frame.empty:
        return {}
    match = frame.loc[frame["exercise"] == selected_exercise]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _is_allowed_selector_exercise(name: str) -> bool:
    lowered = str(name).strip().lower()
    blocked_terms = [
        "air bike",
        "treadmill",
        "plank",
        "leg raise",
        "hanging leg raise",
    ]
    return not any(term in lowered for term in blocked_terms)


def main() -> None:
    apply_theme()

    forecast_path = resolve_forecast_path()
    frame, payload, resolved_path, _ = load_forecast_dataframe(forecast_path)
    heuristic_frame, _, heuristic_path, _ = load_heuristic_benchmark_dataframe(resolve_heuristic_benchmark_path())
    km_frame, _, km_path, _ = load_kaplan_meier_benchmark_dataframe(resolve_kaplan_meier_benchmark_path())
    workouts_path = resolve_workouts_path()
    workouts_signature = (int(workouts_path.stat().st_mtime_ns), int(workouts_path.stat().st_size)) if workouts_path.exists() else (0, 0)
    workouts_frame = load_workout_history_dataframe(str(workouts_path), workouts_signature)

    if not workouts_frame.empty and "exercise_title" in workouts_frame.columns:
        valid_exercises = set(workouts_frame["exercise_title"].dropna().astype(str).unique().tolist())
        if valid_exercises:
            frame = frame.loc[frame["exercise"].isin(valid_exercises)].reset_index(drop=True)
            heuristic_frame = heuristic_frame.loc[heuristic_frame["exercise"].isin(valid_exercises)].reset_index(drop=True)
            km_frame = km_frame.loc[km_frame["exercise"].isin(valid_exercises)].reset_index(drop=True)

    st.title("Progressive Overload Forecast Dashboard")
    st.caption(
        f"Primary forecast source: RSF"
    )
    st.caption("Benchmarks: mean heuristic, kaplan_Meier")

    with st.sidebar:
        st.subheader("Dashboard Controls")
        if st.button("Refresh forecast data"):
            st.cache_data.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### Exercise selector")
        forecast_exercises = sorted(frame["exercise"].dropna().astype(str).unique().tolist()) if not frame.empty else []
        workout_exercises = sorted(workouts_frame["exercise_title"].dropna().astype(str).unique().tolist()) if not workouts_frame.empty and "exercise_title" in workouts_frame.columns else []
        canonical_exercises = workout_exercises or forecast_exercises
        canonical_exercises = [name for name in canonical_exercises if _is_allowed_selector_exercise(name)]
        exercise_options = canonical_exercises

        if not exercise_options:
            st.warning("No valid exercises available for selection.")
            st.stop()

        selector_key = "selected_exercise_v2"
        if st.session_state.get(selector_key) not in exercise_options:
            st.session_state[selector_key] = exercise_options[0]

        selected_exercise = st.selectbox(
            "Exercise (type to search)",
            exercise_options,
            index=exercise_options.index(st.session_state.get(selector_key, exercise_options[0])),
            key=selector_key,
            help="Search by typing in the dropdown and press Enter to select.",
        )

        st.markdown("### Ranking filter")
        threshold = st.slider("Minimum probability within 10 sessions", 0.0, 1.0, 0.0, 0.05)

    if frame.empty:
        st.warning("No predictions available. Generate `outputs/pr_forecast_predictions.json` first.")
        st.stop()

    filtered_frame = frame.loc[frame["exercise"] == selected_exercise].copy()
    summary_metrics = build_summary_metrics(frame, soon_threshold=0.80)
    metric_cols = st.columns(2)
    metric_cols[0].metric("Total exercises tracked", summary_metrics["total_exercises"])
    metric_cols[1].metric("Exercises likely to PR soon", summary_metrics["likely_to_pr_soon"])

    st.markdown("---")
    selected_record = get_selected_record(frame, selected_exercise)
    render_summary_card(selected_record)

    st.markdown("---")

    curve_frames = []
    curve_frames.append(build_labeled_curve_dataframe(_get_exercise_record(frame, selected_exercise), "Model C (RSF)"))
    curve_frames.append(build_labeled_curve_dataframe(_get_exercise_record(heuristic_frame, selected_exercise), "Heuristic Mean"))
    curve_frames.append(build_labeled_curve_dataframe(_get_exercise_record(km_frame, selected_exercise), "Kaplan-Meier"))
    chart_title = f"PR probability curve comparison for {selected_exercise}"
    chart_subtitle = "Model C and benchmarks are overlaid for direct per-exercise comparison."

    curve_df = pd.concat([df for df in curve_frames if not df.empty], ignore_index=True) if curve_frames else pd.DataFrame()
    fig = build_multi_model_curve_figure(curve_df, chart_title, chart_subtitle)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Strength and workload trends")

    history_for_charts = workouts_frame.copy()
    if "exercise_title" in history_for_charts.columns:
        history_for_charts = history_for_charts.loc[history_for_charts["exercise_title"] == selected_exercise].copy()

    weight_progression = build_weight_progression_dataframe(history_for_charts)
    volume_progression = build_training_volume_dataframe(history_for_charts)
    best_set_volume_progression = build_best_set_volume_dataframe(history_for_charts)

    st.markdown("#### Weight Progression Chart")
    weight_title = f"Weight progression over time: {selected_exercise}"
    weight_fig = build_workout_time_series_figure(
        weight_progression,
        y_col="weight_kg",
        title=weight_title,
        y_axis_title="Weight used (kg)",
        line_color="#0f766e",
    )
    st.plotly_chart(weight_fig, width="stretch")

    st.markdown("#### Training Volume Chart")
    volume_title = f"Training volume over time: {selected_exercise}"
    volume_fig = build_workout_time_series_figure(
        volume_progression,
        y_col="training_volume",
        title=volume_title,
        y_axis_title="Training volume (kg x reps)",
        line_color="#1d4ed8",
    )
    st.plotly_chart(volume_fig, width="stretch")

    st.markdown("#### Best Set Volume Chart")
    best_set_volume_title = f"Best set volume over time: {selected_exercise}"
    best_set_volume_fig = build_workout_time_series_figure(
        best_set_volume_progression,
        y_col="best_set_volume",
        title=best_set_volume_title,
        y_axis_title="Best set volume (kg x reps)",
        line_color="#7c3aed",
    )
    st.plotly_chart(best_set_volume_fig, width="stretch")

    st.caption(
        "Weight progression uses daily top set weight; training volume aggregates daily weight x reps across sets; best set volume tracks the highest single-set weight x reps per day."
    )

    st.markdown("---")
    st.subheader("Exercise ranking by 10-session PR probability")
    ranked = build_rankings_table(frame, exercise_filter=selected_exercise, min_probability=threshold)

    if ranked.empty:
        st.info("No exercises match the current ranking filters.")
    else:
        display_df = ranked.copy()
        display_df["current_best_est_1RM"] = pd.to_numeric(display_df["current_best_est_1RM"], errors="coerce")
        display_df["expected_sessions_until_pr"] = pd.to_numeric(display_df["expected_sessions_until_pr"], errors="coerce")
        display_df["probability_of_pr_within_10_sessions"] = pd.to_numeric(display_df["probability_of_pr_within_10_sessions"], errors="coerce") * 100.0
        st.dataframe(
            display_df.rename(
                columns={
                    "exercise": "Exercise",
                    "current_best_est_1RM": "Current estimated 1RM",
                    "expected_sessions_until_pr": "Expected sessions until PR",
                    "probability_of_pr_within_10_sessions": "Probability within 10 sessions (%)",
                    "50_percent_probability_window": "50% PR window",
                    "80_percent_probability_window": "80% PR window",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Current estimated 1RM": st.column_config.NumberColumn(format="%.2f"),
                "Expected sessions until PR": st.column_config.NumberColumn(format="%.2f"),
                "Probability within 10 sessions (%)": st.column_config.NumberColumn(format="%.1f"),
                "50% PR window": st.column_config.NumberColumn(format="%.0f"),
                "80% PR window": st.column_config.NumberColumn(format="%.0f"),
            },
        )

    st.markdown("---")
    st.subheader("Historical workout data")
    if workouts_frame.empty:
        st.info("No historical workout dataset found.")
    else:
        history = workouts_frame.copy()
        if "exercise_title" in history.columns:
            history = history.loc[history["exercise_title"] == selected_exercise].copy()

        if history.empty:
            st.info("No historical workout rows available for this exercise.")
        else:
            if "start_time" in history.columns:
                history = history.sort_values("start_time", ascending=False)

            columns_to_show = [
                col for col in ["start_time", "title", "exercise_title", "set_index", "set_type", "weight_kg", "reps", "rpe"]
                if col in history.columns
            ]
            st.dataframe(history[columns_to_show], width="stretch", hide_index=True)

if __name__ == "__main__":
    main()
