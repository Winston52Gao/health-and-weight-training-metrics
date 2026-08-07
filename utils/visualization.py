"""Plotly visualizations for the PR forecast dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def build_probability_curve_figure(curve: pd.DataFrame, title: str, subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()

    if curve.empty:
        fig.add_annotation(
            text="No probability curve data available.",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=16, color="#94a3b8"),
        )
        fig.update_layout(template="plotly_white", height=420)
        return fig

    fig.add_trace(
        go.Scatter(
            x=curve["sessions_ahead"],
            y=curve["probability_pct"],
            mode="lines+markers",
            name="PR probability",
            line=dict(color="#14b8a6", width=4),
            marker=dict(size=10, color="#0f766e"),
            hovertemplate="Sessions ahead: %{x}<br>Probability: %{y:.1f}%<extra></extra>",
        )
    )

    fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="50% threshold", annotation_position="top left")
    fig.add_hline(y=80, line_dash="dash", line_color="#ef4444", annotation_text="80% threshold", annotation_position="top left")

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=500,
        margin=dict(l=20, r=20, t=70, b=20),
        xaxis=dict(title="Sessions ahead", tickmode="array", tickvals=[1, 3, 5, 8, 10, 15, 20, 25, 30], gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(title="Probability of PR (%)", range=[0, 100], gridcolor="rgba(148,163,184,0.15)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
    )
    if subtitle:
        fig.add_annotation(
            text=subtitle,
            x=0,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="#64748b"),
            align="left",
        )
    return fig


def build_curve_dataframe(record: dict) -> pd.DataFrame:
    curve = record.get("probability_curve") or []
    frame = pd.DataFrame(curve)
    if frame.empty:
        return pd.DataFrame(columns=["sessions_ahead", "probability_pct"])

    probability_col = "probability_of_pr" if "probability_of_pr" in frame.columns else "probability"
    frame = frame.rename(columns={probability_col: "probability"})
    frame["probability_pct"] = pd.to_numeric(frame["probability"], errors="coerce") * 100.0
    frame["sessions_ahead"] = pd.to_numeric(frame["sessions_ahead"], errors="coerce")
    frame = frame.dropna(subset=["sessions_ahead", "probability_pct"]).sort_values("sessions_ahead")
    return frame[["sessions_ahead", "probability_pct"]]


def build_average_curve_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "probability_curve" not in frame.columns:
        return pd.DataFrame(columns=["sessions_ahead", "probability_pct"])

    curves = []
    for _, row in frame.iterrows():
        curve_df = build_curve_dataframe(row.to_dict())
        if not curve_df.empty:
            curves.append(curve_df)

    if not curves:
        return pd.DataFrame(columns=["sessions_ahead", "probability_pct"])

    combined = pd.concat(curves, ignore_index=True)
    averaged = (
        combined.groupby("sessions_ahead", as_index=False)["probability_pct"].mean().sort_values("sessions_ahead")
    )
    return averaged


def build_labeled_curve_dataframe(record: dict, model_label: str) -> pd.DataFrame:
    curve_df = build_curve_dataframe(record)
    if curve_df.empty:
        return pd.DataFrame(columns=["sessions_ahead", "probability_pct", "model"])
    curve_df = curve_df.copy()
    curve_df["model"] = model_label
    return curve_df[["sessions_ahead", "probability_pct", "model"]]


def build_average_labeled_curve_dataframe(frame: pd.DataFrame, model_label: str) -> pd.DataFrame:
    curve_df = build_average_curve_dataframe(frame)
    if curve_df.empty:
        return pd.DataFrame(columns=["sessions_ahead", "probability_pct", "model"])
    curve_df = curve_df.copy()
    curve_df["model"] = model_label
    return curve_df[["sessions_ahead", "probability_pct", "model"]]


def build_multi_model_curve_figure(curves: pd.DataFrame, title: str, subtitle: str | None = None) -> go.Figure:
    fig = go.Figure()

    if curves.empty:
        fig.add_annotation(
            text="No probability curve data available.",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=16, color="#94a3b8"),
        )
        fig.update_layout(template="plotly_white", height=420)
        return fig

    palette = {
        "Model C (RSF)": ("#0f766e", "#14b8a6"),
        "Heuristic Mean": ("#1d4ed8", "#60a5fa"),
        "Kaplan-Meier": ("#b45309", "#f59e0b"),
    }

    for label in curves["model"].dropna().unique().tolist():
        part = curves.loc[curves["model"] == label].sort_values("sessions_ahead")
        line_color, marker_color = palette.get(label, ("#475569", "#94a3b8"))
        fig.add_trace(
            go.Scatter(
                x=part["sessions_ahead"],
                y=part["probability_pct"],
                mode="lines+markers",
                name=label,
                line=dict(color=line_color, width=3),
                marker=dict(size=8, color=marker_color),
                hovertemplate="Model: " + label + "<br>Sessions ahead: %{x}<br>Probability: %{y:.1f}%<extra></extra>",
            )
        )

    fig.add_hline(y=50, line_dash="dash", line_color="#f59e0b", annotation_text="50% threshold", annotation_position="top left")
    fig.add_hline(y=80, line_dash="dash", line_color="#ef4444", annotation_text="80% threshold", annotation_position="top left")

    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, pad=dict(b=20)),
        title_font=dict(color="#000000", size=18),
        height=520,
        margin=dict(l=20, r=20, t=100, b=20),
        xaxis=dict(
            title="Sessions ahead",
            title_font=dict(color="#000000"),
            tickfont=dict(color="#000000"),
            tickmode="array",
            tickvals=[1, 3, 5, 8, 10, 15, 20, 25, 30],
            gridcolor="rgba(148,163,184,0.15)",
        ),
        yaxis=dict(
            title="Probability of PR (%)",
            title_font=dict(color="#000000"),
            tickfont=dict(color="#000000"),
            range=[0, 100],
            gridcolor="rgba(148,163,184,0.15)",
        ),
        legend=dict(orientation="h", yanchor="top", y=0.98, xanchor="left", x=0.01),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13, color="#000000"),
    )
    if subtitle:
        fig.add_annotation(
            text=subtitle,
            x=0,
            y=1.12,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=12, color="#000000"),
            align="left",
        )
    return fig


def build_weight_progression_dataframe(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["workout_date", "weight_kg"])

    frame = history.copy()
    if "start_time" not in frame.columns or "weight_kg" not in frame.columns:
        return pd.DataFrame(columns=["workout_date", "weight_kg"])

    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["weight_kg"] = pd.to_numeric(frame["weight_kg"], errors="coerce")
    frame = frame.dropna(subset=["start_time", "weight_kg"])
    if frame.empty:
        return pd.DataFrame(columns=["workout_date", "weight_kg"])

    frame["workout_date"] = frame["start_time"].dt.date
    daily = (
        frame.groupby("workout_date", as_index=False)["weight_kg"]
        .max()
        .sort_values("workout_date")
    )
    return daily


def build_training_volume_dataframe(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["workout_date", "training_volume"])

    frame = history.copy()
    required = {"start_time", "weight_kg", "reps"}
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame(columns=["workout_date", "training_volume"])

    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["weight_kg"] = pd.to_numeric(frame["weight_kg"], errors="coerce")
    frame["reps"] = pd.to_numeric(frame["reps"], errors="coerce")
    frame = frame.dropna(subset=["start_time", "weight_kg", "reps"])
    if frame.empty:
        return pd.DataFrame(columns=["workout_date", "training_volume"])

    frame["training_volume"] = frame["weight_kg"] * frame["reps"]
    frame["workout_date"] = frame["start_time"].dt.date
    daily = (
        frame.groupby("workout_date", as_index=False)["training_volume"]
        .sum()
        .sort_values("workout_date")
    )
    return daily


def build_best_set_volume_dataframe(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["workout_date", "best_set_volume"])

    frame = history.copy()
    required = {"start_time", "weight_kg", "reps"}
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame(columns=["workout_date", "best_set_volume"])

    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["weight_kg"] = pd.to_numeric(frame["weight_kg"], errors="coerce")
    frame["reps"] = pd.to_numeric(frame["reps"], errors="coerce")
    frame = frame.dropna(subset=["start_time", "weight_kg", "reps"])
    if frame.empty:
        return pd.DataFrame(columns=["workout_date", "best_set_volume"])

    frame["set_volume"] = frame["weight_kg"] * frame["reps"]
    frame["workout_date"] = frame["start_time"].dt.date
    daily = (
        frame.groupby("workout_date", as_index=False)["set_volume"]
        .max()
        .rename(columns={"set_volume": "best_set_volume"})
        .sort_values("workout_date")
    )
    return daily


def build_workout_time_series_figure(
    frame: pd.DataFrame,
    y_col: str,
    title: str,
    y_axis_title: str,
    line_color: str,
) -> go.Figure:
    fig = go.Figure()

    if frame.empty or y_col not in frame.columns or "workout_date" not in frame.columns:
        fig.add_annotation(
            text="No workout history available for this selection.",
            x=0.5,
            y=0.5,
            showarrow=False,
            xref="paper",
            yref="paper",
            font=dict(size=14, color="#94a3b8"),
        )
        fig.update_layout(template="plotly_white", height=340)
        return fig

    fig.add_trace(
        go.Scatter(
            x=frame["workout_date"],
            y=frame[y_col],
            mode="lines+markers",
            line=dict(color=line_color, width=3),
            marker=dict(size=8),
            hovertemplate="Date: %{x}<br>Value: %{y:.2f}<extra></extra>",
            name=y_axis_title,
        )
    )

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=360,
        margin=dict(l=30, r=20, t=70, b=35),
        xaxis=dict(
            title="Workout date",
            showline=True,
            linewidth=1.5,
            linecolor="rgba(100,116,139,0.7)",
            mirror=True,
            ticks="outside",
            tickwidth=1.5,
            ticklen=6,
            tickcolor="rgba(100,116,139,0.7)",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.2)",
            gridwidth=1,
            tickfont=dict(size=13, color="#000000"),
            title_font=dict(size=15, color="#000000"),
            automargin=True,
        ),
        yaxis=dict(
            title=y_axis_title,
            showline=True,
            linewidth=1.5,
            linecolor="rgba(100,116,139,0.7)",
            mirror=True,
            ticks="outside",
            tickwidth=1.5,
            ticklen=6,
            tickcolor="rgba(100,116,139,0.7)",
            showgrid=True,
            gridcolor="rgba(148,163,184,0.2)",
            gridwidth=1,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(100,116,139,0.5)",
            tickfont=dict(size=13, color="#000000"),
            title_font=dict(size=15, color="#000000"),
            automargin=True,
        ),
        title_font=dict(size=18, color="#000000"),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
