"""Sanity checks for processed workout + Fitbit features.

Run to produce `data/sanity_report.md` summarizing missingness, targets,
per-exercise counts, rolling-feature sanity, and feature distributions.

Usage:
    python scripts/sanity_checks.py
"""
from __future__ import annotations

from pathlib import Path
import math
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_files():
    p1 = DATA / "processed_merged.csv"
    p2 = DATA / "features_model1.csv"
    p3 = DATA / "features_model2.csv"
    df1 = pd.read_csv(p1, parse_dates=["date"]) if p1.exists() else None
    f1 = pd.read_csv(p2, parse_dates=["date"]) if p2.exists() else None
    f2 = pd.read_csv(p3, parse_dates=["date"]) if p3.exists() else None
    return df1, f1, f2


def pct(x, n):
    return f"{x}/{n} ({100*x/n:.1f}%)"


def missingness_report(df: pd.DataFrame) -> str:
    n = len(df)
    miss = df.isnull().sum().sort_values()
    lines = ["## Missingness\n"]
    lines.append(f"Total rows: {n}\n")
    for col, c in miss.items():
        lines.append(f"- **{col}**: {pct(int(c), n)}\n")
    lines.append("\n")
    return "".join(lines)


def target_report(df: pd.DataFrame) -> str:
    lines = ["## Target checks\n"]
    for col in ["delta_relative_strength_next_session", "PR_next_session"]:
        if col in df.columns:
            ser = df[col]
            non_na = ser.dropna()
            lines.append(f"### {col}\n")
            lines.append(f"- non-missing: {len(non_na)} / {len(df)}\n")
            if pd.api.types.is_numeric_dtype(ser):
                lines.append(f"- stats: {non_na.describe().to_dict()}\n")
            if pd.api.types.is_integer_dtype(ser) or set(non_na.dropna().unique()) <= {0,1}:
                vc = non_na.value_counts(dropna=True).to_dict()
                lines.append(f"- value counts: {vc}\n")
    lines.append("\n")
    return "".join(lines)


def per_exercise_report(df: pd.DataFrame, min_sessions=10) -> str:
    col = "exercise_title" if "exercise_title" in df.columns else "exercise"
    lines = ["## Per-exercise session counts\n"]
    counts = df[col].value_counts()
    lines.append(f"Total distinct exercises: {counts.size}\n")
    small = counts[counts < min_sessions]
    lines.append(f"Exercises with fewer than {min_sessions} sessions: {len(small)}\n")
    if len(small) > 0:
        sample = small.index.tolist()[:20]
        lines.append(f"- examples: {sample}\n")
    lines.append("\n")
    return "".join(lines)


def rolling_feature_sanity(df: pd.DataFrame) -> str:
    lines = ["## Rolling-feature sanity checks\n"]
    checks = []
    # weekly_volume should be NaN for earliest records where no prior days
    if "weekly_volume" in df.columns:
        grp = df.groupby("exercise_title")
        count_bad = 0
        total = 0
        for _, g in grp:
            g = g.sort_values("date")
            total += 1
            first = g.iloc[0]
            # weekly_volume computed with closed='left' should be NaN or 0 for first session
            if not (math.isnan(first.get("weekly_volume", float('nan'))) or first.get("weekly_volume", 0) == 0):
                count_bad += 1
        lines.append(f"- exercises where first-session weekly_volume is not NA/0: {count_bad} / {total}\n")

    # weekly_volume_z should have roughly zero mean per exercise (where enough data)
    if "weekly_volume_z" in df.columns:
        grp = df.groupby("exercise_title")
        deviations = []
        for _, g in grp:
            s = g["weekly_volume_z"].dropna()
            if len(s) >= 5:
                deviations.append(s.mean())
        if deviations:
            lines.append(f"- mean of weekly_volume_z across exercises (median): {np.median(deviations):.3f}\n")
    lines.append("\n")
    return "".join(lines)


def date_sort_checks(df: pd.DataFrame) -> str:
    lines = ["## Date ordering checks\n"]
    bad = 0
    grp = df.groupby("exercise_title")
    for _, g in grp:
        if not g.sort_values("date").index.equals(g.index):
            bad += 1
    lines.append(f"- exercises with non-sorted date index: {bad} / {len(grp)}\n\n")
    return "".join(lines)


def generate_report():
    df, f1, f2 = load_files()
    if df is None:
        raise FileNotFoundError("data/processed_merged.csv not found — run prepare_features.py first")
    # Full dataset report
    lines = ["# Sanity Report\n\n"]
    lines.append(missingness_report(df))
    lines.append(target_report(df))
    lines.append(per_exercise_report(df))
    lines.append(rolling_feature_sanity(df))
    lines.append(date_sort_checks(df))

    # quick correlations for numeric features
    num = df.select_dtypes(include=["number"]).drop(columns=[c for c in ["PR_next_session"] if c in df.columns], errors="ignore")
    if not num.empty:
        corr = num.corr().abs()
        # keep top correlated pairs
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                pairs.append((cols[i], cols[j], corr.iloc[i,j]))
        pairs_sorted = sorted(pairs, key=lambda x: -x[2])[:20]
        lines.append("## Top absolute correlations (numeric features)\n")
        for a,b,v in pairs_sorted:
            lines.append(f"- {a} vs {b}: {v:.3f}\n")

    report = "".join(lines)
    out = DATA / "sanity_report.md"
    out.write_text(report, encoding="utf8")
    print(f"Wrote {out}")

    # Second report: only data on/after 2024-12-01
    cutoff = pd.to_datetime("2024-12-01").normalize()
    df_post = df.loc[pd.to_datetime(df["date"]) >= cutoff].copy()
    lines2 = [f"# Sanity Report — subset from {cutoff.date()} onward\n\n"]
    lines2.append(missingness_report(df_post))
    lines2.append(target_report(df_post))
    lines2.append(per_exercise_report(df_post))
    lines2.append(rolling_feature_sanity(df_post))
    lines2.append(date_sort_checks(df_post))

    num2 = df_post.select_dtypes(include=["number"]).drop(columns=[c for c in ["PR_next_session"] if c in df_post.columns], errors="ignore")
    if not num2.empty:
        corr2 = num2.corr().abs()
        pairs2 = []
        cols2 = corr2.columns
        for i in range(len(cols2)):
            for j in range(i+1, len(cols2)):
                pairs2.append((cols2[i], cols2[j], corr2.iloc[i,j]))
        pairs2_sorted = sorted(pairs2, key=lambda x: -x[2])[:20]
        lines2.append("## Top absolute correlations (numeric features) — subset\n")
        for a,b,v in pairs2_sorted:
            lines2.append(f"- {a} vs {b}: {v:.3f}\n")

    # simple comparison section
    lines2.append("\n## Comparison to full dataset\n\n")
    full_rows = len(df)
    post_rows = len(df_post)
    lines2.append(f"- total rows (full / post): {full_rows} / {post_rows}\n")
    # exercises
    ex_full = df["exercise_title"].nunique() if "exercise_title" in df.columns else 0
    ex_post = df_post["exercise_title"].nunique() if "exercise_title" in df_post.columns else 0
    lines2.append(f"- distinct exercises (full / post): {ex_full} / {ex_post}\n")
    # PR next session positives
    if "PR_next_session" in df.columns:
        pr_full = int(df["PR_next_session"].fillna(0).astype(int).sum())
        pr_post = int(df_post["PR_next_session"].fillna(0).astype(int).sum())
        lines2.append(f"- PR_next_session positives (full / post): {pr_full} / {pr_post}\n")
    # resting_hr availability
    if "resting_hr" in df.columns:
        avail_full = int(df["resting_hr"].notna().sum())
        avail_post = int(df_post["resting_hr"].notna().sum())
        lines2.append(f"- resting_hr non-missing (full / post): {avail_full} / {avail_post}\n")

    report2 = "".join(lines2)
    out2 = DATA / "sanity_report_post_2024-12-01.md"
    out2.write_text(report2, encoding="utf8")
    print(f"Wrote {out2}")


if __name__ == "__main__":
    generate_report()

