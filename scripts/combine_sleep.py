#!/usr/bin/env python3
"""
Combine Fitbit sleep JSON chunks into a clean dataset.

Outputs:
- combined_sleep.csv : one row per sleep session with duration and stage totals
- per_night_summary.csv : aggregated totals per "night" (night labeled by the end/morning date)

Assumptions:
- Sleep sessions are assigned to a "night" by the date of their end time (so a sleep starting 23:00 on 2026-05-27 and ending 07:00 on 2026-05-28 is attributed to night 2026-05-28).
- The script handles common Fitbit JSON structures for levels: either a `levels.summary` with minutes per stage or `levels.data` entries with seconds per sample.
"""
from pathlib import Path
import json
import pandas as pd
from datetime import datetime, timezone

try:
    from dateutil import parser as dateutil_parser
except Exception:
    dateutil_parser = None


DATA_DIR = Path("fitbit_sleep_chunks")


def parse_iso(ts_str):
    if ts_str is None:
        return None
    if dateutil_parser:
        return dateutil_parser.isoparse(ts_str)
    # fallback to pandas
    return pd.to_datetime(ts_str)


def extract_stage_totals(record):
    # Return dict with deep, light, rem, wake in hours (floats). Missing stages -> 0.
    out = {"deep_hours": 0.0, "light_hours": 0.0, "rem_hours": 0.0, "wake_hours": 0.0}

    levels = record.get("levels") or {}
    # Prefer minutes already computed in levels.summary (common in Fitbit exports)
    summary = levels.get("summary") if isinstance(levels, dict) else None
    if isinstance(summary, dict):
        for k in ["deep", "light", "rem", "wake"]:
            v = summary.get(k)
            # summary entries are often objects like {"count":..., "minutes": ...}
            if isinstance(v, dict) and "minutes" in v:
                try:
                    out[f"{k}_hours"] = float(v["minutes"]) / 60.0
                except Exception:
                    pass
            else:
                # sometimes the summary maps directly to minutes
                try:
                    if v is not None:
                        out[f"{k}_hours"] = float(v) / 60.0
                except Exception:
                    pass

    # Fallback: top-level summary -> stages (minutes)
    top_summary = record.get("summary") or {}
    if isinstance(top_summary, dict):
        stages = top_summary.get("stages") or top_summary.get("stagesSummary")
        if isinstance(stages, dict):
            for k in ["deep", "light", "rem", "wake"]:
                v = stages.get(k)
                try:
                    if v is not None:
                        out[f"{k}_hours"] = max(out[f"{k}_hours"], float(v) / 60.0)
                except Exception:
                    pass

    return out


def load_sleep_records(path: Path):
    records = []
    for p in sorted(path.glob("*.json")):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        # Fitbit exports may wrap records under different keys: 'sleep' or top-level list
        if isinstance(data, dict) and "sleep" in data and isinstance(data["sleep"], list):
            recs = data["sleep"]
        elif isinstance(data, list):
            recs = data
        elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
            # try to pick the largest list
            lists = [v for v in data.values() if isinstance(v, list)]
            recs = max(lists, key=len) if lists else []
        else:
            recs = []

        for r in recs:
            records.append(r)
    return records


def session_to_row(r):
    # find start/end keys
    start_keys = ["startTime", "start_time", "start"]
    end_keys = ["endTime", "end_time", "end"]
    start_ts = None
    end_ts = None
    for k in start_keys:
        if k in r:
            start_ts = r.get(k)
            break
    for k in end_keys:
        if k in r:
            end_ts = r.get(k)
            break

    start_dt = parse_iso(start_ts) if start_ts else None
    end_dt = parse_iso(end_ts) if end_ts else None

    # Some exports give duration in ms
    duration_hours = None
    if end_dt and start_dt:
        duration_hours = (end_dt - start_dt).total_seconds() / 3600.0
    else:
        # try duration key
        dur = r.get("duration") or r.get("length")
        if dur is not None:
            try:
                d = float(dur)
                # guess: if > 10000 then ms
                if d > 10000:
                    duration_hours = d / 1000.0 / 3600.0
                else:
                    duration_hours = d / 3600.0
            except Exception:
                duration_hours = None

    # night_date: assign by end date (morning date)
    night_date = None
    if end_dt:
        night_date = end_dt.date().isoformat()
    elif start_dt and duration_hours is not None:
        # fallback: approximate end
        approx_end = start_dt + pd.Timedelta(hours=duration_hours)
        night_date = approx_end.date().isoformat()

    stages = extract_stage_totals(r)

    # common Fitbit fields
    minutes_asleep = r.get("minutesAsleep") or r.get("minutes_asleep") or r.get("minutesAsleep")
    minutes_awake = r.get("minutesAwake") or r.get("minutes_awake") or r.get("minutesAwake")
    time_in_bed = r.get("timeInBed") or r.get("time_in_bed") or r.get("timeInBed")
    minutes_to_fall = r.get("minutesToFallAsleep") or r.get("minutes_to_fall_asleep")

    # prefer explicit minutes/duration if available for duration_hours
    if duration_hours is None:
        if minutes_asleep is not None:
            try:
                duration_hours = float(minutes_asleep) / 60.0
            except Exception:
                duration_hours = duration_hours

    row = {
        "dateOfSleep": r.get("dateOfSleep") or r.get("date_of_sleep"),
        "start_time": start_dt.isoformat() if start_dt is not None else None,
        "end_time": end_dt.isoformat() if end_dt is not None else None,
        "duration_hours": duration_hours,
        "minutes_asleep": minutes_asleep,
        "minutes_awake": minutes_awake,
        "time_in_bed": time_in_bed,
        "minutes_to_fall_asleep": minutes_to_fall,
        "night_date": night_date,
        "isMainSleep": r.get("isMainSleep"),
        "efficiency": r.get("efficiency") or r.get("efficiency"),
        "logType": r.get("logType") or r.get("log_type"),
        "logId": r.get("logId") or r.get("id"),
        "type": r.get("type"),
    }
    row.update(stages)
    return row


def combine_and_save(out_dir: Path = Path(".")):
    records = load_sleep_records(DATA_DIR)
    rows = [session_to_row(r) for r in records]
    df = pd.DataFrame(rows)

    # normalize types
    # ensure numeric columns are floats
    for c in ["duration_hours", "deep_hours", "light_hours", "rem_hours", "wake_hours"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0

    df = df.sort_values(by=["night_date", "start_time"])  # nice ordering

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_csv = out_dir / "combined_sleep.csv"
    df.to_csv(combined_csv, index=False)

    # write a single JSON file with cleaned sleep records
    combined_json = out_dir / "combined_sleep.json"
    # convert to plain python types (no NaN) for JSON
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    with combined_json.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # per-night aggregation
    agg = df.groupby("night_date")[ ["duration_hours", "deep_hours", "light_hours", "rem_hours", "wake_hours"] ].sum().reset_index()
    agg = agg.rename(columns={"duration_hours": "total_slept_hours"})
    agg_csv = out_dir / "per_night_summary.csv"
    agg.to_csv(agg_csv, index=False)

    return combined_csv, combined_json, agg_csv


def main():
    out_dir = Path("./output_sleep")
    combined_csv, combined_json, agg = combine_and_save(out_dir)
    print(f"Wrote combined sessions CSV to: {combined_csv}")
    print(f"Wrote combined sessions JSON to: {combined_json}")
    print(f"Wrote per-night summary to: {agg}")


if __name__ == "__main__":
    main()
