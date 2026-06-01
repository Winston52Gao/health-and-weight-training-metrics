# Sleep data combining script

Usage:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the script from the workspace root (where `fitbit_sleep_chunks/` lives):

```bash
python scripts/combine_sleep.py
```

Outputs will be written to `output_sleep/`:
- `combined_sleep.csv` — one row per sleep session with `duration_hours` and per-stage hours (`deep_hours`, `light_hours`, `rem_hours`, `wake_hours`).
- `per_night_summary.csv` — aggregated totals per night (night labeled by the sleep end/morning date) with `total_slept_hours` and stage totals.

Notes and assumptions:
- A sleep session is attributed to a "night" by its end date (so sleeps beginning 9pm–11:59pm the previous evening and ending the next morning count for that morning's date; sleeps starting after midnight up to ~1am are grouped with that same morning).
- The script attempts to be flexible with Fitbit JSON variations (it inspects `levels.summary`, `levels.data`, and top-level `summary` keys). If your export uses a different structure, please share a sample and I can adapt the parser.
