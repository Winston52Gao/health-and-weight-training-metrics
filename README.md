# Weight Training PR Prediction

This project predicts the likelihood and timing of new personal records (PRs) for weight training exercises.

## Production Flow

1. New workout data is prepared in `data/workouts.csv`
2. Feature engineering runs in `src/feature_engineering.py`
3. Model C is trained and evaluated from the `src/` production scripts
4. Forecasts are written to `outputs/pr_forecast_predictions.json`
5. The Streamlit dashboard reads that JSON and visualizes the results

## Dashboard

Launch the dashboard with:

```bash
streamlit run app.py
```

The dashboard uses the latest forecast JSON automatically and does not retrain the model.

## Model Layout

- `models/model_A/`: archived standalone workout progression classifier artifacts
- `models/model_C/`: production RSF PR timing model artifacts
- `models/model_C/experiments/`: historical ablations and benchmark outputs

## Archive

Legacy scripts and older model artifacts have been moved into `archive/`.

## Notes

- Production Model C is workout-only
- Model A remains available as a separate experimental model
- Fitbit-based features remain in experimentation only
