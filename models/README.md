# Models

This folder is organized by model family and then by experiment type.

## Layout

- `model_A/`: standalone workout progression classifier artifacts
- `model_C/`: production Random Survival Forest PR timing model artifacts
- `model_C/experiments/`: ablation studies, benchmarking files, and historical stacked-model results

## Model A

Model A is an independent XGBoost classifier that predicts the probability of a PR in the next workout session.

What it was trying to test:
- Whether workout progression features alone can predict an immediate PR
- How much signal exists in progression-related features such as relative strength, PR gap, volume, and training age
- Whether a simple classification framing is useful as a secondary model for analysis, but not for production stacking

Primary outputs:
- `model_A/model_A_workout.joblib`
- `model_A/model_A_metrics.json`
- `model_A/model_A_predictions.csv`
- `model_A/feature_importance_A.json`

## Model C

Model C is the production Random Survival Forest model that predicts time until next PR and produces survival-based PR probabilities over future sessions.

What it was trying to test:
- Whether workout-only progression signals are enough for reliable PR timing prediction
- Whether exercise identity improves ranking or calibration
- Whether Fitbit recovery features add useful signal or only complexity
- Whether Model A stacking adds any measurable benefit over a direct survival model

Primary production outputs:
- `model_C/model_C_rsf_survival.joblib`
- `model_C/model_C_metrics.json`
- `model_C/feature_importance_C.json`
- `model_C/model_C_example_output.json`
- `model_C/pr_timing_estimates.json`

Dashboard-ready output:
- `outputs/pr_forecast_predictions.json`

## Model C experiments and ablations

These files exist to compare feature groups and historical model variants.

### Baseline and production candidate comparisons

- `Baseline median`: naive benchmark using the median training time-to-PR
- `Workout only`: production winner; tests progression features without Fitbit or stacking
- `Workout + exercise`: checks whether exercise identity improves performance
- `Workout + Fitbit`: checks whether sleep, heart rate, and steps improve prediction
- `Full Model C`: all current features together

### Historical stacked-model comparisons

- `Model C with Model A predictions`: tests whether Model A adds value as a stacking input
- `Model C without Model A predictions`: isolates the stacked contribution of Model A
- `No model_a_score`: checks the direct effect of removing the Model A-derived feature

### Feature-group ablations

- `No progression features`: checks how much signal comes from progression-only inputs
- `No volume features`: checks whether volume features matter independently
- `No PR history features`: checks whether past PR history helps timing prediction
- `No training age features`: checks whether experience level features help
- `Raw features only`: measures performance with a minimal feature set

### Fitbit-period sensitivity

- `Fitbit period workout only`: compares workout-only performance on the higher-Fitbit-coverage time window
- `Fitbit period workout + Fitbit`: checks whether Fitbit helps when those features are available

### Key conclusion

Workout-only Model C was selected for production because it provided the best balance of C-index, IBS, MAE, simplicity, and deployment reliability.

## Notes

- Production code lives in `src/`
- Research and ablation code stays in notebooks and experiment artifacts
- Streamlit should read `outputs/pr_forecast_predictions.json` rather than retraining models
