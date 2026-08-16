# Workout PR Prediction

## Overview

This project explores whether my personal workout history can be used to predict when I am likely to achieve a new personal record (PR) on a given exercise.

The original idea was to build an LLM-powered workout chatbot using information from two training books. However, after tracking my own workouts consistently since late 2023, I became more interested in using my personal training data as a data science project. The LLM idea will be temporarily postponed as a result.

The project eventually evolved into a survival analysis problem: rather than simply predicting whether I will PR, I wanted to estimate the probability of achieving a PR over future training sessions.

## Motivation

I have been a casual gym-goer since high school, but I began training more seriously and consistently tracking my workouts around late 2023. I started using a Fitbit in late 2024, giving me additional information about sleep, heart rate, and daily activity.

This gave me the opportunity to investigate a question based on my own training: given my previous training history, how likely am I to achieve a new PR over my next several workouts?

An important aspect of the problem is that PRs do not occur at a constant interval. Some exercises progress quickly, while others can go many sessions without a new PR. In addition, PRs for individual exercises generally become less frequent as muscle gains slow. There are also periods where no PR occurs simply because I have not yet achieved one.

This made survival analysis a natural framework for the problem.

## Data

### Workout Data

My workout data comes from Hevy, which I use to track my training sessions.

The dataset contains roughly three years of increasingly consistent workout history, with detailed information about exercises, sets, repetitions, weights, and dates.

### Fitbit Data

I also used the Fitbit API to retrieve:

- Sleep duration
- Heart rate
- Steps

However, after evaluating the predictive value of these variables, I ultimately did not include Fitbit variables in the final model.

This was an important finding rather than simply a limitation of the project: in my dataset, the additional recovery variables did not provide enough incremental predictive value to justify their inclusion.

## Modeling Approach

### Initial Ensemble Approach

My initial approach was to build an ensemble consisting of three models.

#### Model A - Workout History

Model A used my entire workout history to capture training and progression patterns.

I implemented this model using XGBoost.

The goal was to determine whether historical workout information could provide a useful prediction of PR likelihood next session.

#### Model B - Recovery

Model B focused primarily on Fitbit-derived recovery and activity variables such as sleep, heart rate, and steps.

This model was also implemented using XGBoost.

The goal was to determine whether recovery-related information could improve PR predictions beyond what could be learned from workout history alone.

#### Model C - Final Model Direction

Initially, Model C was designed as a stacked model that incorporated predictions from Models A and B along with the underlying workout data.

However, evaluation showed that:

- Model A performed better than a simple average-gap benchmark.
- Model B provided only a marginal improvement in performance.
- Incorporating predictions from Models A and B into Model C resulted in only a marginal improvement in the C-index.
- The resulting time-to-PR estimates also did not show a meaningful improvement over the simpler approach.

Based on these results, I decided that the ensemble architecture was adding complexity without providing enough predictive value.

I therefore scrapped Models A and B as helper models and focused on a single survival model using the underlying workout data directly. This simplified the overall pipeline and avoided maintaining and retraining multiple intermediate models.

## Final Model: Random Survival Forest

The final model uses a Random Survival Forest (RSF).

Rather than predicting a specific PR date directly, the model estimates the probability of experiencing a PR over future training sessions.

This approach is particularly useful for my dataset because not every exercise has a PR within the observed period.

For example, an exercise may go many sessions without a PR because:

- The exercise is inherently difficult to progress.
- Progression has slowed down.
- I have recently stopped progressing on the exercise.
- I simply have not observed the next PR yet.

A traditional regression model would have difficulty representing this structure because the absence of a PR is not necessarily a known failure; the next PR may occur outside the observed data.

Survival analysis allows these observations to be treated as censored observations.

### Why Random Survival Forest?

I initially attempted to use the RSF to generate a specific estimate for the number of sessions until my next PR.

However, the resulting predictions were often unrealistically conservative.

For exercises where I expected a PR relatively soon based on my previous training experience, the model could produce very large predicted session counts.

This led me to showcase the survival curve rather than a conservative estimated date.

Instead of presenting a potentially misleading point estimate, the final project visualizes the predicted survival/probability curve across future training sessions.

This allows the user to see how the likelihood of a PR changes as more workouts occur.

## Benchmarks

I compared the Random Survival Forest against two simpler baselines.

### Kaplan-Meier Estimate

The Kaplan-Meier estimator provides a non-parametric estimate of the probability of remaining PR-free over time.

Unlike the RSF, it does not use predictor variables.

It therefore provides a useful baseline for determining whether incorporating workout-level features improves upon simply using the historical distribution of PR events.

### Naive Exercise-Specific Benchmark

I also created a simple exercise-specific benchmark based on the historical PR gap.

The benchmark increases probability over future session horizons using time since last PR relative to the typical PR gap for that exercise (with global fallbacks when exercise history is sparse).

The purpose of these benchmarks is to answer a fundamental question: does RSF provide useful information beyond what can be obtained from the historical PR pattern of an exercise?

## Key Findings

The project produced several interesting findings.

### 1) Workout history was substantially more useful than recovery data

The workout-based model performed better than the naive benchmark, while the Fitbit-based model only provided marginal improvement.

This suggested that my training history contained substantially more predictive information about future PRs than the recovery variables I collected.

### 2) The ensemble provided limited additional value

Although combining multiple models initially seemed attractive, Models A and B did not meaningfully improve final predictions when incorporated into Model C.

This motivated the move toward a simpler single-model architecture.

### 3) Point predictions were misleading

The RSF's estimated time-to-event predictions could be overly conservative, particularly for exercises with long historical gaps between PRs.

The survival curve provided a more useful representation of uncertainty.

### 4) Survival analysis fits the problem better than simple regression

Because some exercises have long periods without observed PRs, treating the problem as a time-to-event problem is more appropriate than predicting a fixed number of sessions until the next PR.

## Project Map (Relevant Files)

```text
health and weight training metrics/
|- app.py
|- README.md
|- requirements.txt
|- requirements_production.txt
|- data/
|  |- workouts.csv
|  |- cleaned workouts.csv
|  |- processed_merged.csv
|  |- fitbit_merged.csv
|  |- features_model1.csv
|  |- features_model2.csv
|- src/
|  |- feature_engineering.py
|  |- train_model_C.py
|  |- evaluate_model_C.py
|  |- predict_model_C.py
|  |- benchmarks/
|     |- model_c_benchmarks.py
|- utils/
|  |- data_loader.py
|  |- visualization.py
|- models/
|  |- model_C/
|  |- benchmarks/
|- outputs/
|- notebooks/
|- scripts/
|- archive/
|  |- README.md
|  |- train_models_original.py
|  |- scripts/
|  |- models/
|  |- notebooks/
```

## What src Does

The src directory contains the core modeling pipeline code:

- feature engineering and label construction from workout history
- model training for Model C (RSF)
- evaluation logic and benchmark comparison
- forecast generation and prediction utilities

In short, src is the production modeling layer that transforms historical workout data into model artifacts and prediction outputs used by the dashboard.

## How app.py Builds the Dashboard

app.py is the Streamlit entry point. It does not train models. Instead it:

1. Loads processed prediction outputs and benchmark outputs through utility loaders.
2. Loads workout history for filtering and trend visualization.
3. Lets you choose an exercise and probability threshold in the sidebar.
4. Computes summary metrics and selected-exercise cards.
5. Builds and renders probability-curve comparisons (Model C, heuristic, Kaplan-Meier).
6. Builds historical trend charts (weight progression, total volume, best-set volume).
7. Renders ranking and history tables for the selected exercise.

## Where Archived Stuff Is

Archived/legacy assets are located under archive/:

- archive/train_models_original.py for earlier training logic
- archive/scripts/ for older helper scripts
- archive/models/ for older model artifacts and experiments
- archive/notebooks/ for archived notebooks (including Untitled-1.ipynb)

## Deployment

The results are presented through a Streamlit dashboard.

The dashboard is intended to provide an interactive way to explore model predictions for individual exercises rather than display a single predicted PR date.

Because accessing the Hevy API would require a premium subscription, I currently plan to update dashboard data manually once per month instead of automatically pulling new workout data.

This is sufficient for the project's current purpose since the goal is primarily to demonstrate the modeling approach and visualize how predictions change as new workout data becomes available.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- scikit-survival
- Matplotlib
- Fitbit API
- Streamlit
- Hevy

## Project Takeaway

The main goal of this project is not to build a model that perfectly predicts the exact date of my next PR. Instead, it is an exploration of how personal longitudinal workout data can be transformed into a time-to-event prediction problem.

The progression from a multi-model ensemble to a single Random Survival Forest was itself an important part of the project. Rather than assuming that additional models or additional data would automatically improve predictions, I evaluated incremental value and simplified the architecture when results did not justify the added complexity.

The final result is a model that estimates how the probability of achieving a PR changes over future training sessions, presented through an interactive Streamlit dashboard.

## How This Can Apply in a Work Environment

While the problem in this project is fitness-related, the underlying methodology has application in business settings. Customer churn is one example of survival analysis in practice.

A company may have customers who already churned and customers who are still active during data collection. The eventual churn dates for active customers are censored observations. A Random Survival Forest can incorporate this structure while using customer characteristics to estimate churn probability over time.
