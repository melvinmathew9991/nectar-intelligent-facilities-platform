# Day 3 — Predictive Maintenance & Energy Forecasting (Tasks 2 & 3)

**Branch:** `session-3-maintenance-forecasting`

**Goal:** the two tasks worth the most build-effort per rubric weight (20% + 10%) and the
two that most directly demonstrate modeling judgment, not just pipeline plumbing.

## Tasks

- `maintenance_model.py` + `notebooks/03_predictive_maintenance.ipynb` — time-based
  split, 4-model comparison (LogisticRegression/RandomForest baselines vs.
  XGBoost/LightGBM), selection by **PR-AUC** (not ROC-AUC alone, given ~2% class
  imbalance), SHAP explainability, error analysis by asset type.
- `forecasting.py` + `notebooks/04_energy_forecasting.ipynb` — XGBoost primary model with
  weather-exogenous regressors vs. a walk-forward Holt-Winters baseline, evaluated over a
  trailing 2-week backtest window.
- Save `models/predictive_maintenance.pkl` for the dashboard/API to consume on Day 5.

## Files touched

```
src/nectar/maintenance_model.py
src/nectar/forecasting.py
notebooks/03_predictive_maintenance.ipynb
notebooks/04_energy_forecasting.ipynb
models/predictive_maintenance.pkl
reports/figures/09_task2_eval.png
reports/figures/10_task2_importance.png
reports/figures/11_task2_shap.png
reports/figures/12_task3_forecast.png
reports/figures/13_task3_importance.png
```

## Deliverable at end of day

Both models trained, evaluated, and explained, with a saved artifact ready for the bonus
deliverables on Day 5.

See `docs/build_log.md` §4-5 for what actually went wrong here: a ~8GB-RAM machine
repeatedly killing the training process (which turned out to be hiding two real code
bugs, only surfaced once execution moved from background to foreground), and a
Holt-Winters baseline that initially produced physically impossible negative energy
forecasts.
