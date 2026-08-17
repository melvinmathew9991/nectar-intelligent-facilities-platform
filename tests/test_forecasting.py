"""Sanity-check forecasting.py: MAPE masking, the backtest time split, the
Holt-Winters walk-forward baseline (must beat a flat-mean baseline and never
go negative), and run_per_building's per-building history-length gate."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import forecasting as fc


def _diurnal_energy(n_days, seed=0, start="2025-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_days * 24, freq="1h")
    hour = idx.hour
    energy = 100 + 30 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, 2, len(idx))
    return idx, np.clip(energy, 0, None)


def test_mape_masks_near_zero_actuals():
    y_true = np.array([0.0, 0.5, 10.0, 20.0])
    y_pred = np.array([5.0, 5.0, 11.0, 18.0])
    result = fc.mape(y_true, y_pred)
    assert abs(result - 10.0) < 1e-6


def test_mape_returns_nan_when_all_actuals_near_zero():
    y_true = np.array([0.0, 0.5, 0.9])
    y_pred = np.array([1.0, 1.0, 1.0])
    assert np.isnan(fc.mape(y_true, y_pred))


def test_time_split_backtest_window():
    idx, energy = _diurnal_energy(60)
    df = pd.DataFrame({"timestamp": idx, "energy": energy})
    train, test = fc.time_split(df, backtest_days=14)
    cutoff = df["timestamp"].max() - pd.Timedelta(days=14)
    assert train["timestamp"].max() <= cutoff
    assert test["timestamp"].min() > cutoff
    assert len(train) + len(test) == len(df)


def test_holt_winters_walk_forward_tracks_seasonal_pattern_and_stays_nonnegative():
    idx, energy = _diurnal_energy(40, seed=1)
    s = pd.Series(energy, index=idx)
    train_end = 30 * 24
    n_test = len(s) - train_end

    preds = fc.holt_winters_walk_forward(s, train_end_idx=train_end, n_test=n_test, horizon=24)
    assert len(preds) == n_test
    assert (preds >= 0).all()

    mae = np.mean(np.abs(preds - s.values[train_end:]))
    naive_mae = np.mean(np.abs(s.values[train_end:] - s.values[:train_end].mean()))
    assert mae < naive_mae


def test_fit_primary_model_learns_linear_signal():
    rng = np.random.default_rng(2)
    n = 500
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = 3 * X["f1"] - 2 * X["f2"] + rng.normal(0, 0.1, n)

    model, name = fc.fit_primary_model(X, y)
    assert name in ("XGBoost", "HistGradientBoosting (XGBoost fallback)")
    preds = model.predict(X)
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.9


def test_run_per_building_skips_short_history_scores_sufficient_history():
    idx_long, energy_long = _diurnal_energy(45, seed=3)
    idx_short, energy_short = _diurnal_energy(10, seed=4)

    df = pd.concat([
        pd.DataFrame({"building_id": "B-LONG", "timestamp": idx_long, "energy": energy_long,
                      "f1": np.sin(np.arange(len(idx_long))), "f2": np.cos(np.arange(len(idx_long)))}),
        pd.DataFrame({"building_id": "B-SHORT", "timestamp": idx_short, "energy": energy_short,
                      "f1": np.sin(np.arange(len(idx_short))), "f2": np.cos(np.arange(len(idx_short)))}),
    ], ignore_index=True)

    metrics, preds = fc.run_per_building(df, feats=["f1", "f2"])

    assert "B-LONG" in metrics["building_id"].values
    assert "B-SHORT" not in metrics["building_id"].values
    assert "B-LONG" in preds
    assert "B-SHORT" not in preds

    row = metrics.loc[metrics["building_id"] == "B-LONG"].iloc[0]
    assert np.isfinite(row["MAE"]) and row["MAE"] >= 0
    assert np.isfinite(row["baseline_MAE"]) and row["baseline_MAE"] >= 0
