"""
Task 3 -- Energy Consumption Forecasting. Per-building 24h-ahead hourly
power forecast: a statsmodels Holt-Winters baseline (documented trade-off
vs Prophet -- avoids Windows Stan-compiler friction) plus an XGBoost/
HistGradientBoosting primary model using calendar + lag/rolling + weather
exogenous features from features.build_forecast_features(). Evaluated via
walk-forward backtest over the trailing ~2 weeks of the 90-day window.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from . import config
from .logging_config import get_logger

log = get_logger(__name__)

BACKTEST_DAYS = 14   # trailing ~2 weeks


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true > 1.0   # avoid division blow-up on near-zero (idle) hours
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def time_split(hourly: pd.DataFrame, backtest_days: int = BACKTEST_DAYS):
    cutoff = hourly["timestamp"].max() - pd.Timedelta(days=backtest_days)
    train = hourly[hourly["timestamp"] <= cutoff]
    test = hourly[hourly["timestamp"] > cutoff]
    return train, test


def holt_winters_walk_forward(full_series: pd.Series, train_end_idx: int,
                                n_test: int, horizon: int = 24) -> np.ndarray:
    """Genuine walk-forward baseline: refit on an EXPANDING window and
    forecast only `horizon` (24h) steps ahead at a time, then advance by
    `horizon` and repeat -- a one-shot multi-week-ahead Holt-Winters
    forecast compounds trend error catastrophically over hundreds of
    steps, which is not what "24h-ahead forecasting" means."""
    preds = np.zeros(n_test)
    values = full_series.values
    origin = train_end_idx
    filled = 0
    while filled < n_test:
        step = min(horizon, n_test - filled)
        window = values[:origin]
        try:
            # No trend component: a 90-day window has no reliable long-term
            # trend, and an additive trend compounds badly on this bursty,
            # occupancy-driven signal (verified: trend='add' occasionally
            # forecasts physically-impossible negative energy). Seasonal-only
            # + non-negative clipping is a stable, still-genuinely-weaker
            # baseline than the primary model (the point of having one).
            model = ExponentialSmoothing(window, trend=None, seasonal="add",
                                           seasonal_periods=24,
                                           initialization_method="estimated")
            fit = model.fit(optimized=True)
            preds[filled:filled + step] = np.clip(fit.forecast(step), 0, None)
        except Exception:
            preds[filled:filled + step] = window[-24:].mean()
        origin += step
        filled += step
    return preds


def fit_primary_model(Xtr, ytr):
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=400, learning_rate=0.06, max_depth=6,
                              subsample=0.9, colsample_bytree=0.9,
                              random_state=config.SEED, n_jobs=-1)
        model.fit(Xtr, ytr)
        return model, "XGBoost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                                max_depth=6, random_state=config.SEED)
        model.fit(Xtr, ytr)
        return model, "HistGradientBoosting (XGBoost fallback)"


def run_per_building(hourly: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, dict]:
    """Returns (metrics_df, predictions_by_building) -- predictions_by_building
    maps building_id -> DataFrame(timestamp, actual, baseline_pred, primary_pred)."""
    train, test = time_split(hourly)
    rows = []
    preds = {}

    for building_id, g in hourly.groupby("building_id"):
        g = g.sort_values("timestamp").reset_index(drop=True)
        tr = g[g["timestamp"] <= train["timestamp"].max()]
        te = g[g["timestamp"] > train["timestamp"].max()]
        if len(te) == 0 or len(tr) < 24 * 14:
            continue

        model, model_name = fit_primary_model(tr[feats], tr["energy"])
        primary_pred = model.predict(te[feats])

        try:
            baseline_pred = holt_winters_walk_forward(
                g["energy"], train_end_idx=len(tr), n_test=len(te), horizon=24)
        except Exception as e:
            log.warning(f"Holt-Winters failed for {building_id}: {e}")
            baseline_pred = np.full(len(te), tr["energy"].mean())

        rows.append({
            "building_id": building_id, "model": model_name,
            "MAE": mean_absolute_error(te["energy"], primary_pred),
            "RMSE": np.sqrt(mean_squared_error(te["energy"], primary_pred)),
            "MAPE": mape(te["energy"].values, primary_pred),
            "baseline_MAE": mean_absolute_error(te["energy"], baseline_pred),
            "baseline_RMSE": np.sqrt(mean_squared_error(te["energy"], baseline_pred)),
            "baseline_MAPE": mape(te["energy"].values, baseline_pred),
        })
        preds[building_id] = pd.DataFrame({
            "timestamp": te["timestamp"].values, "actual": te["energy"].values,
            "primary_pred": primary_pred, "baseline_pred": np.asarray(baseline_pred),
        })

    return pd.DataFrame(rows), preds
