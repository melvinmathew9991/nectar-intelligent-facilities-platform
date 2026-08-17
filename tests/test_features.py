"""Verify features.py has no look-ahead leakage: rolling/lag windows only use
past-or-present data, and the 24h-ahead target is genuinely forward-looking."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import features


def _synthetic_telemetry(n_hours=48, asset_id="TEST-A-01"):
    idx = pd.date_range("2025-01-01", periods=n_hours * 6, freq="10min")
    df = pd.DataFrame({
        "timestamp": idx, "site_id": "CBE", "building_id": "CBE-B1", "asset_id": asset_id,
        "temperature": np.linspace(10, 20, len(idx)), "humidity": 50.0,
        "pressure": 5.0, "vibration": 2.0, "power_consumption": 100.0,
        "occupancy_count": 10.0, "operating_mode": "Cooling", "fault_flag": 0,
    })
    return df


def _synthetic_metadata(asset_id="TEST-A-01"):
    return pd.DataFrame([{
        "asset_id": asset_id, "site_id": "CBE", "building_id": "CBE-B1",
        "asset_name": "Chiller-TEST", "asset_type": "Chiller", "manufacturer": "Test",
        "installation_date": pd.Timestamp("2020-01-01"), "capacity": 300.0,
        "parent_asset_id": np.nan,
    }])


def _long_varying_telemetry(n_hours, asset_id, seed):
    """Genuinely time-varying (not flat) signal, long enough that truncating
    history would show up in the rolling/slope features if window-locality
    didn't hold."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_hours * 6, freq="10min")
    n = len(idx)
    t = np.arange(n)
    return pd.DataFrame({
        "timestamp": idx, "site_id": "CBE", "building_id": "CBE-B1", "asset_id": asset_id,
        "temperature": 15 + 5 * np.sin(2 * np.pi * t / (24 * 6)) + rng.normal(0, 0.3, n),
        "humidity": 50.0 + rng.normal(0, 2.0, n),
        "pressure": 5.0 + 0.5 * np.sin(2 * np.pi * t / (48 * 6)) + rng.normal(0, 0.1, n),
        "vibration": 2.0 + 0.3 * np.sin(2 * np.pi * t / (12 * 6)) + rng.normal(0, 0.05, n),
        "power_consumption": 100.0 + 20 * np.sin(2 * np.pi * t / (24 * 6)) + rng.normal(0, 2.0, n),
        "occupancy_count": 10.0, "operating_mode": "Cooling", "fault_flag": 0,
    })


def test_maintenance_target_is_forward_looking_not_backward():
    """A single fault at hour 40 should flip target_24h=1 starting exactly at
    hour 16 (40-24), not before, and should NOT stay 1 for 24h AFTER the fault
    (that would indicate a backward-looking bug)."""
    tel = _synthetic_telemetry(n_hours=48)
    meta = _synthetic_metadata()
    fault_hour_idx = 40 * 6   # row index at hour 40
    tel.loc[fault_hour_idx, "fault_flag"] = 1

    df, feats = features.build_maintenance_features(tel, meta, rotating_only=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # A 144-row forward window spans rows [t, t+143] (23h50m ahead of t, since
    # the window includes t itself) -- so the flip lands at fault_idx-143, one
    # row later than a naive 24h*6=144-row offset would suggest.
    flip_idx = fault_hour_idx - 143
    before_window = df.loc[: flip_idx - 1, "target_24h"]
    assert (before_window == 0).all(), "target_24h should be 0 more than ~24h before the fault"

    at_flip = df.loc[flip_idx, "target_24h"]
    assert at_flip == 1, "target_24h should flip to 1 at the forward-window boundary"

    long_after = df.loc[fault_hour_idx + 25 * 6:, "target_24h"]
    assert (long_after == 0).all(), "target_24h should NOT still be 1 long after the fault (no backward leakage)"


def test_maintenance_rolling_features_do_not_see_future():
    """A sensor spike placed near the END of the series must not affect
    rolling features computed at the START."""
    tel = _synthetic_telemetry(n_hours=48)
    meta = _synthetic_metadata()
    tel.loc[280:290, "vibration"] = 999.0   # spike near the end (~hour 47)

    df, feats = features.build_maintenance_features(tel, meta, rotating_only=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    early_rows = df.iloc[:100]
    assert early_rows["vibration_roll_mean_24h"].max() < 100, \
        "early rolling features leaked a future spike"
    assert early_rows["vibration_roll_max_24h"].max() < 100


def test_maintenance_no_nan_in_features():
    tel = _synthetic_telemetry(n_hours=48)
    meta = _synthetic_metadata()
    df, feats = features.build_maintenance_features(tel, meta, rotating_only=True)
    assert not df[feats].isna().any().any()


def test_forecast_lags_must_be_geq_horizon():
    with pytest.raises(AssertionError):
        features.build_forecast_features(
            _synthetic_telemetry(200), pd.DataFrame({"site_id": [], "timestamp": [],
                                                        "outdoor_temp": [], "outdoor_humidity": []}),
            horizon=24, lags=(1, 48))


def test_forecast_lags_equal_to_horizon_is_accepted():
    """min(lags) == horizon is the boundary case the assertion (>=) must
    accept, not just reject below-horizon lags."""
    tel = _synthetic_telemetry(n_hours=400)
    weather = pd.DataFrame({
        "site_id": ["CBE"] * 400, "timestamp": pd.date_range("2025-01-01", periods=400, freq="1h"),
        "outdoor_temp": 25.0, "outdoor_humidity": 50.0,
    })
    df, feats = features.build_forecast_features(tel, weather, horizon=24, lags=(24, 48))
    assert "lag_24" in feats


def test_forecast_features_no_leakage_window():
    """roll_mean_{w} at time T must be computed strictly from data at or
    before T-horizon -- verify by checking the earliest valid row is at
    least max(lags) hours after the series start."""
    tel = _synthetic_telemetry(n_hours=400)
    weather = pd.DataFrame({
        "site_id": ["CBE"] * 400, "timestamp": pd.date_range("2025-01-01", periods=400, freq="1h"),
        "outdoor_temp": 25.0, "outdoor_humidity": 50.0,
    })
    df, feats = features.build_forecast_features(tel, weather, horizon=24, lags=(24, 48), roll=(24,))
    min_ts = df["timestamp"].min()
    assert min_ts >= tel["timestamp"].min() + pd.Timedelta(hours=48)
    assert not df[feats].isna().any().any()


def test_trailing_window_matches_full_history():
    """The dashboard's live-scoring path (dashboard/app.py::load_live_failure_scores)
    trims telemetry to a trailing 36h window before calling
    build_maintenance_features, on the premise that every rolling/lag/slope
    feature there looks strictly backward within a <=24h window -- so the
    LAST row per asset (the only row live-scoring actually uses) is
    unaffected by discarding older history. Verify that claim directly:
    the full-history and trailing-window computations must produce
    byte-identical features for that last row."""
    tel = pd.concat([
        _long_varying_telemetry(n_hours=200, asset_id="TEST-A-01", seed=1),
        _long_varying_telemetry(n_hours=200, asset_id="TEST-B-01", seed=2),
    ], ignore_index=True)
    meta = pd.concat([
        _synthetic_metadata(asset_id="TEST-A-01"),
        _synthetic_metadata(asset_id="TEST-B-01"),
    ], ignore_index=True)

    df_full, feats = features.build_maintenance_features(tel, meta)
    latest_full = (df_full.sort_values("timestamp").groupby("asset_id").tail(1)
                   .set_index("asset_id").sort_index())

    cutoff = tel["timestamp"].max() - pd.Timedelta(hours=36)
    recent = tel[tel["timestamp"] >= cutoff]
    df_trunc, _ = features.build_maintenance_features(recent, meta)
    latest_trunc = (df_trunc.sort_values("timestamp").groupby("asset_id").tail(1)
                    .set_index("asset_id").sort_index())

    assert list(latest_full.index) == list(latest_trunc.index)
    for asset_id in latest_full.index:
        assert latest_full.loc[asset_id, "timestamp"] == latest_trunc.loc[asset_id, "timestamp"]
        full_vals = latest_full.loc[asset_id, feats].values.astype(float)
        trunc_vals = latest_trunc.loc[asset_id, feats].values.astype(float)
        np.testing.assert_allclose(
            full_vals, trunc_vals, rtol=1e-9, atol=1e-9,
            err_msg=f"trailing-window truncation changed live-scoring features for {asset_id}")


def test_need_target_false_skips_label_without_changing_features():
    """need_target=False (used by the dashboard's live-scoring path, which
    never uses target_24h) must skip computing the label column entirely,
    while leaving every feature value identical to the need_target=True case."""
    tel = _long_varying_telemetry(n_hours=100, asset_id="TEST-A-01", seed=3)
    meta = _synthetic_metadata(asset_id="TEST-A-01")
    tel.loc[50, "fault_flag"] = 1   # plant a fault so target_24h isn't trivially all-0

    df_with, feats_with = features.build_maintenance_features(tel, meta, need_target=True)
    df_without, feats_without = features.build_maintenance_features(tel, meta, need_target=False)

    assert "target_24h" in df_with.columns
    assert "target_24h" not in df_without.columns
    assert feats_with == feats_without

    df_with = df_with.sort_values("timestamp").reset_index(drop=True)
    df_without = df_without.sort_values("timestamp").reset_index(drop=True)
    np.testing.assert_allclose(
        df_with[feats_with].values.astype(float),
        df_without[feats_without].values.astype(float),
        rtol=1e-9, atol=1e-9,
        err_msg="need_target=False changed feature values, not just the label column")
