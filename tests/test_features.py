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
