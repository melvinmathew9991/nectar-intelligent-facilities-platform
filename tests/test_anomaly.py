"""Verify each of the four anomaly detectors actually catches the failure
shape it claims to (see anomaly.py module docstring), and -- just as
important -- does NOT flag the normal diurnal cycle it must not confuse
for an anomaly."""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import anomaly as an


def _diurnal_series(n_days=20, col="power_consumption", noise_sd=1.0, seed=0,
                     asset_id="A1", freq="1h"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days * 24, freq=freq)
    hour = idx.hour
    values = 100 + 20 * np.sin(2 * np.pi * (hour - 6) / 24) + rng.normal(0, noise_sd, len(idx))
    return pd.DataFrame({"timestamp": idx, "asset_id": asset_id, col: values})


def test_rolling_mad_z_flags_isolated_spike_not_calm_region():
    rng = np.random.default_rng(1)
    s = pd.Series(10.0 + rng.normal(0, 0.1, 200))
    s.iloc[100] = 500.0
    z = an.rolling_mad_z(s, window=48)
    assert abs(z.iloc[100]) > 8
    calm = z.drop(index=range(90, 111)).dropna()
    assert (calm.abs() < 8).mean() > 0.95


def test_seasonal_mad_z_does_not_flag_normal_diurnal_cycle():
    """Regression guard: a plain rolling z-score misjudges this bimodal signal
    (see module docstring) -- seasonal_mad_z must not repeat that mistake."""
    df = _diurnal_series(n_days=20, noise_sd=1.0)
    z = an.seasonal_mad_z(df, "power_consumption")
    assert (z.abs() > 8).mean() < 0.02


def test_seasonal_mad_z_flags_genuine_deviation():
    df = _diurnal_series(n_days=20, noise_sd=1.0)
    spike_idx = 5 * 24 + 14
    df.loc[spike_idx, "power_consumption"] += 200
    z = an.seasonal_mad_z(df, "power_consumption")
    assert abs(z.loc[spike_idx]) > 8


def test_statistical_flags_detects_planted_spike_only():
    df = _diurnal_series(n_days=20, noise_sd=1.0)
    df["vibration"] = 2.0 + np.random.default_rng(2).normal(0, 0.05, len(df))
    df["temperature"] = 20.0
    spike_idx = 300
    df.loc[spike_idx, "vibration"] = 50.0

    out = an.statistical_flags(df, seasonal_cols=("power_consumption",),
                                rolling_cols=("vibration", "temperature"), z_thresh=8.0)
    assert out.loc[spike_idx, "stat_anomaly"] == 1
    assert out["stat_anomaly"].mean() < 0.05


def test_cusum_drift_flags_detects_sustained_drift_not_stable_noise():
    rng = np.random.default_rng(3)
    stable = rng.normal(0, 1, 200)
    drift = np.linspace(0, 30, 200)
    s = pd.Series(np.concatenate([stable, drift]))
    flags = an.cusum_drift_flags(s, threshold=8.0, drift=1.0)
    assert flags.iloc[:200].sum() == 0
    assert flags.iloc[200:].sum() > 0


def test_cusum_flags_all_does_not_flag_pure_diurnal_cycle():
    """Regression guard for the documented bug: raw CUSUM on a diurnal signal
    flagged the daily cycle itself >45% of the time before deseasonalizing."""
    df = _diurnal_series(n_days=20, col="temperature", noise_sd=0.5)
    out = an.cusum_flags_all(df, col="temperature")
    assert out["cusum_anomaly"].mean() < 0.05


def test_isolation_forest_flags_detects_planted_multivariate_outliers():
    rng = np.random.default_rng(4)
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "timestamp": idx, "asset_id": "CHL-01",
        "temperature": 7.0 + rng.normal(0, 0.3, n),
        "humidity": 45.0 + rng.normal(0, 2.0, n),
        "pressure": 6.5 + rng.normal(0, 0.25, n),
        "vibration": 2.2 + rng.normal(0, 0.25, n),
        "power_consumption": 180.0 + rng.normal(0, 5.0, n),
    })
    outlier_idx = [10, 100, 200]
    df.loc[outlier_idx, ["temperature", "humidity", "pressure", "vibration", "power_consumption"]] = \
        [50.0, 5.0, 40.0, 20.0, 900.0]
    metadata = pd.DataFrame([{"asset_id": "CHL-01", "asset_type": "Chiller"}])

    out = an.isolation_forest_flags(df, metadata, contamination=0.02)
    flagged_ts = set(out.loc[out["iso_anomaly"] == 1, "timestamp"])
    assert set(idx[outlier_idx]).issubset(flagged_ts)


def test_isolation_forest_flags_only_rotating_equipment():
    rng = np.random.default_rng(5)
    n = 50
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    df = pd.DataFrame({
        "timestamp": list(idx) * 2,
        "asset_id": ["CHL-01"] * n + ["ENV-01"] * n,
        "temperature": rng.normal(20, 1, n * 2),
        "humidity": rng.normal(50, 1, n * 2),
        "pressure": rng.normal(1, 0.1, n * 2),
        "vibration": rng.normal(0.02, 0.01, n * 2),
        "power_consumption": rng.normal(0.01, 0.001, n * 2),
    })
    metadata = pd.DataFrame([
        {"asset_id": "CHL-01", "asset_type": "Chiller"},
        {"asset_id": "ENV-01", "asset_type": "EnvSensor"},
    ])
    out = an.isolation_forest_flags(df, metadata)
    assert set(out["asset_id"].unique()) == {"CHL-01"}


def test_changepoint_flags_detects_regime_shift():
    idx = pd.date_range("2025-01-01", periods=5 * 24 * 6, freq="10min")
    rng = np.random.default_rng(6)
    half = len(idx) // 2
    vibration = np.concatenate([
        rng.normal(2.0, 0.1, half),
        rng.normal(6.0, 0.1, len(idx) - half),
    ])
    df = pd.DataFrame({"asset_id": "CHL-01", "timestamp": idx, "vibration": vibration})
    metadata = pd.DataFrame([{"asset_id": "CHL-01", "asset_type": "Chiller"}])

    result = an.changepoint_flags(df, metadata, col="vibration", penalty=8.0)
    assert "CHL-01" in result
    assert len(result["CHL-01"]) >= 1
    shift_time = idx[half]
    closest = min(result["CHL-01"], key=lambda t: abs((t - shift_time).total_seconds()))
    assert abs((closest - shift_time).total_seconds()) < 6 * 3600


def test_validate_against_faults_shows_higher_anomaly_rate_near_faults():
    rng = np.random.default_rng(7)
    n = 300
    idx = pd.date_range("2025-01-01", periods=n, freq="10min")
    fault_flag = np.zeros(n, dtype=int)
    fault_flag[150] = 1
    iso_anomaly = (rng.random(n) < 0.1).astype(int)
    iso_anomaly[147:154] = 1
    df = pd.DataFrame({"asset_id": "A1", "timestamp": idx, "fault_flag": fault_flag,
                        "iso_anomaly": iso_anomaly})

    result = an.validate_against_faults(df, anomaly_col="iso_anomaly", window_hours=1)
    assert result.loc[1] > result.loc[0]
