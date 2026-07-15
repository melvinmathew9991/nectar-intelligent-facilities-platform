"""
Shared feature engineering -- imported by the Task 2 training notebook,
scripts/run_pipeline.py, and the dashboard's live scoring (exactly one
implementation of the feature contract). All rolling/lag features are
computed per-asset, sorted by time, using only past-or-present information
-- no look-ahead leakage.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .preprocessing import impute_sensors, SENSOR_COLS

STEPS_PER_HOUR = 6
MAINT_WINDOWS_HOURS = (1, 6, 24)
MAINT_SENSOR_COLS = ["temperature", "vibration", "power_consumption", "pressure"]


# ---------------------------------------------------------------------------
# Task 2: predictive-maintenance features + 24h-ahead target
# ---------------------------------------------------------------------------
def build_maintenance_features(telemetry: pd.DataFrame, metadata: pd.DataFrame,
                                 rotating_only: bool = True):
    df = telemetry.copy()

    if rotating_only:
        rot_ids = metadata.loc[metadata["asset_type"].isin(config.ROTATING_TYPES), "asset_id"]
        df = df[df["asset_id"].isin(rot_ids)].copy()

    for col in SENSOR_COLS:
        df[f"was_missing_{col}"] = df[col].isna().astype(int)

    df = impute_sensors(df, cols=SENSOR_COLS)
    df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    meta_cols = metadata[["asset_id", "asset_type", "site_id", "capacity", "installation_date"]]
    df = df.merge(meta_cols, on="asset_id", how="left", suffixes=("", "_meta"))
    df["asset_age_days"] = (df["timestamp"] - df["installation_date"]).dt.days
    df["capacity"] = df["capacity"].fillna(df["capacity"].median())

    df = pd.concat([df, pd.get_dummies(df["operating_mode"], prefix="mode", dtype=int)], axis=1)
    df = pd.concat([df, pd.get_dummies(df["asset_type"], prefix="type", dtype=int)], axis=1)

    feat_cols = []

    def _per_asset(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp")
        for col in MAINT_SENSOR_COLS:
            for w in MAINT_WINDOWS_HOURS:
                n = w * STEPS_PER_HOUR
                g[f"{col}_roll_mean_{w}h"] = g[col].rolling(n, min_periods=1).mean()
                g[f"{col}_roll_std_{w}h"] = g[col].rolling(n, min_periods=1).std().fillna(0)
                g[f"{col}_roll_min_{w}h"] = g[col].rolling(n, min_periods=1).min()
                g[f"{col}_roll_max_{w}h"] = g[col].rolling(n, min_periods=1).max()
                g[f"{col}_slope_{w}h"] = (g[col] - g[col].shift(n)) / w
            g[f"{col}_delta_1step"] = g[col].diff()
        return g

    df = (df.groupby("asset_id", group_keys=False)[df.columns]
          .apply(_per_asset).reset_index(drop=True))

    for col in MAINT_SENSOR_COLS:
        for w in MAINT_WINDOWS_HOURS:
            feat_cols += [f"{col}_roll_mean_{w}h", f"{col}_roll_std_{w}h",
                          f"{col}_roll_min_{w}h", f"{col}_roll_max_{w}h", f"{col}_slope_{w}h"]
        feat_cols.append(f"{col}_delta_1step")
    df[feat_cols] = df[feat_cols].fillna(0)

    base_feats = (["temperature", "humidity", "pressure", "vibration", "power_consumption",
                    "occupancy_count", "asset_age_days", "capacity"]
                  + [f"was_missing_{c}" for c in SENSOR_COLS])
    mode_feats = [c for c in df.columns if c.startswith("mode_")]
    type_feats = [c for c in df.columns if c.startswith("type_")]
    feature_names = base_feats + mode_feats + type_feats + feat_cols

    # ---- 24h-ahead target: does a fault occur in the NEXT 24h? (forward-looking) ----
    def _target(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("timestamp")
        fut = g["fault_flag"].values[::-1]
        fut = pd.Series(fut).rolling(24 * STEPS_PER_HOUR, min_periods=1).max().values[::-1]
        g["target_24h"] = (fut > 0).astype(int)
        return g

    df = (df.groupby("asset_id", group_keys=False)[df.columns]
          .apply(_target).reset_index(drop=True))

    return df, feature_names


# ---------------------------------------------------------------------------
# Task 3: building-level hourly energy forecast features (genuinely H-ahead)
# ---------------------------------------------------------------------------
def build_forecast_features(telemetry: pd.DataFrame, weather: pd.DataFrame,
                              horizon: int = 24, lags=(24, 48, 72, 168), roll=(24, 168)):
    """IMPORTANT (correctness): to forecast energy at time T+H using only
    information available at time T, every lag/rolling window must be
    anchored >= H hours in the past. Sub-horizon lags would be 1-step-ahead
    forecasting in disguise."""
    assert min(lags) >= horizon, "all lags must be >= forecast horizon (no leakage)"

    hourly = (telemetry.set_index("timestamp")
              .groupby("building_id")["power_consumption"]
              .resample("1h").sum().reset_index().rename(columns={"power_consumption": "energy"}))
    hourly = hourly.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    site_of_building = telemetry.drop_duplicates("building_id").set_index("building_id")["site_id"]
    hourly["site_id"] = hourly["building_id"].map(site_of_building)
    w_hourly = weather.set_index("timestamp").groupby("site_id").resample("1h").mean(numeric_only=True)
    w_hourly = w_hourly.drop(columns=["site_id"], errors="ignore").reset_index()
    hourly = hourly.merge(w_hourly, on=["site_id", "timestamp"], how="left")

    hourly["hour"] = hourly["timestamp"].dt.hour
    hourly["dow"] = hourly["timestamp"].dt.dayofweek
    hourly["is_weekend"] = (hourly["dow"] >= 5).astype(int)
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["hour"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["hour"] / 24)

    def _per_bldg(b: pd.DataFrame) -> pd.DataFrame:
        b = b.sort_values("timestamp")
        for l in lags:
            b[f"lag_{l}"] = b["energy"].shift(l)
        for w in roll:
            b[f"roll_mean_{w}"] = b["energy"].shift(horizon).rolling(w, min_periods=1).mean()
        # weather forecast is assumed known ahead (as in real NWP feeds) --
        # shifted by exactly -horizon relative to lag features is NOT needed
        # since weather at T+H is the forecast target's own timestamp, always legitimate.
        return b

    hourly = (hourly.groupby("building_id", group_keys=False)[hourly.columns]
              .apply(_per_bldg).reset_index(drop=True))

    feat_cols = (["hour", "dow", "is_weekend", "hour_sin", "hour_cos",
                  "outdoor_temp", "outdoor_humidity"]
                 + [f"lag_{l}" for l in lags] + [f"roll_mean_{w}" for w in roll])
    hourly = hourly.dropna(subset=[f"lag_{max(lags)}"]).reset_index(drop=True)
    return hourly, feat_cols
