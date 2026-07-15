"""
Cleaning / resampling utilities shared by every downstream notebook and
script. Loading + missing-value handling live here once; nothing else in
the repo re-implements them.
"""
from __future__ import annotations
import os
import pandas as pd

from . import config
from .logging_config import get_logger

log = get_logger(__name__)

SENSOR_COLS = ["temperature", "humidity", "pressure", "vibration"]


def read_csv_with_parquet_cache(csv_path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Transparent Parquet mirror of a large CSV (telemetry, anomalies --
    anything with a real row count) -- same content, much faster to read
    back on every subsequent call. Regenerated automatically whenever the
    CSV is newer than the cached copy (mtime check), so it can never
    silently serve stale data if the source CSV is regenerated. Shared by
    preprocessing.load_raw() and the dashboard's anomalies.csv load --
    exactly one caching implementation, not one per caller."""
    parquet_path = os.path.splitext(csv_path)[0] + ".parquet"
    if (os.path.exists(parquet_path)
            and os.path.getmtime(parquet_path) >= os.path.getmtime(csv_path)):
        return pd.read_parquet(parquet_path)

    df = pd.read_csv(csv_path, parse_dates=parse_dates)
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        log.warning(f"Could not write parquet cache for {csv_path} ({e}); "
                    "will re-read from CSV next time")
    return df


def load_raw() -> dict[str, pd.DataFrame]:
    telemetry = read_csv_with_parquet_cache(os.path.join(config.DATA_RAW_DIR, "sensor_telemetry.csv"),
                                              parse_dates=["timestamp"])
    metadata = pd.read_csv(os.path.join(config.DATA_RAW_DIR, "asset_metadata.csv"),
                            parse_dates=["installation_date"])
    connectivity = pd.read_csv(os.path.join(config.DATA_RAW_DIR, "asset_connectivity.csv"))
    weather = pd.read_csv(os.path.join(config.DATA_RAW_DIR, "weather.csv"), parse_dates=["timestamp"])
    telemetry = telemetry.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    return {"telemetry": telemetry, "metadata": metadata, "connectivity": connectivity, "weather": weather}


def validate(telemetry: pd.DataFrame, metadata: pd.DataFrame, connectivity: pd.DataFrame) -> dict:
    """Lightweight schema / referential-integrity report, mirroring the checks
    Task 5's data-quality audit performs on the metadata/connectivity side."""
    asset_ids = set(metadata["asset_id"])
    conn_ids = set(connectivity["source_asset_id"]) | set(connectivity["target_asset_id"])
    report = {
        "telemetry_rows": len(telemetry),
        "n_assets": len(metadata),
        "n_edges": len(connectivity),
        "date_range": (telemetry["timestamp"].min(), telemetry["timestamp"].max()),
        "missing_pct": (telemetry[SENSOR_COLS].isna().mean() * 100).round(2).to_dict(),
        "fault_rate_pct": round(telemetry["fault_flag"].mean() * 100, 4),
        "orphan_assets": metadata.loc[metadata["parent_asset_id"].isna()
                                       & ~metadata["asset_id"].isin(conn_ids), "asset_id"].tolist(),
        "invalid_parent_assets": metadata.loc[metadata["parent_asset_id"].notna()
                                                & ~metadata["parent_asset_id"].isin(asset_ids),
                                                "asset_id"].tolist(),
        "duplicate_edges": int(connectivity.duplicated(["source_asset_id", "target_asset_id"]).sum()),
    }
    return report


def impute_sensors(telemetry: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """Per-asset linear interpolation (bridges short dropouts) with a global
    median fallback for any residual (e.g. all-NaN edge run) missingness."""
    cols = cols or SENSOR_COLS
    df = telemetry.sort_values(["asset_id", "timestamp"]).copy()

    def _fill(g):
        g[cols] = g[cols].interpolate(method="linear", limit_direction="both")
        return g

    df = df.groupby("asset_id", group_keys=False)[df.columns].apply(_fill)
    for c in cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    return df


def attach_weather(telemetry: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    return telemetry.merge(weather, on=["site_id", "timestamp"], how="left")


def resample_hourly(telemetry: pd.DataFrame) -> pd.DataFrame:
    """10-min -> hourly resample per asset (mean for continuous sensors, max
    for fault_flag so no fault event is averaged away)."""
    agg = {"temperature": "mean", "humidity": "mean", "pressure": "mean",
           "vibration": "mean", "power_consumption": "mean", "occupancy_count": "mean",
           "fault_flag": "max"}
    keep_cols = ["site_id", "building_id"]

    def _resample(g):
        g = g.set_index("timestamp")
        out = g[list(agg)].resample("1h").agg(agg)
        for c in keep_cols:
            out[c] = g[c].iloc[0]
        mode = g["operating_mode"].resample("1h").agg(lambda s: s.mode().iat[0] if len(s) else "Idle")
        out["operating_mode"] = mode
        return out.reset_index()

    df = telemetry.groupby("asset_id", group_keys=True)[
        telemetry.columns].apply(_resample).reset_index(level=0)
    return df.reset_index(drop=True)


def save_processed(df: pd.DataFrame, name: str) -> str:
    os.makedirs(config.DATA_PROCESSED_DIR, exist_ok=True)
    path = os.path.join(config.DATA_PROCESSED_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    log.info(f"Saved processed dataset -> {path} ({len(df):,} rows)")
    return path


def run() -> pd.DataFrame:
    raw = load_raw()
    report = validate(raw["telemetry"], raw["metadata"], raw["connectivity"])
    log.info(f"Validation report: {report}")

    clean = impute_sensors(raw["telemetry"])
    clean = attach_weather(clean, raw["weather"])
    save_processed(clean, "telemetry_clean")
    return clean


if __name__ == "__main__":
    run()
