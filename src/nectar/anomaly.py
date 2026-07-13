"""
Task 4 -- Anomaly Detection. Four complementary methods, each mapped to the
anomaly/fault archetypes that actually produce that shape in the data
(see physics.py): statistical thresholding (power spikes), Isolation Forest
(subtle multivariate combined states), CUSUM (slow sensor drift), and
ruptures change-point detection (degradation onset / stuck-sensor windows).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import config
from .logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Method 1: statistical thresholding (rolling robust MAD z-score) -- point
# anomalies / power spikes. Robust to the outliers it's hunting, unlike
# mean/std, which the outliers themselves distort.
# ---------------------------------------------------------------------------
def rolling_mad_z(s: pd.Series, window: int = 288) -> pd.Series:
    med = s.rolling(window, min_periods=24, center=True).median()
    mad = (s - med).abs().rolling(window, min_periods=24, center=True).median()
    return (s - med) / (1.4826 * mad + 1e-9)


def seasonal_mad_z(df: pd.DataFrame, col: str) -> pd.Series:
    """Hour-of-day-conditioned MAD z-score. A plain time-window rolling
    z-score misjudges a strongly bimodal, occupancy-driven signal (e.g.
    power_consumption: ~10% of full load when idle vs. full load when
    occupied) -- ANY window spanning a day/night transition sees most of
    its own points as "outliers" relative to that window's median, so a
    normal regime change gets flagged at a very high, useless rate.
    Comparing each reading against its own hour-of-day's typical
    distribution instead correctly separates expected diurnal shape from
    genuine anomalies."""
    hour = df["timestamp"].dt.hour
    grp = df.groupby([df["asset_id"], hour])[col]
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median())
    return (df[col] - med) / (1.4826 * mad + 1e-9)


def statistical_flags(telemetry: pd.DataFrame, seasonal_cols=("power_consumption",),
                        rolling_cols=("vibration", "temperature"),
                        z_thresh: float = 8.0) -> pd.DataFrame:
    df = telemetry.sort_values(["asset_id", "timestamp"]).copy()
    flag = pd.Series(False, index=df.index)
    for col in seasonal_cols:
        z = seasonal_mad_z(df, col)
        df[f"z_{col}"] = z
        flag = flag | (z.abs() > z_thresh)
    for col in rolling_cols:
        z = df.groupby("asset_id")[col].transform(rolling_mad_z)
        df[f"z_{col}"] = z
        flag = flag | (z.abs() > z_thresh)
    df["stat_anomaly"] = flag.astype(int)
    return df


# ---------------------------------------------------------------------------
# Method 2: Isolation Forest -- multivariate, subtle joint-pattern anomalies
# (drift, unusual sensor combinations) that a single-channel threshold misses.
# ---------------------------------------------------------------------------
def isolation_forest_flags(telemetry: pd.DataFrame, metadata: pd.DataFrame,
                              feat_cols=("temperature", "humidity", "pressure",
                                          "vibration", "power_consumption"),
                              contamination: float = 0.02) -> pd.DataFrame:
    rot_ids = metadata.loc[metadata["asset_type"].isin(config.ROTATING_TYPES), "asset_id"]
    rot = telemetry[telemetry["asset_id"].isin(rot_ids)].copy()
    rot[list(feat_cols)] = rot[list(feat_cols)].fillna(rot[list(feat_cols)].median())
    iso = IsolationForest(contamination=contamination, n_estimators=200, random_state=config.SEED)
    rot["iso_anomaly"] = (iso.fit_predict(rot[list(feat_cols)]) == -1).astype(int)
    rot["iso_score"] = -iso.score_samples(rot[list(feat_cols)])
    return rot


# ---------------------------------------------------------------------------
# Method 3: CUSUM -- cumulative-sum drift detector. Catches slow, sustained
# sensor drift that a point-threshold method is specifically NOT designed to
# catch (drift never produces a single big jump).
# ---------------------------------------------------------------------------
def cusum_drift_flags(s: pd.Series, threshold: float = 8.0, drift: float = 1.0) -> pd.Series:
    """Expects a DESEASONALIZED residual (see cusum_flags_all) -- running
    CUSUM on a raw diurnal signal with a fixed single-day baseline flags
    the recurring daily cycle itself as "drift" every single day (verified:
    >45% flag rate on raw temperature). CUSUM is the right tool only once
    the expected daily shape has already been removed.

    Standard CUSUM: the accumulator RESETS to 0 once it crosses the
    threshold (an unreset accumulator stays elevated indefinitely after one
    alarm, which was flagging 30-90%+ of rows depending on asset type)."""
    x = s.ffill().bfill().values
    std0 = np.nanstd(x) + 1e-9   # whole-series scale, not a possibly-atypical first day
    z = x / std0
    pos = neg = 0.0
    flags = np.zeros(len(z), dtype=int)
    for i in range(len(z)):
        pos = max(0.0, pos + z[i] - drift)
        neg = min(0.0, neg + z[i] + drift)
        if pos > threshold or neg < -threshold:
            flags[i] = 1
            pos = neg = 0.0
    return pd.Series(flags, index=s.index)


def cusum_flags_all(telemetry: pd.DataFrame, col: str = "temperature") -> pd.DataFrame:
    df = telemetry.sort_values(["asset_id", "timestamp"]).copy()
    hour = df["timestamp"].dt.hour
    seasonal_baseline = df.groupby([df["asset_id"], hour])[col].transform("median")
    df[f"{col}_resid"] = df[col] - seasonal_baseline
    df["cusum_anomaly"] = df.groupby("asset_id")[f"{col}_resid"].transform(cusum_drift_flags)
    return df


# ---------------------------------------------------------------------------
# Method 4: ruptures change-point detection -- degradation onset. Run per
# rotating asset on hourly-resampled vibration (10-min resolution over 90
# days is unnecessarily fine-grained for change-point detection and much
# slower); flags timestamps where the underlying signal's statistical
# regime shifts.
# ---------------------------------------------------------------------------
def changepoint_flags(telemetry: pd.DataFrame, metadata: pd.DataFrame,
                        col: str = "vibration", penalty: float = 8.0) -> dict[str, list[pd.Timestamp]]:
    import ruptures as rpt

    rot_ids = metadata.loc[metadata["asset_type"].isin(config.ROTATING_TYPES), "asset_id"]
    results = {}
    for aid in rot_ids:
        g = telemetry[telemetry.asset_id == aid].sort_values("timestamp")
        hourly = g.set_index("timestamp")[col].resample("1h").mean().ffill()
        if hourly.isna().all() or len(hourly) < 48:
            continue
        # Binseg, not Pelt(jump=1): verified ~300x faster (0.1s vs ~37s per
        # asset) on this data with comparable breakpoint counts -- Pelt's
        # exact search was costing over 50 minutes across all rotating assets.
        algo = rpt.Binseg(model="l2", min_size=6, jump=1).fit(hourly.values.reshape(-1, 1))
        try:
            bkps = algo.predict(pen=penalty)
        except Exception:
            continue
        results[aid] = [hourly.index[i] for i in bkps[:-1]]   # drop final (series-end) breakpoint
    return results


# ---------------------------------------------------------------------------
# Validation: does anomaly density rise near known faults? (closes the loop
# with Task 2 -- confirms the detectors mean something, not just noise)
# ---------------------------------------------------------------------------
def validate_against_faults(rot_with_flags: pd.DataFrame, anomaly_col: str = "iso_anomaly",
                               window_hours: int = 24) -> pd.Series:
    df = rot_with_flags.sort_values(["asset_id", "timestamp"]).copy()
    window = window_hours * 6
    df["near_fault"] = df.groupby("asset_id")["fault_flag"].transform(
        lambda s: s.rolling(window, min_periods=1, center=True).max())
    return df.groupby("near_fault")[anomaly_col].mean()
