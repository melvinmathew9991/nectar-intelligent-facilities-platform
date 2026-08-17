"""
Synthetic outdoor weather generator — the root causal driver for cooling
load, chiller runtime, operating_mode, and power_consumption across the
whole telemetry generator (see physics.py).

Per site: seasonal mean (city climate) + diurnal sinusoid (peaks mid-
afternoon) + AR(1) day-to-day noise, at the same 10-min resolution as the
telemetry.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config
from .logging_config import get_logger

log = get_logger(__name__)

# Diurnal phase: sin(2*pi*(hour-PHASE)/24) peaks at hour = PHASE + 6.
# PHASE=9 -> peak at hour 15 (mid-afternoon), trough at hour 3 (pre-dawn).
DIURNAL_PHASE_HOUR = 9
AR1_PHI = 0.6


def _ar1_daily_noise(n_days: int, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """One AR(1) draw per calendar day; held constant within each day when broadcast."""
    noise = np.zeros(n_days)
    for i in range(1, n_days):
        noise[i] = AR1_PHI * noise[i - 1] + rng.normal(0.0, sigma)
    return noise


def generate_site_weather(site_cfg: dict, rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range(config.DATE_START, config.DATE_END, freq=config.FREQ)
    n = len(idx)
    hour = idx.hour.values.astype(float) + idx.minute.values.astype(float) / 60.0
    day_of_year = idx.dayofyear.values.astype(float)

    seasonal_temp = (site_cfg["temp_annual_mean"]
                      - site_cfg["temp_annual_amp"] * np.cos(2 * np.pi * day_of_year / 365.0))
    diurnal_phase = np.sin(2 * np.pi * (hour - DIURNAL_PHASE_HOUR) / 24.0)
    diurnal_temp = site_cfg["temp_diurnal_amp"] * diurnal_phase

    days = idx.normalize().unique()
    temp_day_noise = _ar1_daily_noise(len(days), sigma=1.0, rng=rng)
    temp_noise_per_ts = pd.Series(temp_day_noise, index=days).reindex(idx.normalize()).values

    outdoor_temp = (seasonal_temp + diurnal_temp + temp_noise_per_ts
                     + rng.normal(0.0, 0.3, n))

    hum_day_noise = _ar1_daily_noise(len(days), sigma=2.5, rng=rng)
    hum_noise_per_ts = pd.Series(hum_day_noise, index=days).reindex(idx.normalize()).values
    outdoor_humidity = (site_cfg["humidity_base"]
                         - site_cfg["humidity_diurnal_amp"] * diurnal_phase
                         + hum_noise_per_ts + rng.normal(0.0, 1.5, n))
    outdoor_humidity = np.clip(outdoor_humidity, 15.0, 95.0)

    return pd.DataFrame({
        "site_id": site_cfg["site_id"],
        "timestamp": idx,
        "outdoor_temp": outdoor_temp.round(2),
        "outdoor_humidity": outdoor_humidity.round(2),
    })


def generate_weather() -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    frames = [generate_site_weather(cfg, rng) for cfg in config.SITES.values()]
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Generated weather: {len(df):,} rows across {df.site_id.nunique()} sites, "
             f"{df.timestamp.min()} -> {df.timestamp.max()}")
    return df


def save_weather(df: pd.DataFrame, path: str | None = None) -> str:
    path = path or os.path.join(config.DATA_RAW_DIR, "weather.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"Saved weather.csv -> {path}")
    return path


def plot_weather_sanity_check(df: pd.DataFrame, out_path: str | None = None) -> str:
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(config.FIGURES_DIR, "00_weather_sanity.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for site_id, g in df.groupby("site_id"):
        g = g.sort_values("timestamp")
        axes[0].plot(g["timestamp"], g["outdoor_temp"], lw=0.6, label=site_id)
        day1 = g[g["timestamp"].dt.normalize() == g["timestamp"].dt.normalize().iloc[0]]
        axes[1].plot(day1["timestamp"].dt.hour + day1["timestamp"].dt.minute / 60,
                      day1["outdoor_temp"], label=site_id)
    axes[0].set(title="90-day seasonal trend (outdoor_temp)", xlabel="date", ylabel="degC")
    axes[0].legend(fontsize=8)
    axes[1].set(title="Single-day diurnal shape (outdoor_temp)", xlabel="hour", ylabel="degC")
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    log.info(f"Saved weather sanity plot -> {out_path}")
    return out_path


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    weather_df = generate_weather()
    save_weather(weather_df)
    plot_weather_sanity_check(weather_df)
