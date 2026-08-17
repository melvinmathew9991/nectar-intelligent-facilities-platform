"""
Per-asset-type physical signal models, fault archetypes, and graph-coupled
load propagation for the Nectar synthetic telemetry generator.

Vectorized shared-driver approach (see PLAN.md 1.4): one building-level
"cooling load index" time series (from outdoor_temp + occupancy) drives
every asset's signal via its position in the hierarchy, plus asset-specific
noise/fault/anomaly injection on top. This makes a chiller and its child
AHUs move together physically without an iterative control-loop simulator.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Building-level context: occupancy, cooling load index, operating mode
# ---------------------------------------------------------------------------
def occupancy_fraction(timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Weekday office bump in [0,1]: ramps 9am->1pm->6pm, ~0 outside, damped on weekends."""
    hour = timestamps.hour.values.astype(float) + timestamps.minute.values.astype(float) / 60.0
    start, end = config.OCCUPANCY_START_HOUR, config.OCCUPANCY_END_HOUR
    span = end - start
    raw = np.clip(np.sin(np.pi * (hour - start) / span), 0.0, 1.0)
    raw = np.where((hour >= start) & (hour <= end), raw, 0.0)
    is_weekend = timestamps.dayofweek.values >= 5
    factor = np.where(is_weekend, config.WEEKEND_OCCUPANCY_FACTOR, 1.0)
    return raw * factor


def build_building_context(building_ids: list[str], site_of_building: dict,
                            weather_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """One row per (building_id, timestamp): outdoor conditions, occupancy,
    cooling_load_index, operating_mode. Shared by every asset in that building."""
    idx = pd.date_range(config.DATE_START, config.DATE_END, freq=config.FREQ)
    occ_frac = occupancy_fraction(idx)

    frames = []
    for b in building_ids:
        site_id = site_of_building[b]
        w = weather_df[weather_df.site_id == site_id].sort_values("timestamp").reset_index(drop=True)
        assert len(w) == len(idx), "weather series must align with the telemetry time index"

        max_headcount = rng.uniform(60, 160)
        occupancy_count = occ_frac * max_headcount * rng.uniform(0.9, 1.1, len(idx))

        temp_excess = np.clip((w["outdoor_temp"].values - 22.0) / 15.0, 0.0, 1.0)
        cooling_load = np.clip(0.6 * temp_excess + 0.4 * occ_frac, 0.0, 1.0)

        is_occupied = occ_frac > 0.05
        # Coldest hours (diurnal trough) fall overnight, before the building
        # is occupied -- a fixed absolute threshold would (almost) never
        # coincide with occupied hours. Use a per-site threshold calibrated
        # on occupied-hour temperatures instead, so the rare, genuine cold
        # snap (e.g. a Jan morning in Bangalore) actually produces a small
        # but real Heating occurrence (<1% of rows, per PLAN.md's assumption
        # of a South-Indian-climate edge case -- not a real winter season).
        occ_hour_temps = w["outdoor_temp"].values[is_occupied]
        cold_threshold = np.quantile(occ_hour_temps, 0.02) if len(occ_hour_temps) else -np.inf
        is_cold = w["outdoor_temp"].values < cold_threshold
        operating_mode = np.where(is_occupied & is_cold, "Heating",
                          np.where(is_occupied & ~is_cold, "Cooling", "Idle"))

        frames.append(pd.DataFrame({
            "building_id": b,
            "site_id": site_id,
            "timestamp": idx,
            "outdoor_temp": w["outdoor_temp"].values,
            "outdoor_humidity": w["outdoor_humidity"].values,
            "occupancy_frac": occ_frac,
            "occupancy_count": occupancy_count,
            "cooling_load_index": cooling_load,
            "operating_mode": operating_mode,
        }))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Per-asset-type baseline signal generation (vectorized over one asset's
# full timeline). Returns a dict of numpy arrays, no faults/anomalies yet.
# ---------------------------------------------------------------------------
def _part_load_power_curve(load_index: np.ndarray, kind: str) -> np.ndarray:
    if kind == "chiller":
        return 0.15 + 0.85 * load_index ** 1.3           # non-linear part-load ratio
    if kind == "fan_affinity":
        return 0.05 + 0.95 * load_index ** 3               # fan affinity law (P ~ speed^3)
    if kind == "pump_affinity":
        return 0.1 + 0.9 * load_index ** 2.2                # pump affinity law
    raise ValueError(kind)


def generate_asset_baseline(asset_type: str, ctx: pd.DataFrame, age_years: float,
                             rng: np.random.Generator) -> dict:
    n = len(ctx)
    p = config.SIGNAL_PARAMS[asset_type]
    load = ctx["cooling_load_index"].values
    mode = ctx["operating_mode"].values
    is_idle = mode == "Idle"

    age_wear = np.clip(age_years / 10.0, 0.0, 1.0)   # older assets run slightly hotter/noisier

    if asset_type == "Chiller":
        temperature = (p["temperature"]["baseline"] + 0.4 * age_wear
                        + rng.normal(0, p["temperature"]["noise_sd"], n)
                        + np.where(is_idle, ctx["outdoor_temp"].values * 0.05, 0.0))
        pressure = (p["pressure"]["baseline"] + 3.0 * load + 0.5 * age_wear
                     + rng.normal(0, p["pressure"]["noise_sd"], n))
        vibration = (p["vibration"]["baseline"] + 1.2 * load + 0.8 * age_wear
                      + rng.normal(0, p["vibration"]["noise_sd"], n))
        power = config.POWER_BASELINE["Chiller"] * _part_load_power_curve(load, "chiller")
        power = np.where(is_idle, power * 0.08, power)
        humidity = p["humidity"]["baseline"] + rng.normal(0, p["humidity"]["noise_sd"], n)

    elif asset_type == "AHU":
        temperature = (p["temperature"]["baseline"] + 3.0 * (1 - load)
                        + rng.normal(0, p["temperature"]["noise_sd"], n))
        pressure = (p["pressure"]["baseline"] + 150.0 * load + 30.0 * age_wear
                     + rng.normal(0, p["pressure"]["noise_sd"], n))
        vibration = (p["vibration"]["baseline"] + 0.9 * load + 0.6 * age_wear
                      + rng.normal(0, p["vibration"]["noise_sd"], n))
        power = config.POWER_BASELINE["AHU"] * _part_load_power_curve(load, "fan_affinity")
        power = np.where(is_idle, power * 0.15, power)
        humidity = (ctx["outdoor_humidity"].values * 0.5 + p["humidity"]["baseline"] * 0.5
                     + rng.normal(0, p["humidity"]["noise_sd"], n))

    elif asset_type == "Pump":
        temperature = (p["temperature"]["baseline"] + 2.0 * load
                        + rng.normal(0, p["temperature"]["noise_sd"], n))
        pressure = (p["pressure"]["baseline"] + 100.0 * load + 20.0 * age_wear
                     + rng.normal(0, p["pressure"]["noise_sd"], n))
        vibration = (p["vibration"]["baseline"] + 1.0 * load + 0.7 * age_wear
                      + rng.normal(0, p["vibration"]["noise_sd"], n))
        power = config.POWER_BASELINE["Pump"] * _part_load_power_curve(load, "pump_affinity")
        power = np.where(is_idle, power * 0.1, power)
        humidity = p["humidity"]["baseline"] + rng.normal(0, p["humidity"]["noise_sd"], n)

    elif asset_type == "EnvSensor":
        temperature = (p["temperature"]["baseline"] + 1.5 * (load - 0.5)
                        + rng.normal(0, p["temperature"]["noise_sd"], n))
        pressure = p["pressure"]["baseline"] + rng.normal(0, p["pressure"]["noise_sd"], n)
        vibration = np.abs(rng.normal(p["vibration"]["baseline"], p["vibration"]["noise_sd"], n))
        power = np.abs(rng.normal(config.POWER_BASELINE["EnvSensor"], 0.005, n))
        humidity = (ctx["outdoor_humidity"].values * 0.4 + p["humidity"]["baseline"] * 0.6
                     + rng.normal(0, p["humidity"]["noise_sd"], n))

    elif asset_type == "EnergyMeter":
        temperature = p["temperature"]["baseline"] + rng.normal(0, p["temperature"]["noise_sd"], n)
        pressure = p["pressure"]["baseline"] + rng.normal(0, p["pressure"]["noise_sd"], n)
        vibration = np.abs(rng.normal(p["vibration"]["baseline"], p["vibration"]["noise_sd"], n))
        power = np.full(n, np.nan)   # filled later from monitored asset(s)
        humidity = p["humidity"]["baseline"] + rng.normal(0, p["humidity"]["noise_sd"], n)

    else:
        raise ValueError(f"unknown asset_type {asset_type}")

    return {
        "temperature": temperature, "humidity": humidity, "pressure": pressure,
        "vibration": vibration, "power_consumption": power,
    }


# ---------------------------------------------------------------------------
# Fault archetypes: ramp shape (0->1 over ramp_hours) applied to specific
# channels, direction depending on archetype. Returns the per-channel delta
# arrays (same length as the ramp+duration window) to add onto baseline.
# ---------------------------------------------------------------------------
ARCHETYPE_EFFECTS = {
    # channel: signed direction/weight (sign = direction of drift; magnitude is a
    # relative weight, not a fraction of baseline range -- the actual absolute
    # delta at ramp peak comes from ARCHETYPE_MAGNITUDE below)
    "refrigerant_leak":   {"pressure": -1.0, "temperature": 0.6, "power": -0.15},
    "condenser_fouling":  {"pressure": 1.0, "power": 0.35},
    "filter_clogging":    {"pressure": 1.0},
    "fan_bearing_wear":   {"vibration": 1.0},
    "cavitation":         {"vibration": 1.0, "pressure": -0.4},
    "bearing_wear":       {"vibration": 1.0},
}

ARCHETYPE_MAGNITUDE = {   # absolute magnitude at ramp peak, per channel/unit
    "refrigerant_leak":  {"pressure": 3.5, "temperature": 4.0, "power": 25.0},
    "condenser_fouling": {"pressure": 2.5, "power": 60.0},
    "filter_clogging":   {"pressure": 220.0},
    "fan_bearing_wear":  {"vibration": 4.5},
    "cavitation":        {"vibration": 3.5, "pressure": -90.0},
    "bearing_wear":      {"vibration": 4.0},
}


def ramp_shape(ramp_hours: int, duration_hours: int, steps_per_hour: int) -> np.ndarray:
    """Monotonic 0->1 ramp over ramp_hours, then holds at 1.0 for duration_hours
    (the fault-flagged window) — unlike a naive slice, this INCLUDES the fault
    rows at peak severity (no snap-back artifact)."""
    ramp_n = ramp_hours * steps_per_hour
    dur_n = duration_hours * steps_per_hour
    ramp = np.linspace(0.0, 1.0, ramp_n, endpoint=False)
    hold = np.ones(dur_n)
    return np.concatenate([ramp, hold])


def apply_fault_episode(series: dict, start_idx: int, archetype: str,
                          ramp_hours: int, duration_hours: int, steps_per_hour: int,
                          scale: float = 1.0) -> tuple[int, int]:
    """Mutates `series` in place, adding the archetype's ramp effect starting
    at start_idx. Returns (fault_start_idx, fault_end_idx) = the rows where
    fault_flag should be set to 1 (only when scale==1.0, i.e. the primary asset)."""
    shape = ramp_shape(ramp_hours, duration_hours, steps_per_hour)
    n = len(shape)
    end = min(start_idx + n, len(series["temperature"]))
    shape = shape[: end - start_idx]

    effects = ARCHETYPE_EFFECTS[archetype]
    magnitudes = ARCHETYPE_MAGNITUDE[archetype]
    channel_map = {"pressure": "pressure", "temperature": "temperature",
                    "vibration": "vibration", "power": "power_consumption"}
    for ch, direction in effects.items():
        arr_key = channel_map[ch]
        delta = direction * magnitudes[ch] * shape * scale
        series[arr_key][start_idx:end] += delta

    ramp_n = ramp_hours * steps_per_hour
    fault_start = start_idx + ramp_n
    fault_end = end
    return fault_start, fault_end


# ---------------------------------------------------------------------------
# Standalone anomalies (independent of fault archetypes) — Task 4 target
# ---------------------------------------------------------------------------
def inject_power_spikes(power: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_spikes = rng.integers(*config.ANOMALY_SPIKE_COUNT_PER_ASSET)
    idxs = rng.choice(len(power), size=min(n_spikes, len(power)), replace=False)
    mult = rng.uniform(*config.ANOMALY_SPIKE_MULTIPLIER, size=len(idxs))
    power = power.copy()
    power[idxs] = power[idxs] * mult
    return power, idxs


def inject_slow_drift(values: np.ndarray, rng: np.random.Generator,
                        window_hours: int = 200, steps_per_hour: int = 6) -> tuple[np.ndarray, slice]:
    n = window_hours * steps_per_hour
    if n >= len(values):
        n = len(values) // 2
    start = int(rng.integers(0, len(values) - n))
    drift_total = rng.uniform(2.0, 5.0) * rng.choice([-1, 1])
    drift = np.linspace(0.0, drift_total, n)
    out = values.copy()
    out[start:start + n] += drift
    return out, slice(start, start + n)


def inject_stuck_sensor(values: np.ndarray, rng: np.random.Generator,
                          duration_hours_range=(3, 12), steps_per_hour: int = 6) -> tuple[np.ndarray, slice]:
    dur = int(rng.integers(*duration_hours_range) * steps_per_hour)
    start = int(rng.integers(0, max(1, len(values) - dur)))
    stuck_value = values[start]
    out = values.copy()
    out[start:start + dur] = stuck_value
    return out, slice(start, start + dur)
