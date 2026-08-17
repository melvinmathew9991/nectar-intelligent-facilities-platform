"""
Central configuration for the Nectar synthetic IoT dataset + pipeline.

Single source of truth for: sites/buildings/asset-hierarchy shape, the
simulation date range, RNG seeding, and per-asset-type physical parameter
tables (used by physics.py to generate realistic signal profiles and by
docs/data_dictionary.md to document them).
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
# Committed trailing slice of the full dataset (see scripts/build_demo_slice.py).
# Small enough to live in git, so the dashboard runs on a hosted deployment where
# the gitignored full-size telemetry CSV isn't present.
DATA_DEMO_DIR = os.path.join(ROOT_DIR, "data", "demo")
# 4 days, not more: Streamlit Community Cloud caps container memory (~2.7GB) and a
# 14-day slice was enough to blank the hosted app on load. 4 days still comfortably
# covers everything the dashboard computes -- live scoring reads a trailing 36h, and
# the widest rolling feature is 24h -- while cutting the loaded frame by ~70%. The
# dashboard never calls build_forecast_features (which would need lag_168 = 7 days).
DEMO_DAYS = 4
# The slice ends here rather than at the dataset's tail. The dashboard's live
# scoring reads each asset's most recent 36h, and the final 36h of the full
# dataset happens to contain no imminent faults -- every asset scores 0.10-0.27
# against a 0.569 threshold, so the failure-prediction panel renders empty.
# This timestamp was selected as the hour with the highest count of rotating-asset
# fault onsets in the following 24h (see scripts/build_demo_slice.py), giving a
# hosted demo that shows the model actually firing. It is a different *moment*,
# not different data or a different model.
DEMO_END = "2025-02-15 15:00:00"
MODELS_DIR = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

# ---------------------------------------------------------------------------
# Time range: 90 days, 10-minute resolution
# ---------------------------------------------------------------------------
DATE_START = "2025-01-01"
DATE_END = "2025-03-31 23:50:00"
FREQ = "10min"

# ---------------------------------------------------------------------------
# Sites: 3 South Indian cities with genuinely distinct climate profiles.
# temp_annual_mean/amplitude drive a seasonal sinusoid; diurnal_amplitude
# drives the day/night swing; humidity_base/amplitude analogous.
# ---------------------------------------------------------------------------
SITES = {
    "CBE": {
        "site_id": "CBE",
        "name": "Coimbatore",
        "temp_annual_mean": 27.0,   # deg C, moderate
        "temp_annual_amp": 2.5,     # seasonal swing (Jan cooler than Mar)
        "temp_diurnal_amp": 6.0,
        "humidity_base": 62.0,
        "humidity_diurnal_amp": 15.0,
    },
    "CHN": {
        "site_id": "CHN",
        "name": "Chennai",
        "temp_annual_mean": 30.5,   # hot/humid
        "temp_annual_amp": 3.0,
        "temp_diurnal_amp": 5.5,
        "humidity_base": 72.0,
        "humidity_diurnal_amp": 12.0,
    },
    "BLR": {
        "site_id": "BLR",
        "name": "Bangalore",
        "temp_annual_mean": 23.5,   # mild
        "temp_annual_amp": 2.0,
        "temp_diurnal_amp": 7.0,
        "humidity_base": 58.0,
        "humidity_diurnal_amp": 16.0,
    },
}
SITE_IDS = list(SITES.keys())
BUILDINGS_PER_SITE = 3

# ---------------------------------------------------------------------------
# Asset hierarchy shape (per building). Randomized within ranges (see
# data_generation.py::build_metadata) to land ~15-18 assets/building
# (~150-160 total across 9 buildings) while keeping every building's tree
# structurally consistent: Chiller -> AHUs -> EnvSensor, Chiller -> Pump ->
# EnergyMeter, plus one building-level main EnergyMeter.
# ---------------------------------------------------------------------------
# Mirrors the brief's own example: Chiller and Pump are SIBLING roots within
# a building (not Pump nested under Chiller) -- Chiller -> AHUs -> EnvSensor,
# Pump -> EnergyMeter (submeter), plus one building-level main EnergyMeter.
CHILLERS_PER_BUILDING = 2           # fixed
AHUS_PER_CHILLER = (2, 3)           # inclusive randint range, per chiller
PUMPS_PER_BUILDING = 2              # fixed
ENVSENSORS_PER_AHU = 1
SUBMETER_PER_PUMP = 1               # EnergyMeter monitoring each pump
MAIN_METERS_PER_BUILDING = 1

ASSET_TYPES = ["Chiller", "AHU", "Pump", "EnergyMeter", "EnvSensor"]
ROTATING_TYPES = ["Chiller", "AHU", "Pump"]   # can develop mechanical faults

MANUFACTURERS = {
    "Chiller": ["Carrier", "Trane", "York", "Daikin"],
    "AHU": ["Trane", "Daikin", "Voltas", "Blue Star"],
    "Pump": ["Grundfos", "KSB", "Wilo"],
    "EnergyMeter": ["Schneider Electric", "Siemens", "L&T"],
    "EnvSensor": ["Honeywell", "Siemens", "Johnson Controls"],
}

CAPACITY_RANGES = {   # rated capacity, unit varies by type (see data_dictionary.md)
    "Chiller": (250, 500),     # tons of refrigeration
    "AHU": (40, 150),          # kCFM airflow-equivalent rating
    "Pump": (20, 80),          # kW rated
    "EnergyMeter": None,       # not physically meaningful -> NaN
    "EnvSensor": None,
}

INSTALL_AGE_YEARS_RANGE = (0.5, 8.0)   # asset age at simulation start

# ---------------------------------------------------------------------------
# Per-asset-type physical signal parameter tables (see docs/data_dictionary.md
# for the standards/units this is grounded in: ISO 10816-3 vibration severity
# bands, typical chilled-water loop pressures, ASHRAE supply-air ranges).
# ---------------------------------------------------------------------------
SIGNAL_PARAMS = {
    "Chiller": {
        "temperature": {"baseline": 7.0, "noise_sd": 0.3, "unit": "degC (chilled water supply)"},
        "pressure": {"baseline": 6.5, "range": (4.0, 12.0), "noise_sd": 0.25, "unit": "bar (refrigerant)"},
        "vibration": {"baseline": 2.2, "noise_sd": 0.25, "unit": "mm/s RMS (ISO 10816-3, compressor)"},
        "humidity": {"baseline": 45.0, "noise_sd": 2.0, "unit": "% (near-ambient, plant room)"},
    },
    "AHU": {
        "temperature": {"baseline": 14.0, "noise_sd": 0.6, "unit": "degC (supply air)"},
        "pressure": {"baseline": 300.0, "range": (100.0, 500.0), "noise_sd": 20.0, "unit": "Pa (static)"},
        "vibration": {"baseline": 1.4, "noise_sd": 0.2, "unit": "mm/s RMS (ISO 10816, fan/bearing)"},
        "humidity": {"baseline": 50.0, "noise_sd": 4.0, "unit": "% RH (supply/return air)"},
    },
    "Pump": {
        "temperature": {"baseline": 12.0, "noise_sd": 0.5, "unit": "degC (loop fluid)"},
        "pressure": {"baseline": 250.0, "range": (80.0, 400.0), "noise_sd": 15.0, "unit": "kPa (discharge)"},
        "vibration": {"baseline": 1.8, "noise_sd": 0.3, "unit": "mm/s RMS (ISO 10816, bearing/cavitation)"},
        "humidity": {"baseline": 48.0, "noise_sd": 2.0, "unit": "% (near-ambient, plant room)"},
    },
    "EnergyMeter": {
        "temperature": {"baseline": 28.0, "noise_sd": 1.0, "unit": "degC (panel ambient)"},
        "pressure": {"baseline": 1.0, "noise_sd": 0.02, "unit": "n/a (near-constant placeholder)"},
        "vibration": {"baseline": 0.05, "noise_sd": 0.02, "unit": "mm/s (near-zero baseline)"},
        "humidity": {"baseline": 50.0, "noise_sd": 3.0, "unit": "% (panel ambient)"},
    },
    "EnvSensor": {
        "temperature": {"baseline": 23.5, "noise_sd": 0.8, "unit": "degC (zone, ASHRAE comfort band 22-26)"},
        "pressure": {"baseline": 1.0, "noise_sd": 0.02, "unit": "n/a (near-ambient placeholder)"},
        "vibration": {"baseline": 0.02, "noise_sd": 0.01, "unit": "mm/s (n/a, no moving parts)"},
        "humidity": {"baseline": 50.0, "noise_sd": 5.0, "unit": "% RH (zone)"},
    },
}

# Power model: baseline load (kW) at full part-load, per asset type
POWER_BASELINE = {
    "Chiller": 180.0,
    "AHU": 18.0,
    "Pump": 12.0,
    "EnergyMeter": 0.05,   # parasitic/panel load only
    "EnvSensor": 0.01,
}

# ---------------------------------------------------------------------------
# Occupancy (weekday office curve, building-level, shared across assets)
# ---------------------------------------------------------------------------
OCCUPANCY_PEAK_HOUR = 13
OCCUPANCY_START_HOUR = 9
OCCUPANCY_END_HOUR = 18
WEEKEND_OCCUPANCY_FACTOR = 0.05

# ---------------------------------------------------------------------------
# Fault archetypes (rotating equipment only): each episode has a leading-
# indicator degradation ramp (FAULT_RAMP_HOURS, upper bound exclusive -- actual
# max is 47h, not 48) before a short fault_flag=1 event.
# ---------------------------------------------------------------------------
FAULT_TARGET_EPISODES = (150, 250)   # NOT wired into the generator as a target --
                                       # only used as a loose tolerance band by
                                       # test_fault_episode_count_in_range. Actual
                                       # episode count is emergent from the
                                       # per-asset age-driven binomial draw below.
FAULT_RAMP_HOURS = (6, 48)
FAULT_DURATION_HOURS = (1, 3)

FAULT_ARCHETYPES = {
    "Chiller": ["refrigerant_leak", "condenser_fouling"],
    "AHU": ["filter_clogging", "fan_bearing_wear"],
    "Pump": ["cavitation", "bearing_wear"],
}
# Secondary (smaller) effect applied to direct downstream assets during a
# parent Chiller/Pump fault, via the connectivity graph.
DOWNSTREAM_FAULT_EFFECT_SCALE = 0.25

# ---------------------------------------------------------------------------
# Standalone anomalies (Task 4 target, independent of fault archetypes)
# ---------------------------------------------------------------------------
ANOMALY_SPIKE_COUNT_PER_ASSET = (2, 5)
ANOMALY_SPIKE_MULTIPLIER = (2.5, 4.0)
ANOMALY_DRIFT_ASSET_FRACTION = 0.20
ANOMALY_STUCK_ASSET_FRACTION = 0.05
ANOMALY_STUCK_DURATION_HOURS = (3, 12)

# ---------------------------------------------------------------------------
# Missingness & injected data-quality issues
# ---------------------------------------------------------------------------
TELEMETRY_MISSING_FRAC = (0.02, 0.05)   # random per-cell gaps
TELEMETRY_OUTAGE_WINDOWS_PER_SITE = (1, 3)  # a few longer per-asset outages
TELEMETRY_OUTAGE_DURATION_HOURS = (4, 24)

DQ_ORPHAN_ASSETS = 2            # fully isolated fixture assets: no parent, no connectivity edges
DQ_INVALID_PARENT = 2            # assets whose parent_asset_id references a nonexistent asset_id
DQ_DUPLICATE_EDGES = 1            # duplicated connectivity row
DQ_MISSING_RELATIONSHIPS = 2      # metadata implies a parent link but no matching connectivity.csv row

# Energy meter reconciliation: main meter = sum(sub-meters) * (1 + loss)
UNMETERED_LOSS_FRAC = (0.05, 0.10)
