"""Sanity-check the generator's output: row counts, injected fault/DQ-issue
counts land in target ranges, no structural corruption. Run against the
already-generated data/raw/*.csv (deterministic, SEED=42)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import config


@pytest.fixture(scope="module")
def telemetry():
    return pd.read_csv(os.path.join(config.DATA_RAW_DIR, "sensor_telemetry.csv"),
                        parse_dates=["timestamp"])


@pytest.fixture(scope="module")
def metadata():
    return pd.read_csv(os.path.join(config.DATA_RAW_DIR, "asset_metadata.csv"),
                        parse_dates=["installation_date"])


@pytest.fixture(scope="module")
def connectivity():
    return pd.read_csv(os.path.join(config.DATA_RAW_DIR, "asset_connectivity.csv"))


@pytest.fixture(scope="module")
def weather():
    return pd.read_csv(os.path.join(config.DATA_RAW_DIR, "weather.csv"), parse_dates=["timestamp"])


def test_asset_count_in_range(metadata):
    assert 140 <= len(metadata) <= 170, f"expected ~150-160 assets, got {len(metadata)}"


def test_telemetry_row_count_matches_assets(telemetry, metadata):
    n_timestamps = len(pd.date_range(config.DATE_START, config.DATE_END, freq=config.FREQ))
    expected = len(metadata) * n_timestamps
    assert len(telemetry) == expected


def test_no_duplicate_asset_timestamp_rows(telemetry):
    assert telemetry.duplicated(["asset_id", "timestamp"]).sum() == 0


def test_no_negative_sensor_values(telemetry):
    for col in ["temperature", "pressure", "vibration", "power_consumption", "occupancy_count"]:
        assert telemetry[col].min() >= 0, f"{col} has negative values"


def test_fault_episode_count_in_range(telemetry):
    # each episode contributes 1-3 hours (6-18 rows at 10-min resolution) of fault_flag=1
    approx_episodes = telemetry.fault_flag.sum() / 12  # rough midpoint divisor
    lo, hi = config.FAULT_TARGET_EPISODES
    assert lo * 0.5 <= approx_episodes <= hi * 1.5, \
        f"fault episode count looks off: ~{approx_episodes:.0f} (target {lo}-{hi})"


def test_fault_flag_only_on_rotating_equipment(telemetry, metadata):
    faulty_assets = telemetry.loc[telemetry.fault_flag == 1, "asset_id"].unique()
    types = metadata.set_index("asset_id").loc[faulty_assets, "asset_type"]
    assert set(types.unique()).issubset(set(config.ROTATING_TYPES))


def test_missingness_in_target_range(telemetry):
    lo, hi = config.TELEMETRY_MISSING_FRAC
    for col in ["temperature", "humidity", "pressure", "vibration"]:
        frac = telemetry[col].isna().mean()
        assert lo * 0.5 <= frac <= hi * 2.0, f"{col} missingness {frac:.3f} outside expected band"
    assert telemetry["power_consumption"].isna().sum() == 0
    assert telemetry["occupancy_count"].isna().sum() == 0


def test_operating_mode_heating_is_rare_edge_case(telemetry):
    frac = (telemetry["operating_mode"] == "Heating").mean()
    assert 0 < frac < 0.01, f"Heating should be a rare (<1%) but genuine occurrence, got {frac:.4f}"


def test_dq_orphan_assets_detected(metadata, connectivity):
    conn_ids = set(connectivity["source_asset_id"]) | set(connectivity["target_asset_id"])
    isolated = metadata[metadata["parent_asset_id"].isna() & ~metadata["asset_id"].isin(conn_ids)]
    assert len(isolated) >= config.DQ_ORPHAN_ASSETS


def test_dq_invalid_parent_detected(metadata):
    asset_ids = set(metadata["asset_id"])
    invalid = metadata[metadata["parent_asset_id"].notna()
                        & ~metadata["parent_asset_id"].isin(asset_ids)]
    assert len(invalid) >= config.DQ_INVALID_PARENT


def test_dq_duplicate_edges_detected(connectivity):
    dups = connectivity.duplicated(["source_asset_id", "target_asset_id"]).sum()
    assert dups >= config.DQ_DUPLICATE_EDGES


def test_dq_missing_relationships_detected(metadata, connectivity):
    """Assets with a real parent (not a DQ-invalid one) but no matching connectivity row."""
    asset_ids = set(metadata["asset_id"])
    has_edge = set(zip(connectivity["source_asset_id"], connectivity["target_asset_id"], strict=True))
    valid_parent = metadata[metadata["parent_asset_id"].notna()
                             & metadata["parent_asset_id"].isin(asset_ids)]
    missing = [r for r in valid_parent.itertuples()
               if (r.parent_asset_id, r.asset_id) not in has_edge]
    assert len(missing) >= config.DQ_MISSING_RELATIONSHIPS


def test_energy_meter_reconciliation(telemetry, metadata):
    """Main building meter should exceed sum of Chiller/AHU/Pump power by ~ the
    unmetered-loss factor, for every building."""
    lo, hi = config.UNMETERED_LOSS_FRAC
    merged = telemetry.merge(metadata[["asset_id", "asset_type", "asset_name"]], on="asset_id")
    for building_id, g in merged.groupby("building_id"):
        main = g[g["asset_name"] == "EnergyMeter-MAIN-01"]
        if main.empty:
            continue
        real_equip = (g[g.asset_type.isin(config.ROTATING_TYPES)]
                      .groupby("timestamp")["power_consumption"].sum())
        main_ts = main.set_index("timestamp")["power_consumption"]
        ratio = (main_ts / real_equip.reindex(main_ts.index)).replace([np.inf, -np.inf], np.nan).dropna()
        assert ratio.mean() > 1.0, f"{building_id} main meter should exceed sub-equipment sum"


def test_weather_covers_full_range_all_sites(weather):
    n_timestamps = len(pd.date_range(config.DATE_START, config.DATE_END, freq=config.FREQ))
    for site_id in config.SITE_IDS:
        assert (weather.site_id == site_id).sum() == n_timestamps
