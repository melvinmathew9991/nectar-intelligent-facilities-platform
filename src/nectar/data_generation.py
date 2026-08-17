"""
Orchestrates synthetic data generation for the Nectar Intelligent Facilities
Platform: asset metadata + hierarchy, connectivity edges (with intentional
data-quality issues), and physics-grounded telemetry (with fault archetypes
and standalone anomalies). See PLAN.md sections 1.1-1.8.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from . import config, physics
from . import weather as weather_mod
from .logging_config import get_logger

log = get_logger(__name__)

TYPE_CODE = {"Chiller": "CHL", "AHU": "AHU", "Pump": "PMP",
             "EnergyMeter": "MTR", "EnvSensor": "EVS"}
STEPS_PER_HOUR = 6   # 10-min resolution


# ---------------------------------------------------------------------------
# 1. Metadata + hierarchy + connectivity
# ---------------------------------------------------------------------------
def _make_asset_row(asset_id, name, site_id, building_id, asset_type, parent_id, rng):
    age_years = rng.uniform(*config.INSTALL_AGE_YEARS_RANGE)
    install_date = pd.Timestamp(config.DATE_START) - pd.Timedelta(days=age_years * 365.25)
    cap_range = config.CAPACITY_RANGES[asset_type]
    capacity = round(rng.uniform(*cap_range), 1) if cap_range else np.nan
    return {
        "asset_id": asset_id, "site_id": site_id, "building_id": building_id,
        "asset_name": name, "asset_type": asset_type,
        "manufacturer": rng.choice(config.MANUFACTURERS[asset_type]),
        "installation_date": install_date.normalize(),
        "capacity": capacity, "parent_asset_id": parent_id,
    }


def build_metadata(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets, edges = [], []

    for site_id in config.SITES:
        for b in range(1, config.BUILDINGS_PER_SITE + 1):
            building_id = f"{site_id}-B{b}"
            ahu_seq = env_seq = 0
            chiller_ids, pump_ids = [], []

            for c in range(1, config.CHILLERS_PER_BUILDING + 1):
                chiller_id = f"{building_id}-CHL-{c:02d}"
                chiller_ids.append(chiller_id)
                assets.append(_make_asset_row(chiller_id, f"Chiller-{c:02d}", site_id,
                                               building_id, "Chiller", np.nan, rng))
                n_ahus = rng.integers(config.AHUS_PER_CHILLER[0], config.AHUS_PER_CHILLER[1] + 1)
                for _ in range(n_ahus):
                    ahu_seq += 1
                    ahu_id = f"{building_id}-AHU-{ahu_seq:02d}"
                    assets.append(_make_asset_row(ahu_id, f"AHU-{ahu_seq:02d}", site_id,
                                                   building_id, "AHU", chiller_id, rng))
                    edges.append((chiller_id, ahu_id, "Supplies"))

                    env_seq += 1
                    env_id = f"{building_id}-EVS-{env_seq:02d}"
                    assets.append(_make_asset_row(env_id, f"EnvSensor-{env_seq:02d}", site_id,
                                                   building_id, "EnvSensor", ahu_id, rng))
                    edges.append((ahu_id, env_id, "Controls"))

            for pmp in range(1, config.PUMPS_PER_BUILDING + 1):
                pump_id = f"{building_id}-PMP-{pmp:02d}"
                pump_ids.append(pump_id)
                assets.append(_make_asset_row(pump_id, f"Pump-{pmp:02d}", site_id,
                                               building_id, "Pump", np.nan, rng))
                meter_id = f"{building_id}-MTR-SUB-{pmp:02d}"
                assets.append(_make_asset_row(meter_id, f"EnergyMeter-SUB-{pmp:02d}", site_id,
                                               building_id, "EnergyMeter", pump_id, rng))
                edges.append((pump_id, meter_id, "Monitors"))

            main_meter_id = f"{building_id}-MTR-MAIN-01"
            assets.append(_make_asset_row(main_meter_id, "EnergyMeter-MAIN-01", site_id,
                                           building_id, "EnergyMeter", np.nan, rng))
            # Main meter aggregates every major consuming asset in the building --
            # give it explicit "Monitors" edges so it's a legitimate root, not a
            # false-positive orphan in Task 5's data-quality audit.
            for eid in chiller_ids + pump_ids:
                edges.append((main_meter_id, eid, "Monitors"))

    metadata = pd.DataFrame(assets)
    connectivity = pd.DataFrame(edges, columns=["source_asset_id", "target_asset_id", "connection_type"])
    connectivity["relationship_strength"] = rng.uniform(0.4, 1.0, len(connectivity)).round(2)

    log.info(f"Built metadata: {len(metadata)} assets across "
             f"{metadata.building_id.nunique()} buildings; {len(connectivity)} connectivity edges")
    return metadata, connectivity


# ---------------------------------------------------------------------------
# 2. Intentional data-quality issues (see PLAN.md 1.8)
# ---------------------------------------------------------------------------
def inject_dq_issues(metadata: pd.DataFrame, connectivity: pd.DataFrame,
                       rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata = metadata.copy()
    connectivity = connectivity.copy()
    issues = {}

    # (a) Fully isolated orphan fixture assets: no parent, no connectivity edges at all.
    orphan_rows = []
    for i in range(config.DQ_ORPHAN_ASSETS):
        site_id = rng.choice(config.SITE_IDS)
        building_id = f"{site_id}-B{rng.integers(1, config.BUILDINGS_PER_SITE + 1)}"
        asset_type = rng.choice(["EnvSensor", "EnergyMeter"])
        asset_id = f"{site_id}-ISO-{i+1:03d}"
        orphan_rows.append(_make_asset_row(asset_id, f"Legacy-{asset_type}-{i+1:02d}",
                                            site_id, building_id, asset_type, np.nan, rng))
    metadata = pd.concat([metadata, pd.DataFrame(orphan_rows)], ignore_index=True)
    issues["orphan_assets"] = [r["asset_id"] for r in orphan_rows]

    # (b) Assets whose parent_asset_id references a nonexistent asset.
    candidates = metadata[metadata["parent_asset_id"].isna()
                            & ~metadata["asset_id"].isin(issues["orphan_assets"])]
    invalid_targets = rng.choice(candidates["asset_id"].values,
                                   size=config.DQ_INVALID_PARENT, replace=False)
    ghost_parents = []
    for i, aid in enumerate(invalid_targets):
        ghost_id = f"{aid.rsplit('-', 2)[0]}-GHOST-{i+1:03d}"
        metadata.loc[metadata["asset_id"] == aid, "parent_asset_id"] = ghost_id
        ghost_parents.append((aid, ghost_id))
    issues["invalid_parent"] = ghost_parents

    # (c) Duplicate connectivity row(s).
    dup_rows = connectivity.sample(n=config.DQ_DUPLICATE_EDGES, random_state=int(rng.integers(1e9)))
    connectivity = pd.concat([connectivity, dup_rows], ignore_index=True)
    issues["duplicate_edges"] = dup_rows[["source_asset_id", "target_asset_id"]].values.tolist()

    # (d) Missing relationships: metadata keeps the parent link, but the
    # corresponding connectivity.csv row is dropped -- a gap between sources.
    # Restricted to edges that actually mirror a metadata parent_asset_id
    # link (excludes e.g. main-meter "Monitors" edges, which have no such
    # metadata counterpart and so can't produce this specific DQ signature).
    parent_lookup = metadata.set_index("asset_id")["parent_asset_id"]
    is_hierarchy_edge = connectivity.apply(
        lambda r: parent_lookup.get(r["target_asset_id"]) == r["source_asset_id"], axis=1)
    droppable_idx = connectivity.index[is_hierarchy_edge].difference(dup_rows.index)
    drop_idx = rng.choice(droppable_idx, size=config.DQ_MISSING_RELATIONSHIPS, replace=False)
    issues["missing_relationships"] = connectivity.loc[
        drop_idx, ["source_asset_id", "target_asset_id"]].values.tolist()
    connectivity = connectivity.drop(index=drop_idx).reset_index(drop=True)

    log.info(f"Injected DQ issues: {issues}")
    return metadata, connectivity, issues


# ---------------------------------------------------------------------------
# 3. Telemetry generation
# ---------------------------------------------------------------------------
def generate_telemetry(metadata: pd.DataFrame, weather_df: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    building_ids = sorted(metadata["building_id"].unique())
    site_of_building = (metadata.drop_duplicates("building_id")
                         .set_index("building_id")["site_id"].to_dict())
    ctx_all = physics.build_building_context(building_ids, site_of_building, weather_df, rng)
    idx = pd.date_range(config.DATE_START, config.DATE_END, freq=config.FREQ)
    n = len(idx)

    active = metadata[metadata["asset_type"].isin(config.ASSET_TYPES)].copy()
    non_meter = active[active["asset_type"] != "EnergyMeter"]
    parent_of = metadata.set_index("asset_id")["parent_asset_id"].to_dict()
    children_of: dict[str, list[str]] = {}
    for aid, pid in parent_of.items():
        if isinstance(pid, str):
            children_of.setdefault(pid, []).append(aid)

    series_by_asset: dict[str, dict] = {}
    fault_episodes = []   # (asset_id, asset_type, archetype, start_idx, fault_start, fault_end)

    # ---- baseline + primary fault injection (rotating equipment) ----
    for _, row in non_meter.iterrows():
        ctx = ctx_all[ctx_all.building_id == row.building_id].sort_values("timestamp").reset_index(drop=True)
        age_years = (pd.Timestamp(config.DATE_START) - row.installation_date).days / 365.25
        series = physics.generate_asset_baseline(row.asset_type, ctx, age_years, rng)

        if row.asset_type in config.ROTATING_TYPES:
            # Elevated fault probability with age (real, non-decorative wear-out
            # effect, not just a baseline signal offset) -- mean episode count
            # rises from ~1.3 (new) to ~2.3 (oldest, ~8y) across INSTALL_AGE_YEARS_RANGE.
            age_factor = np.clip(age_years / config.INSTALL_AGE_YEARS_RANGE[1], 0.0, 1.0)
            n_faults = 1 + rng.binomial(2, 0.15 + 0.5 * age_factor)
            min_gap = 200 * STEPS_PER_HOUR
            used_starts = []
            for _ in range(n_faults):
                archetype = rng.choice(config.FAULT_ARCHETYPES[row.asset_type])
                ramp_h = int(rng.integers(*config.FAULT_RAMP_HOURS))
                dur_h = int(rng.integers(config.FAULT_DURATION_HOURS[0], config.FAULT_DURATION_HOURS[1] + 1))
                span = (ramp_h + dur_h) * STEPS_PER_HOUR
                for _attempt in range(10):
                    start = int(rng.integers(0, n - span - 1))
                    if all(abs(start - u) > min_gap for u in used_starts):
                        break
                used_starts.append(start)
                f_start, f_end = physics.apply_fault_episode(
                    series, start, archetype, ramp_h, dur_h, STEPS_PER_HOUR, scale=1.0)
                fault_episodes.append((row.asset_id, row.asset_type, archetype, start, f_start, f_end))

        series_by_asset[row.asset_id] = series

    # ---- secondary (downstream) fault effect on direct children ----
    for asset_id, asset_type, archetype, start, f_start, f_end in fault_episodes:
        if asset_type not in ("Chiller", "Pump"):
            continue
        for child_id in children_of.get(asset_id, []):
            if child_id not in series_by_asset:
                continue
            ramp_h = (f_start - start) // STEPS_PER_HOUR
            dur_h = max(1, (f_end - f_start) // STEPS_PER_HOUR)
            physics.apply_fault_episode(series_by_asset[child_id], start, archetype,
                                          max(ramp_h, 1), dur_h, STEPS_PER_HOUR,
                                          scale=config.DOWNSTREAM_FAULT_EFFECT_SCALE)

    # ---- standalone anomalies (independent of faults) ----
    for series in series_by_asset.values():
        series["power_consumption"], _ = physics.inject_power_spikes(series["power_consumption"], rng)
        if rng.random() < config.ANOMALY_DRIFT_ASSET_FRACTION:
            series["temperature"], _ = physics.inject_slow_drift(series["temperature"], rng)
        if rng.random() < config.ANOMALY_STUCK_ASSET_FRACTION:
            series["vibration"], _ = physics.inject_stuck_sensor(
                series["vibration"], rng, config.ANOMALY_STUCK_DURATION_HOURS, STEPS_PER_HOUR)

    # ---- assemble non-meter rows ----
    frames = []
    for _, row in non_meter.iterrows():
        s = series_by_asset[row.asset_id]
        ctx = ctx_all[ctx_all.building_id == row.building_id].sort_values("timestamp").reset_index(drop=True)
        occ_jitter = rng.uniform(0.97, 1.03, n)
        fault_flag = np.zeros(n, dtype=int)
        for aid, _, _, _, f_start, f_end in fault_episodes:
            if aid == row.asset_id:
                fault_flag[f_start:f_end] = 1
        frames.append(pd.DataFrame({
            "timestamp": idx, "site_id": row.site_id, "building_id": row.building_id,
            "asset_id": row.asset_id,
            "temperature": s["temperature"], "humidity": s["humidity"],
            "pressure": s["pressure"], "vibration": np.clip(s["vibration"], 0, None),
            "power_consumption": np.clip(s["power_consumption"], 0, None),
            "occupancy_count": np.round(ctx["occupancy_count"].values * occ_jitter).clip(min=0),
            "operating_mode": ctx["operating_mode"].values,
            "fault_flag": fault_flag,
        }))
    telemetry = pd.concat(frames, ignore_index=True)

    # ---- energy meters: submeters mirror their monitored pump; main meter
    # aggregates all real consuming equipment in the building + unmetered loss ----
    meter_rows = metadata[metadata["asset_type"] == "EnergyMeter"]
    meter_frames = []
    for _, row in meter_rows.iterrows():
        ctx = ctx_all[ctx_all.building_id == row.building_id].sort_values("timestamp").reset_index(drop=True)
        if isinstance(row.parent_asset_id, str) and row.parent_asset_id in series_by_asset:
            monitored_power = series_by_asset[row.parent_asset_id]["power_consumption"]
            power = np.clip(monitored_power, 0, None) * rng.uniform(0.98, 1.02, n)
        else:
            bldg_power = telemetry.loc[telemetry.building_id == row.building_id]
            agg = (bldg_power.groupby("timestamp")["power_consumption"].sum()
                   .reindex(idx, fill_value=0.0).values)
            loss = rng.uniform(*config.UNMETERED_LOSS_FRAC)
            power = agg * (1 + loss)
        p = config.SIGNAL_PARAMS["EnergyMeter"]
        meter_frames.append(pd.DataFrame({
            "timestamp": idx, "site_id": row.site_id, "building_id": row.building_id,
            "asset_id": row.asset_id,
            "temperature": p["temperature"]["baseline"] + rng.normal(0, p["temperature"]["noise_sd"], n),
            "humidity": p["humidity"]["baseline"] + rng.normal(0, p["humidity"]["noise_sd"], n),
            "pressure": p["pressure"]["baseline"] + rng.normal(0, p["pressure"]["noise_sd"], n),
            "vibration": np.abs(rng.normal(p["vibration"]["baseline"], p["vibration"]["noise_sd"], n)),
            "power_consumption": np.clip(power, 0, None),
            "occupancy_count": np.round(ctx["occupancy_count"].values * rng.uniform(0.97, 1.03, n)).clip(min=0),
            "operating_mode": ctx["operating_mode"].values,
            "fault_flag": 0,
        }))
    telemetry = pd.concat([telemetry] + meter_frames, ignore_index=True)

    # ---- missingness: per-cell random gaps + a few longer outage windows ----
    sensor_cols = ["temperature", "humidity", "pressure", "vibration"]
    miss_frac = rng.uniform(*config.TELEMETRY_MISSING_FRAC)
    mask = rng.random((len(telemetry), len(sensor_cols))) < miss_frac
    for j, col in enumerate(sensor_cols):
        telemetry.loc[mask[:, j], col] = np.nan

    for site_id in config.SITE_IDS:
        site_assets = metadata.loc[metadata.site_id == site_id, "asset_id"].values
        n_outages = rng.integers(*config.TELEMETRY_OUTAGE_WINDOWS_PER_SITE)
        for _ in range(n_outages):
            aid = rng.choice(site_assets)
            dur = int(rng.integers(*config.TELEMETRY_OUTAGE_DURATION_HOURS) * STEPS_PER_HOUR)
            asset_mask = telemetry.asset_id == aid
            asset_positions = np.flatnonzero(asset_mask.values)
            if len(asset_positions) <= dur:
                continue
            start = rng.integers(0, len(asset_positions) - dur)
            outage_positions = asset_positions[start:start + dur]
            telemetry.loc[telemetry.index[outage_positions], sensor_cols] = np.nan

    telemetry[sensor_cols] = telemetry[sensor_cols].round(3)
    telemetry["power_consumption"] = telemetry["power_consumption"].round(3)
    telemetry = telemetry.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)

    log.info(f"Generated telemetry: {len(telemetry):,} rows, "
             f"{telemetry.asset_id.nunique()} assets, "
             f"fault_flag rate {telemetry.fault_flag.mean()*100:.3f}%, "
             f"{len(fault_episodes)} fault episodes")
    return telemetry


# ---------------------------------------------------------------------------
# 4. Orchestration
# ---------------------------------------------------------------------------
def generate_all(save: bool = True) -> dict:
    rng = np.random.default_rng(config.SEED)

    weather_df = weather_mod.generate_weather()
    metadata, connectivity = build_metadata(rng)
    metadata, connectivity, dq_issues = inject_dq_issues(metadata, connectivity, rng)
    telemetry = generate_telemetry(metadata, weather_df, rng)

    if save:
        os.makedirs(config.DATA_RAW_DIR, exist_ok=True)
        weather_mod.save_weather(weather_df)
        metadata.to_csv(os.path.join(config.DATA_RAW_DIR, "asset_metadata.csv"), index=False)
        connectivity.to_csv(os.path.join(config.DATA_RAW_DIR, "asset_connectivity.csv"), index=False)
        telemetry.to_csv(os.path.join(config.DATA_RAW_DIR, "sensor_telemetry.csv"), index=False)
        log.info(f"Saved 4 raw datasets -> {config.DATA_RAW_DIR}")

    return {"weather": weather_df, "metadata": metadata, "connectivity": connectivity,
            "telemetry": telemetry, "dq_issues": dq_issues}


if __name__ == "__main__":
    generate_all()
