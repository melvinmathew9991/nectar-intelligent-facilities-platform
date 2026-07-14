# Data Dictionary — Per-Asset-Type Physical Grounding

This document formalizes the units, ranges, and real-world standards `src/nectar/physics.py`
and `src/nectar/config.py::SIGNAL_PARAMS` were built against — the artifact that backs up
the "physically plausible" claim made throughout this project, not just an assertion in
the README.

## Sensor Telemetry Schema

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | 10-minute resolution, 2025-01-01 → 2025-03-31 (90 days) |
| `site_id` | str | `CBE` (Coimbatore) / `CHN` (Chennai) / `BLR` (Bangalore) |
| `building_id` | str | `{site_id}-B{1..3}` |
| `asset_id` | str | `{building_id}-{TYPE}-{seq}`, e.g. `CBE-B1-CHL-01` |
| `temperature` | float, °C | meaning is asset-type-specific — see table below |
| `humidity` | float, % RH | asset-type-specific |
| `pressure` | float | unit is asset-type-specific (bar / Pa / kPa) — see table |
| `vibration` | float, mm/s RMS | ISO 10816-consistent severity bands for rotating equipment |
| `power_consumption` | float, kWh | |
| `occupancy_count` | float | building-level shared curve, small per-asset jitter |
| `operating_mode` | str | `Cooling` / `Heating` / `Idle` |
| `fault_flag` | int (0/1) | terminal event of a fault episode (rotating equipment only) |

## Per-Asset-Type Signal Profiles

| Asset Type | Temperature | Pressure | Vibration | Power Model |
|---|---|---|---|---|
| **Chiller** | Chilled-water supply, baseline ≈7°C — typical commercial chilled-water loop supply temp | Refrigerant, baseline ≈6.5 bar, range 4–12 bar — typical R-134a/R-410A operating range | Baseline ≈2.2 mm/s RMS, escalating during faults — ISO 10816-3 zone A/B (good) → C/D (unacceptable) for large rotating machines | Non-linear part-load ratio curve: `0.15 + 0.85·load^1.3` × 180 kW baseline; ~8% of that when Idle |
| **AHU** | Supply air, baseline ≈14°C — typical ASHRAE supply-air setpoint band | Static, baseline ≈300 Pa, range 100–500 Pa — typical ducted AHU static pressure range | Baseline ≈1.4 mm/s RMS — fan/bearing, ISO 10816 general-machinery bands | Fan affinity law: `P ∝ speed³`, i.e. `0.05 + 0.95·load³` × 18 kW baseline |
| **Pump** | Loop fluid temp, baseline ≈12°C, rises with load | Discharge, baseline ≈250 kPa, range 80–400 kPa — typical circulation-pump discharge range | Baseline ≈1.8 mm/s RMS — bearing/cavitation signature | Pump affinity law: `P ∝ flow^~2.2`, i.e. `0.1 + 0.9·load^2.2` × 12 kW baseline |
| **EnergyMeter** | Panel ambient, ≈28°C, near-flat | Not physically meaningful — near-constant placeholder | Near-zero baseline (no moving parts) | Submeter: mirrors its monitored Pump's power (±2% metering noise). Main meter: `sum(Chiller+AHU+Pump power in building) × (1 + unmetered_loss)`, loss ∈ [5%, 10%] |
| **EnvSensor** | Zone temp, baseline ≈23.5°C — ASHRAE 55 comfort band (22–26°C) | Ambient, near-flat | Near-zero (no moving parts) | Negligible (sensor's own parasitic draw only) |

## Fault Archetypes (rotating equipment only)

Each fault episode is a **6–48h monotonic ramp** on the affected channel(s), then holds at
peak severity for the flagged (`fault_flag=1`) window (1–3h) — the ramp rows themselves
carry `fault_flag=0`, which is the entire point of predictive maintenance (learning to
predict the flag from the *pre-flag* degradation signature).

| Archetype | Asset Type | Channels Affected | Direction |
|---|---|---|---|
| `refrigerant_leak` | Chiller | pressure ↓, temperature ↑, power ↓ | falling suction pressure + rising approach temp + falling efficiency |
| `condenser_fouling` | Chiller | pressure ↑, power ↑ | rising condenser pressure and power draw |
| `filter_clogging` | AHU | pressure ↑ | rising static pressure, slow drift |
| `fan_bearing_wear` | AHU | vibration ↑ | vibration ramp |
| `cavitation` | Pump | vibration ↑, pressure ↓ | vibration + pressure instability |
| `bearing_wear` | Pump | vibration ↑ | gradual vibration ramp |

A Chiller/Pump fault also applies a smaller (25% scale) secondary effect to its **direct
downstream children** via the connectivity graph over the same window (e.g. an AHU's
supply temperature drifts up during its parent Chiller's fault) — this is what gives
Task 5's failure-propagation analysis real telemetry evidence, not just topology.

## Standalone Anomalies (Task 4 target, independent of fault archetypes)

| Type | Mechanism | Target |
|---|---|---|
| Power spike | 2–5 random hours per asset, 2.5–4× normal power | all active asset types |
| Slow drift | additive linear drift over a ~200h window | ~20% of assets, temperature channel |
| Stuck sensor | flatlined value over a 3–12h window | ~5% of assets, vibration channel |

## Data-Quality Fixtures (intentional, by design)

| Issue | Count | Purpose |
|---|---|---|
| Orphan assets (no parent, no connectivity edges) | 2 | tests unreachable-node detection |
| Invalid parent (`parent_asset_id` references a nonexistent asset) | 2 | tests referential-integrity checks |
| Duplicate connectivity row | 1 | tests dedup logic |
| Missing relationship (metadata implies a link, `asset_connectivity.csv` omits it) | 2 | tests cross-source consistency checks |

All counts are asserted in `tests/test_data_generation.py` and `tests/test_graph.py`, not
just documented — see `VERIFICATION` in the README for how to reproduce this table.
