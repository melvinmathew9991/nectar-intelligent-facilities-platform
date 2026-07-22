# Nectar Data Scientist Challenge — Build Plan (v2, finalized)

## Context

Take-home assessment, greenfield `D:\Nectar`. No real data exists, so the foundation of this
whole submission is a synthetic-but-physically-grounded IoT dataset — everything downstream (EDA findings, model learnability, anomaly signatures, connectivity failure-impact narrative) is only as credible as the generator. This version (v2) replaces v1's generic noise-based generator with a domain-grounded one (real HVAC/BMS physics, ASHRAE/ISO-standard ranges, causal coupling across the asset graph) after an explicit review pass against the brief and IoT domain practice. **This is the version we build against — no further plan revisions expected.**

Confirmed scope:
- Build everything: 5 core tasks + a Streamlit dashboard.
- Data scale: ~2M telemetry rows, 90 days, 10-min resolution, ~150-160 assets across 9 buildings/3 sites.
- 4th synthetic dataset `weather.csv` (site_id, timestamp, outdoor_temp, outdoor_humidity) added beyond the brief's 3 named datasets — documented as an explicit assumption (real BMS deployments commonly ingest a weather feed as an exogenous input for load/forecasting).
- Effort allocated in proportion to rubric weights: deepest on Predictive Modeling (20%) and
  Connectivity Analysis (15%); EDA (15%) thorough; Forecasting/Anomaly (10% each) solid but leaner.

## Repo Layout

```
D:\Nectar\
  data/
    raw/                     telemetry.csv, asset_metadata.csv, asset_connectivity.csv, weather.csv
    processed/                cleaned/feature-engineered parquet outputs
  src/nectar/
    config.py                 sites/buildings/assets, date range, seeds, per-asset-type physical parameter tables
    weather.py                 per-site synthetic outdoor temp/humidity generator (diurnal+seasonal)
    data_generation.py         orchestrates metadata, connectivity, and physics-grounded telemetry generation
    physics.py                 per-asset-type signal models (chiller/AHU/pump/meter/env-sensor), fault archetypes, graph-coupled load propagation
    preprocessing.py           missing-value handling, cleaning, resampling utilities
    features.py                 shared feature-engineering fns (rolling stats, lags) used by training AND the dashboard
    maintenance_model.py       train/eval predictive maintenance model, SHAP, artifact saving
    forecasting.py              energy forecasting models (statsmodels baseline + XGBoost w/ weather regressor)
    anomaly.py                   multi-method anomaly detection framework
    graph.py                     NetworkX asset graph, query functions, failure-impact simulation, data-quality checks
    logging_config.py            standard logging setup (no print statements in src/)
  notebooks/
    01_data_generation.ipynb
    02_eda.ipynb
    03_predictive_maintenance.ipynb
    04_energy_forecasting.ipynb
    05_anomaly_detection.ipynb
    06_connectivity_analysis.ipynb
  scripts/
    run_pipeline.py             one-command reproduction: generate -> preprocess -> train all -> detect -> graph -> artifacts
  tests/
    test_data_generation.py     row counts, injected fault/issue counts land in target ranges
    test_features.py             time-split has no leakage, rolling features shift correctly
    test_graph.py                 query functions correct on a small fixture graph
  models/                       saved model artifacts (.pkl/.json) + feature list
  docs/
    data_dictionary.md           per-asset-type units/ranges/standards used by the generator (the methodology proof)
  reports/figures/               exported chart PNGs referenced by the report
  reports/Nectar_DS_Challenge_Report.md
  dashboard/app.py
  requirements.txt
  README.md
```

## 1. Synthetic Data Design — Domain-Grounded Generator

### 1.1 Scale & topology
3 sites (Coimbatore / Chennai / Bangalore, distinct climate profiles) × 3 buildings/site = 9
buildings. ~15-18 assets/building (Chillers, AHUs, Pumps, Energy Meters, Environmental sensors) → ~150-160 assets. Telemetry at 10-min resolution, 90 days ≈ 2M rows.

**Naming**: `asset_id` structured code (e.g. `CBE-B1-CHL-01`), `asset_name` human-readable and scoped per building matching the brief's own examples (`Chiller-01`, `AHU-03`, `Pump-02`) so the Task 5 demo queries read naturally.

### 1.2 Outdoor weather driver (`weather.py`, → `weather.csv`)
Per site, per-timestamp outdoor_temp/humidity = city-specific seasonal mean + diurnal sinusoid
(peak mid-afternoon) + AR(1)-style day-to-day noise. City profiles: Chennai (hot/humid, ~26-38°C),
Bangalore (mild, ~18-30°C), Coimbatore (moderate, ~22-33°C). This is the root driver for cooling
load, chiller runtime, operating_mode, and power_consumption — gives EDA and forecasting a real,
explainable causal chain instead of decorative noise.

### 1.3 Per-asset-type physical signal models (`physics.py`)
Same 4 schema columns (temperature, humidity, pressure, vibration) mean different physical
quantities per asset_type, generated with realistic units/ranges (formalized in
`docs/data_dictionary.md`):

| asset_type | temperature | pressure | vibration | power_consumption |
|---|---|---|---|---|
| Chiller | chilled water supply, ~6-8°C | refrigerant, 4-12 bar | compressor; ISO 10816-3 severity bands | part-load-ratio curve vs. site cooling-load index (non-linear) |
| AHU | supply air, ~12-16°C | static pressure, 100-500 Pa | fan/bearing; ISO 10816 bands | ∝ fan-speed³ (affinity law) |
| Pump | loop fluid temp | discharge, kPa | bearing/cavitation signature | ∝ flow via affinity law |
| Energy Meter | ambient panel temp, near-flat | not physically meaningful, near-constant | near-zero baseline | aggregation target (see 1.5) |
| Env. Sensor | zone temp, 22-26°C comfort band | ambient, near-flat | n/a (near-zero) | n/a (near-zero/omitted signal) |

humidity: dynamic mainly for AHU (supply/return air, 40-60% RH) and Env. Sensor (zone RH);
near-ambient/flat elsewhere.

### 1.4 Causal coupling across the connectivity graph
Rather than a slow per-timestep control-loop simulator, use a **vectorized shared-driver
approach**: compute one site/building "cooling load index" time series from outdoor_temp +
occupancy, then derive each asset's signal as a function of that shared index (via its position in the `parent_asset_id`/connectivity graph) plus asset-specific noise. This makes a chiller and its child AHUs move together physically, without an iterative simulator — keeps ~2M-row generation
fast (seconds, not minutes).

**Occupancy**: weekday office curve (9am-6pm peak, near-zero nights/weekends), building-level, shared across all assets in that building at a given timestamp (consistent with a BMS tagging readings with building occupancy state). Drives cooling load and, downstream, `operating_mode` (Cooling when occupied + OAT above deadband, Idle otherwise; Heating held to <1% of readings, documented as a South-Indian-climate edge case rather than a real seasonal mode).

**Asset age effect**: `installation_date` gives each asset a small baseline efficiency
penalty/vibration offset and elevated fault probability with age — real, non-decorative EDA
finding ("older equipment fails more") and a legitimate model feature.

### 1.5 Energy meter hierarchy
Building main meter ≈ sum(sub-meters/assets) + 5-10% unmetered loss — gives Task 3 forecasting a coherent aggregation target and a natural reconciliation data-quality check to mention.

### 1.6 Fault archetypes (asset-type specific, `physics.py`)
~150-250 fault episodes total (~1-2% of asset-days, realistic BMS fault prevalence), each drawn from an archetype matched to its asset_type, with a 6-48h **leading-indicator degradation window** before a short `fault_flag=1` event (the actual logged fault), then reset. `fault_flag` marks only the terminal event — the degradation window itself carries no flag, which is the entire point of predictive maintenance (predicting the flag before it fires from leading signatures):
- Chiller: refrigerant leak (falling suction pressure + rising approach temp + falling efficiency); condenser fouling (rising condenser pressure/power)
- AHU: filter clogging (rising static pressure, slow drift); fan bearing wear (vibration ramp)
- Pump: cavitation (vibration + pressure spikes); bearing wear (gradual vibration ramp)
- Energy meter: comms dropout / stuck-value rather than mechanical degradation
Fault episodes on a Chiller/Pump also apply a smaller secondary effect to directly-downstream
assets via the connectivity graph (e.g. AHU supply temp drifts up during a parent chiller fault) — this is what gives Task 5's failure-impact analysis real telemetry evidence, not just a metadata
graph.

### 1.7 Anomalies (separate from faults, for Task 4)
Power spikes, stuck-sensor dropout (flatlined value), slow sensor drift that doesn't necessarily lead to a fault — distinct injection from the fault archetypes above so Task 4's multi-method framework (z-score/IQR for spikes, CUSUM/rolling-slope for drift, change-point for degradation onset, Isolation Forest for combined multivariate states) has a real reason to be multi-method.

### 1.8 Missingness & data-quality issues
Telemetry: ~2-5% random gaps + a few longer per-asset outage windows.
Connectivity/metadata: a duplicate edge, 2-3 orphan assets (no parent, no connectivity edge), one invalid `parent_asset_id` (references non-existent or cross-site asset) — injected deliberately so
Task 5's data-quality assessment has real findings to report, and `tests/test_data_generation.py` can assert the detector actually finds them.

All counts/ranges/seeds centralized in `config.py`; `np.random.default_rng(seed)` used
consistently for reproducibility.

## 2. Task 1 — EDA (`notebooks/02_eda.ipynb`)
Distributions per sensor & asset_type, missing-value analysis, temporal patterns (hourly/weekly heatmaps), cross-site/cross-asset-type comparisons, correlation heatmap, outdoor-temp-vs-power relationship (now real, not decorative), asset-age-vs-fault-rate finding, explicit answers to "what drives failures" (pre-fault vs normal window comparison) and "what drives energy consumption." Figures saved to `reports/figures/`. Ends with key-observations + business-insights summary.

## 3. Task 2 — Predictive Maintenance (`src/nectar/maintenance_model.py`, notebook 03)
- **Label**: fault within next 24h, from the archetype-driven fault episodes.
- **Features** (`features.py`, shared with the dashboard): rolling mean/std/min/max/slope at 1h/6h/24h per sensor, rate-of-change, asset age, asset_type, site/building, operating_mode, occupancy,
missingness flags. Archetype-awareness means slope-based features should have real separating power (e.g. static-pressure slope for AHUs, vibration slope for pumps/chillers).
- **Split**: time-based (train ~70 days / test trailing ~20 days) — no random shuffle.
- **Imbalance**: scale_pos_weight/class_weight; report PR-AUC alongside ROC-AUC.
- **Models**: LightGBM + XGBoost primary, Logistic Regression + RandomForest baselines; select by PR-AUC/F1 on held-out window.
- **Explainability**: SHAP summary/dependence; error analysis on false negatives/positives,
broken out by asset_type/fault-archetype since they now have distinct signatures.
- Deepest task per rubric weight (20%) — most feature/error-analysis rigor goes here.

## 4. Task 3 — Energy Forecasting (`src/nectar/forecasting.py`, notebook 04)
Target: next-24h hourly power_consumption per building.
- Baseline: statsmodels Holt-Winters/SARIMAX per building (documented trade-off vs Prophet: avoids
  Windows Stan-compiler build friction, no material accuracy cost at this scale).
- Primary: XGBoost with calendar features, lags/rolling stats, **plus outdoor_temp/humidity from weather.csv as exogenous regressors** — standard real-world practice, now genuinely available.
- Walk-forward backtest over last ~2 weeks. Metrics: MAE, RMSE, MAPE per building. LSTM/TFT
explicitly considered and deprioritized (documented trade-off: added complexity/training time, no clear accuracy edge at this volume).

## 5. Task 4 — Anomaly Detection (`src/nectar/anomaly.py`, notebook 05)
Statistical thresholding (spikes) + Isolation Forest (multivariate combined states) + rolling- slope/CUSUM (drift) + `ruptures` change-point detection (degradation onset) — each method mapped to the fault/anomaly archetypes that actually produce that shape in the data. Combined score/severity, visualized flagged windows, counts by asset/type/site, business recommendation per category.

## 6. Task 5 — Connectivity Analysis (`src/nectar/graph.py`, notebook 06)
NetworkX DiGraph: Site → Building → Asset + parent_asset_id + connectivity edges (typed, weighted). Hierarchy tree + full dependency graph visualization (pyvis, `cdn_resources='local'`).
Query functions matching the brief's own examples: `get_connected_assets(asset_id)`,
`get_downstream_impact(asset_id)`, `get_upstream_dependencies(asset_id)`,
`get_assets_by_site(site_id)`, `get_isolated_assets()`.
**Failure impact analysis**: simulate Chiller/Pump node removal, compute disconnected downstream subgraph — now backed by real correlated telemetry evidence from §1.6, not just topology — quantify affected assets/occupants, give mitigation recommendations.
**Data quality checks**: duplicate edges, orphan assets, invalid parent-child mappings, self-loops, validated against the deliberately-injected issues from §1.8. Second-deepest task per rubric weight (15%).

## 7. Bonus A — Streamlit Dashboard (`dashboard/app.py`)
Reads precomputed artifacts. Tabs: Site Overview, Asset Health (Task 2 probabilities), Energy
Trends (Task 3 forecasts incl. weather overlay), Anomaly Alerts (Task 4), Asset Connectivity
(interactive graph + Task 5 query functions as a simple UI).

## 7b. Bonus B — Model Deployment + GraphQL (`api/main.py`, `api/schema.py`)
FastAPI service: `POST /predict_failure` (raw telemetry in, feature engineering run
inside the endpoint via the shared `features.py`, failure probability out), plus
`/assets/{id}/downstream_impact` and `/upstream_dependencies`. A Strawberry GraphQL
schema mounted at `/graphql` in the same app implements the brief's example queries
(`connectedAssets`, `downstreamImpact`, `assetsBySite`, `isolatedAssets`) as thin
resolvers over the Task 5 query functions in §6 — no graph logic duplicated. Added
post-submission (2026-07-22) after initially being built, then trimmed from bonus scope,
then restored at the user's request with a real GraphQL implementation this time.

## 8. Reproducibility, Testing, Docs
- `scripts/run_pipeline.py`: one command, full pipeline, writes all artifacts.
- `tests/`: sanity-check generator output ranges/counts, no time-split leakage, graph query
correctness on a fixture — serves the Code Quality rubric line concretely.
- `docs/data_dictionary.md`: the per-asset-type unit/range/standard table (§1.3) formalized — this is the artifact that actually proves the domain-expertise claim to an evaluator.
- `README.md`: setup, run order, architecture (mermaid), all assumptions (synthetic-data
disclosure, weather.csv as added scope, South-Indian-climate mode assumption, statsmodels-over-Prophet, LSTM/TFT deprioritization).
- `reports/Nectar_DS_Challenge_Report.md`: ~5-page equivalent, per-task results + business impact + connectivity findings + assumptions appendix.

## Tech Stack (`requirements.txt`)
pandas, numpy, scikit-learn, lightgbm, xgboost, shap, statsmodels, ruptures, matplotlib, seaborn, networkx, pyvis, streamlit, fastapi, uvicorn, pydantic, strawberry-graphql, pyarrow, jupyter, pytest. (`plotly` was an initially-planned dependency, added to `requirements.txt`, then removed as unused once the dashboard settled on native Streamlit charts — see `docs/build_log.md`.)

## Build Sequence
1. Scaffold folders + `requirements.txt` + `config.py`.
2. `weather.py` → `weather.csv`, sanity-plot per-site seasonal/diurnal shape.
3. `physics.py` + `data_generation.py` → generate all 4 raw datasets; `tests/test_data_generation.py` passes.
4. `preprocessing.py` → processed telemetry.
5. Notebook 02 EDA.
6. `features.py` (+ `tests/test_features.py`).
7. `maintenance_model.py` + notebook 03.
8. `forecasting.py` + notebook 04.
9. `anomaly.py` + notebook 05.
10. `graph.py` (+ `tests/test_graph.py`) + notebook 06.
11. `dashboard/app.py`.
12. `scripts/run_pipeline.py`, `docs/data_dictionary.md`, `reports/...md`, `README.md`, final
    end-to-end run via the pipeline script.

## Verification
- `pytest tests/` green before moving past each relevant build step.
- Generator output: row counts, fault-episode count, injected data-quality issue counts land in
  target ranges (asserted, not eyeballed).
- Each model notebook: metrics computed and sane (AUC > 0.5, no NaN leakage), artifacts saved.
- `streamlit run dashboard/app.py` launches cleanly before calling the build done.

## Assumptions (explicit, documented in README + report)
- All data synthetic, disclosed clearly — patterns engineered to be *learnable and physically
  plausible*, not a claim about real building physics.
- South Indian climate → Cooling/Idle dominate; Heating <1% edge case.
- `weather.csv` added beyond the brief's 3 datasets — real BMS deployments commonly use a weather
  feed; documented as additive scope, not a deviation from the required 3.
- statsmodels in place of Prophet (Windows build friction); XGBoost as primary forecaster.
- Effort/depth allocated per rubric weighting: Predictive Modeling and Connectivity get the most
  rigor; Forecasting/Anomaly solid but leaner.
