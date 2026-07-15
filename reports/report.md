# Nectar Intelligent Facilities Platform — Data Scientist Challenge Report

**Author:** Melvin Mathew | **Date:** 2026-07-12 | Full source, notebooks, and figures in the accompanying repository.

---

## 1. Problem Understanding

Nectar's Intelligent Facilities Platform ingests continuous sensor telemetry from
interconnected building assets (Chillers, AHUs, Pumps, Energy Meters, Environmental
Sensors) across multiple commercial sites. The brief asks for five analytical
capabilities — EDA, predictive maintenance, energy forecasting, anomaly detection, and
multi-asset connectivity analysis — plus an optional operations dashboard, built on data
the brief itself flagged was not actually supplied. Rather
than a generic noise-based simulator, this submission builds a **domain-grounded
synthetic generator**: per-asset-type physical signal models (ISO 10816-consistent
vibration bands, realistic chilled-water/duct-static/pump-discharge pressure ranges),
a shared weather-and-occupancy-driven cooling-load index that causally couples every
asset in a building, asset-type-specific fault archetypes with genuine multi-hour
degradation ramps, and deliberately planted data-quality defects — documented in full
in `docs/data_dictionary.md`. This is what makes every downstream result below a
finding *in the data*, not just a claim in this report.

---

## 2. Task 1 — Exploratory Data Analysis

Distributions, missing-value analysis (~2.3-2.7% per sensor column, 0% on
power/occupancy), hourly/day-of-week power heatmaps, and cross-site/cross-asset-type
comparisons confirmed the three configured climate profiles (Coimbatore/Chennai/
Bangalore) produce visibly distinct `outdoor_temp` distributions that cascade into
distinct Chiller power distributions per site — the weather-driven coupling works
end-to-end, not just at the generator level.

**What drives failures:** vibration in the **forward-looking** 24h window before a fault
is **1.48× higher** than normal operation (3.23 vs 2.18 mm/s) — a genuine, measurable
precursor, not decorative noise. Fault rate also rises monotonically with asset age
(0.062% → 0.099% across four age brackets), a legitimate wear-out effect built into the
generator's fault-episode sampling, not just its baseline signal offset.

**What drives energy consumption:** `power_consumption` correlates positively with both
`outdoor_temp` (r=0.22) and `occupancy_count` (r=0.33) — both via the shared
`cooling_load_index` driver. `Cooling` mode carries materially higher average power
than `Idle`; `Heating` is a rare (<1% of rows) but genuinely cold-morning-driven edge
case in these South-Indian climate profiles, not a real winter season.

---

## 3. Task 2 — Predictive Maintenance (rubric weight 20%)

**Target:** `target_24h` — will this rotating asset (Chiller/AHU/Pump) fault in the next
24 hours? Strictly forward-looking, verified leak-free in `tests/test_features.py`.
**Features (82):** rolling mean/std/min/max/slope at 1h/6h/24h per sensor
(temperature/vibration/power/pressure), 1-step rate-of-change, asset age, capacity,
asset-type/operating-mode one-hot, per-sensor missingness flags.
**Split:** time-based (first 70 days train / trailing ~20 days test) — no random
shuffle. **Models:** LogisticRegression and RandomForest baselines vs. XGBoost and
LightGBM primary candidates, **selected by PR-AUC** (more honest than ROC-AUC alone
under ~2% class imbalance).

| Model | Precision | Recall | F1 | ROC-AUC | **PR-AUC** |
|---|---|---|---|---|---|
| **RandomForest (selected)** | 0.884 | 0.764 | 0.819 | 0.896 | **0.777** |
| LightGBM | 0.897 | 0.713 | 0.794 | 0.878 | 0.756 |
| XGBoost | 0.871 | 0.696 | 0.774 | 0.879 | 0.745 |
| LogisticRegression | 0.122 | 0.803 | 0.212 | 0.892 | 0.566 |

At the F1-optimal threshold (0.569): **Precision 0.90, Recall 0.75, F1 0.82,
ROC-AUC 0.896**. Top features are dominated by **vibration and pressure slope** over
24h — the model has learned that a *sustained rate of change*, not just an
instantaneous reading, is the strongest failure precursor, matching the physics baked
into the generator's fault archetypes. SHAP summary confirms this via per-prediction
attribution. Error analysis by asset type shows false negatives cluster at the *onset*
of degradation, when the rolling signal is still weak — acceptable for a
maintenance-scheduling use case operating on hours-to-days.

**Business impact:** ops teams get a full day's notice to schedule intervention at this
precision/recall trade-off, reducing unplanned downtime and emergency-repair cost.

**Honest limitation:** metrics reflect a cleaner synthetic degradation ramp than real
equipment noise; the *pipeline and feature strategy* transfer to real telemetry, not
this exact number.

---

## 4. Task 3 — Energy Consumption Forecasting (rubric weight 10%)

24h-ahead hourly `power_consumption` per building. **Primary:** XGBoost with calendar
features (hour/day-of-week/weekend, sin/cos hour encoding), lags of 24/48/72/168h and
rolling means anchored ≥24h in the past (leakage guard: `assert min(lags) >= horizon`,
tested), plus `outdoor_temp`/`outdoor_humidity` as exogenous regressors. **Baseline:**
statsmodels Holt-Winters, walk-forward (refit on an expanding window, forecast only 24h
at a time — a one-shot multi-week forecast was found during development to compound
trend error into physically impossible negative energy values).

| | MAE | RMSE | MAPE |
|---|---|---|---|
| **XGBoost (primary)** | 39.8 | 88.2 | **9.27%** |
| Holt-Winters (baseline) | 618.6 | 1018.5 | 161.0% |

The baseline's poor showing is a genuine, expected finding, not a bug: Holt-Winters'
smooth additive seasonal decomposition cannot represent the sharp, near-step
occupancy-driven transitions in this signal (idle ~65-80 kWh overnight → thousands
within 1-2 hours on weekday mornings) without an explicit calendar input, unlike the
primary model. Feature importance shows `lag_168` (same hour, one week prior)
dominates — a strong weekly rhythm the model correctly leverages.

**Business impact:** ~9% next-day MAPE supports proactive demand-response and
peak-shaving scheduling decisions with reasonable confidence.

---

## 5. Task 4 — Anomaly Detection (rubric weight 10%)

A four-method framework, each mapped to the archetype that produces that shape:

| Method | Catches | Rate |
|---|---|---|
| Statistical (seasonal MAD z-score) | Power spikes, sharp point anomalies | 2.21% |
| Isolation Forest | Subtle multivariate joint-pattern anomalies | 2.00% |
| CUSUM (deseasonalized) | Slow, sustained sensor drift | 0.37% |
| Change-point detection (Binseg) | Degradation onset | 92 points, 79 assets |

**Two real bugs found and fixed during development**, both the same root cause: naively
applying a rolling-window statistical method to a *strongly diurnal* signal
(power/temperature) makes the daily cycle itself look like an anomaly, at rates of
20-90%+. Both statistical thresholding and CUSUM now compare each reading against its
own **hour-of-day baseline** first — bringing rates down to the plausible, actionable
levels above.

**Validation:** Isolation Forest anomaly rate within 24h of a known fault is **2.5×**
higher than elsewhere (4.92% vs 1.94%) — anomalies are a genuine early-warning signal,
independently confirming the same physical degradation signature Task 2's model learns.

**Business recommendations:** route by IsoForest severity score, not flag count;
treat a *cluster* of anomalies (not a single flag) as a maintenance trigger; use
CUSUM/change-point flags specifically on assets Task 2 scores as borderline, since they
catch slow-onset degradation between a classifier's rolling-window updates.

---

## 6. Task 5 — Multi-Asset Connectivity Analysis (rubric weight 15%)

A directed graph (151 nodes, 140 edges) built from `asset_metadata.parent_asset_id`
(hierarchy) + `asset_connectivity.csv` (typed/weighted relationships), mirroring the
brief's own example structure (Chiller and Pump as sibling roots within a building).

**Data-quality audit** — every deliberately planted issue confirmed found (also
asserted in `tests/test_graph.py` on an independent fixture):
- 2 fully isolated (orphan) assets
- 2 assets with an invalid parent reference
- 1 duplicated connectivity row
- 2 metadata-vs-connectivity relationship gaps

**Failure propagation:** the worst-case Chiller failure impacts 6 downstream assets (3
AHUs + 3 EnvSensors) — real operational risk, not just topology, since Task 1 confirmed
cooling load is genuinely occupancy/temperature driven. Query functions
(`get_connected_assets`, `get_downstream_impact`, `get_upstream_dependencies`,
`get_assets_by_site`, `get_isolated_assets`) match the brief's own example queries
exactly and are exposed in the notebook.

**Recommendation:** prioritize redundancy investment on the highest-downstream-count
Chillers/Pumps; flag downstream sensors during an upstream fault so operators triage the
root cause first, not secondary symptoms.

---

## 7. Bonus Deliverable

**Streamlit dashboard** (`dashboard/app.py`): 6 sections including **live** failure
scoring (the saved model is actually invoked against each asset's current feature
vector, not just confirmed present) and an interactive connectivity/failure-impact
explorer. Verified running headlessly with no server-side exceptions.

---

## 8. Assumptions & Trade-offs (full list in `README.md`)

Synthetic data with an added `weather.csv` exogenous feed; Statsmodels over Prophet
(Windows build friction, no material accuracy cost); LSTM/TFT deprioritized (no clear
accuracy edge at this data volume); a memory-budgeted 300k-row training subsample for
Task 2 (documented, not silent) given this environment's ~8GB RAM ceiling training 4
models concurrently.

## 9. Reproducibility

`python scripts/run_pipeline.py` reproduces every number in this report headlessly in
under 5 minutes — verified to match the individually-executed notebooks' metrics
exactly on a full from-scratch data regeneration (`SEED=42`). 49 automated tests
(`pytest tests/`) assert generator output ranges, feature-leakage guards, graph
query correctness, and forecasting/anomaly-detection/maintenance-model behavior.
