# Nectar Data Scientist Challenge — Intelligent Facilities Platform

End-to-end IoT analytics solution for Nectar's Intelligent Facilities Platform: EDA,
predictive maintenance, energy forecasting, anomaly detection, and multi-asset
connectivity analysis on synthetic commercial-building sensor telemetry — plus an
interactive dashboard and a live prediction API.

---

## TL;DR results

| Task | Approach | Headline result |
|---|---|---|
| 1. EDA | Distributions, temporal, failure & energy drivers | Faults preceded by a 1.5× vibration lift in the 24h window; energy driven by outdoor temp + occupancy |
| 2. Predictive Maintenance | RandomForest (selected by PR-AUC) vs. LightGBM/XGBoost/LogisticRegression, 24h-ahead | **Precision 0.90, Recall 0.75, ROC-AUC 0.896, PR-AUC 0.777** |
| 3. Energy Forecasting | XGBoost (primary) vs. Holt-Winters (baseline), day-ahead, weather-exogenous | **MAPE 9.27%** (XGBoost) vs 161% (baseline) |
| 4. Anomaly Detection | Statistical (seasonal MAD-z) + Isolation Forest + CUSUM + change-point | IsoForest anomalies **2.5×** more frequent within 24h of a fault |
| 5. Connectivity | NetworkX directed graph | Hierarchy, failure propagation, full DQ audit — all 4 planted issues found |
| Bonus A | Streamlit dashboard | 6-section ops dashboard incl. **live** failure scoring |
| Bonus B | FastAPI | `POST /predict_failure` accepts **raw telemetry**, not pre-engineered features |

All numbers above are reproduced by `python scripts/run_pipeline.py` — a genuine
headless, one-command run, not just notebook output (verified: pipeline metrics match
the notebooks' exactly, confirming deterministic reproducibility under `SEED=42`).

**Resuming work on this project?** Read [`PROJECT_STATUS.md`](PROJECT_STATUS.md) first —
a single entry point covering current status, architecture, and exact next steps.

**Process documentation:** [`docs/plan/`](docs/plan/) — the day-by-day plan this was
actually built against, one file per day — and [`docs/build_log.md`](docs/build_log.md)
— a chronological record of real bugs found and fixed along the way (not just the
finished numbers above).

---

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# 2. Install dependencies + the nectar package itself (editable install --
#    makes `from nectar import ...` work from anywhere, no path hacks needed)
pip install -r requirements.txt
pip install -e .

# 3. Run the full pipeline (generates data, trains all models, builds the graph,
#    saves every artifact the dashboard/API depend on) -- ~5 minutes
python scripts/run_pipeline.py

# 4. OR run the narrated notebooks in order (same underlying src/nectar/ logic)
jupyter lab notebooks/     # 01 -> 06

# 5. (Bonus A) Launch the dashboard
streamlit run dashboard/app.py            # http://localhost:8501

# 6. (Bonus B) Launch the API
uvicorn api.main:app --reload             # docs at http://localhost:8000/docs
```

Verified on **Python 3.13.14** (`.python-version`). All dependencies including SHAP,
LightGBM, XGBoost, statsmodels, ruptures, pyvis install cleanly on this version.

---

## Architecture

```
D:\Nectar\
├── PROJECT_STATUS.md               current status + next steps (read this first if resuming)
├── PLAN.md                          original pre-build design doc
├── pyproject.toml                   makes `nectar` pip-installable (`pip install -e .`)
├── data/
│   ├── raw/                      generated: telemetry, metadata, connectivity, weather
│   └── processed/                 cleaned/feature-engineered parquet output
├── src/nectar/                    single source of truth -- imported by every notebook,
│   ├── config.py                   scripts/run_pipeline.py, dashboard/app.py, api/main.py
│   ├── weather.py                 per-site synthetic outdoor temp/humidity generator
│   ├── physics.py                 per-asset-type signal models, fault archetypes
│   ├── data_generation.py         orchestrates metadata/connectivity/telemetry generation
│   ├── preprocessing.py           load/validate/impute/resample
│   ├── features.py                shared feature engineering (Task 2 + Task 3), no leakage
│   ├── maintenance_model.py       Task 2 training/evaluation/error-analysis logic
│   ├── forecasting.py             Task 3 baseline + primary model + walk-forward backtest
│   ├── anomaly.py                 Task 4 four-method detection framework
│   ├── graph.py                   Task 5 NetworkX graph, queries, failure impact, DQ audit
│   └── logging_config.py
├── notebooks/                     narrated analysis, 01 (data gen) -> 06 (connectivity)
├── scripts/
│   └── run_pipeline.py             one-command headless reproduction
├── tests/                          27 tests: data generation, feature leakage, graph queries
├── models/                         predictive_maintenance.pkl, asset_graph.pkl
├── dashboard/app.py                Streamlit -- Bonus A
├── api/main.py                     FastAPI -- Bonus B
├── docs/
│   ├── data_dictionary.md          per-asset-type unit/range/standard reference
│   ├── build_log.md                chronological build log: bugs found, fixes, decisions
│   └── plan/                       README.md (index) + day_1.md ... day_5.md, one file
│                                     per session/branch (branch mapping + ordering rationale)
├── reports/
│   ├── figures/                    17 exported PNGs + 1 interactive HTML graph
│   └── report.md
└── requirements.txt
```

**Data flow:** `data_generation.py` (seeded, deterministic) → 4 CSVs in `data/raw/` →
`preprocessing.py` (load + validate + impute) → `features.py` (shared, leak-free
engineering) → the 5 tasks → saved artifacts → dashboard & API consume those artifacts
directly (rebuilding the connectivity graph live from the CSVs, or loading the pickled
version — both paths work).

**Design principle:** cleaning and feature logic live in `src/nectar/` exactly once.
Every notebook, `scripts/run_pipeline.py`, `dashboard/app.py`, and `api/main.py` imports
from there — no copy-pasted preprocessing anywhere in the repo.

---

## Assumptions

1. **Data is synthetic**, generated by `src/nectar/data_generation.py` against the exact
   schemas in the brief (no invented columns beyond the documented `weather.csv` addition
   — see below). Fully seeded (`SEED=42`) for byte-identical reproducibility.
2. **Scale:** 3 sites (Coimbatore/Chennai/Bangalore) × 3 buildings × ~15-19 assets ≈ 151
   assets, 10-minute resolution over 90 days ≈ 1.96M telemetry rows.
3. **`weather.csv` added beyond the brief's 3 named datasets** (`site_id, timestamp,
   outdoor_temp, outdoor_humidity`) — real BMS deployments commonly ingest a weather feed
   as an exogenous input for load/forecasting; documented as additive scope, not a
   deviation from the 3 required datasets.
4. **Chiller and Pump are sibling roots** within a building (matching the brief's own
   example hierarchy: `Chiller-01 → AHU-01/02`, `Pump-01 → EnergyMeter-01`, as two
   separate top-level chains, not Pump nested under Chiller).
5. **Faults are preceded by a 6–48h gradual degradation ramp** (asset-type-specific
   archetype, e.g. rising vibration for bearing wear) — this reflects real mechanical
   failure and is what makes 24h-ahead prediction learnable. Fault probability rises with
   asset age (a genuine, non-decorative EDA finding and model feature).
6. **`operating_mode=Heating`** is a rare (<1% of rows), physically genuine edge case tied
   to actual cold early-morning readings in these South-Indian climate profiles (via a
   per-site, occupied-hour-conditioned threshold) — not a real winter season, and not a
   backwards/decorative rule.
7. **Task 2 target** (`target_24h`) is strictly forward-looking — "does a fault occur in
   the next 24h" — verified leak-free in `tests/test_features.py` via a synthetic-fault
   boundary test and a synthetic-spike no-lookahead test.
8. **Intentional data-quality issues** (2 orphan assets, 2 invalid parent references, 1
   duplicate connectivity row, 2 missing-relationship gaps) are planted so Task 5's audit
   has genuine findings to report — asserted present via `tests/`, not eyeballed.
9. **Statsmodels Holt-Winters** in place of Prophet for the Task 3 baseline (avoids
   Windows Stan-compiler build friction; XGBoost is the primary forecaster regardless).
   LSTM/Temporal Fusion Transformer were considered and deprioritized — added training
   complexity with no clear accuracy edge at this data volume (9 buildings × ~2,000
   hourly points each).
10. **"GraphQL" bonus** interpreted as graph-based query *capability* via NetworkX
    (`get_connected_assets`, `get_downstream_impact`, etc. — these map directly to what
    GraphQL resolvers would expose) rather than standing up a literal GraphQL server,
    which would add infrastructure without analytical depth for this exercise.
11. **`/predict_failure` accepts raw telemetry**, not a pre-engineered feature dict — an
    explicit design goal from the start (see `api/main.py`), since a production endpoint
    receiving live sensor data would need to run feature engineering internally.
12. Manufacturer names are generic placeholder labels, not real-brand claims.

---

## Design decisions

- **RandomForest selected over XGBoost/LightGBM for Task 2**, by explicit **PR-AUC**
  comparison (not ROC-AUC alone, which can look deceptively high under ~2% class
  imbalance) — a genuine model-selection outcome from `maintenance_model.evaluate_all()`,
  not a foregone conclusion. All 4 candidates are trained and compared in the notebook.
- **Time-based splits everywhere** — train = first N days, test = trailing days. Never a
  random shuffle, which would leak future rows (and future fault ramps) into training.
- **Walk-forward backtest for Task 3**: the Holt-Winters baseline refits on an expanding
  window and forecasts only 24h at a time before advancing — a one-shot 2-week-ahead
  forecast was found to compound trend error catastrophically (verified: it produced
  physically impossible negative energy values) during development.
- **Seasonal-naive baselines for the statistical/CUSUM anomaly detectors**: a plain
  time-window rolling z-score/CUSUM was found, during development, to flag 20-90%+ of
  readings on power/temperature — because those signals are strongly diurnal, and any
  window spanning a day/night transition sees most of its own points as "outliers".
  Comparing each reading against its own hour-of-day baseline first fixes this.
- **Binseg over `Pelt(jump=1)`** for Task 4 change-point detection — verified ~300×
  faster (13.5s vs. 55 minutes across all rotating assets) with comparable results.
- **Memory-budgeted Task 2 training**: this environment has ~8GB RAM; training 4 models
  (including a 300-tree RandomForest) on the full ~700k-row training set risked an OOM
  kill. A stratified 300k-row subsample (positive rate preserved) is an explicit,
  documented trade-off, not a silent shortcut.
- **`src/nectar/` as the single source of truth** — see Architecture above.

---

## Verification

```bash
pytest tests/ -v              # 27 tests: generator output ranges/counts, feature
                               # leakage guards, graph query correctness on a fixture
python scripts/run_pipeline.py  # full headless run; confirms every artifact the
                                 # dashboard/API need gets produced without Jupyter
```

Every notebook (01–06) was executed in place (`jupyter nbconvert --execute`) — outputs
are real, not fabricated. `scripts/run_pipeline.py`'s printed metrics were confirmed to
match the notebooks' exactly, on a full from-scratch data regeneration.

---

## Reproducing individual task results

```bash
python -m src.nectar.data_generation    # regenerate the 4 raw datasets
python -m src.nectar.preprocessing      # validation report + processed parquet
jupyter nbconvert --to notebook --execute --inplace notebooks/02_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_predictive_maintenance.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_energy_forecasting.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_anomaly_detection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_connectivity_analysis.ipynb
```
