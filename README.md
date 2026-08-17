# Nectar Data Scientist Challenge — Intelligent Facilities Platform

[![live demo](https://img.shields.io/badge/live%20demo-streamlit-ff4b4b.svg)](https://nectar-intelligent-facilities-platform.streamlit.app)
[![tests](https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform/actions/workflows/tests.yml/badge.svg)](https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**▶ Live dashboard: [nectar-intelligent-facilities-platform.streamlit.app](https://nectar-intelligent-facilities-platform.streamlit.app)**
— real model inference on a committed 4-day data slice, not a screenshot. See
[Hosted deployment](#hosted-deployment-streamlit-community-cloud) for how and why.

End-to-end IoT analytics solution for Nectar's Intelligent Facilities Platform: EDA,
predictive maintenance, energy forecasting, anomaly detection, and multi-asset
connectivity analysis on synthetic commercial-building sensor telemetry — plus an
interactive dashboard with live prediction scoring.

---

## TL;DR results

| Task | Approach | Headline result |
|---|---|---|
| 1. EDA | Distributions, temporal, failure & energy drivers | Faults preceded by a 1.48× vibration lift in the 24h window; energy driven by outdoor temp + occupancy |
| 2. Predictive Maintenance | RandomForest (selected by PR-AUC) vs. LightGBM/XGBoost/LogisticRegression, 24h-ahead | **Precision 0.90, Recall 0.75, ROC-AUC 0.896, PR-AUC 0.777** |
| 3. Energy Forecasting | XGBoost (primary) vs. Holt-Winters (baseline), day-ahead, weather-exogenous | **MAPE 9.27%** (XGBoost) vs 161% (baseline) |
| 4. Anomaly Detection | Statistical (seasonal MAD-z) + Isolation Forest + CUSUM + change-point | IsoForest anomalies **2.5×** more frequent within 24h of a fault |
| 5. Connectivity | NetworkX directed graph | Hierarchy, failure propagation, full DQ audit — all 4 planted issues found |
| 6. Operations Dashboard | Streamlit, consuming the same `src/nectar/` modules | 6-section ops dashboard incl. **live** failure scoring |
| 7. Model Deployment | FastAPI `/predict_failure` + GraphQL | Real HTTP model deployment, and a Strawberry GraphQL schema over the Task 5 graph implementing the brief's own example queries |

All numbers above are reproduced by `python scripts/run_pipeline.py` — a genuine
headless, one-command run, not just notebook output (verified: pipeline metrics match
the notebooks' exactly, confirming deterministic reproducibility under `SEED=42`).

**Reviewing this project?** Start with [`reports/report.md`](reports/report.md) — the
~5-page results write-up covering all 5 tasks, the model-selection reasoning, and the
honest limitations of each result. Then [Setup](#setup) below to run it yourself.

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
#    saves every artifact the dashboard depends on) -- ~8-9 minutes
python scripts/run_pipeline.py

# 4. OR run the narrated notebooks in order (same underlying src/nectar/ logic)
jupyter lab notebooks/     # 01 -> 06

# 5. Launch the operations dashboard (task 6)
streamlit run dashboard/app.py            # http://localhost:8501

# 6. Launch the model-deployment service (task 7)
uvicorn api.main:app --reload             # http://localhost:8000/docs
                                            # http://localhost:8000/graphql (GraphiQL)
```

Verified on **Python 3.13.14** (`.python-version`). All dependencies including SHAP,
LightGBM, XGBoost, statsmodels, ruptures, pyvis install cleanly on this version.

---

## Hosted deployment (Streamlit Community Cloud)

**Live at [nectar-intelligent-facilities-platform.streamlit.app](https://nectar-intelligent-facilities-platform.streamlit.app).**
Cold starts take a few seconds — Community Cloud sleeps idle apps, and the first load
builds features and scores every rotating asset before the page draws.

The full dataset can't be committed — `sensor_telemetry.csv` is 177MB (over GitHub's
100MB file limit) and `dashboard/anomalies.csv` is 99MB, and feature-engineering 1.96M
rows doesn't fit a 1GB hosted container anyway. So the repo carries a **committed
4-day Parquet slice** (`data/demo/`, ~1.9MB total) built by:

```bash
python scripts/build_demo_slice.py     # run after scripts/run_pipeline.py
```

`preprocessing.load_raw()` uses the full CSV whenever it's present and falls back to the
slice only when it isn't (`preprocessing.demo_mode()`), so a local checkout that has run
the pipeline is never silently downgraded — asserted in
`tests/test_preprocessing.py::test_demo_mode_only_when_full_csv_absent_and_slice_present`.

The window ends at `config.DEMO_END` (`2025-02-15 15:00`) rather than at the dataset's
tail. The dashboard's live scoring reads each asset's most recent 36h, and the final 36h
of the full dataset contains no imminent faults — every asset scores 0.10–0.27 against a
0.569 threshold, so the failure-prediction panel renders empty. `DEMO_END` is the hour
with the most rotating-asset fault onsets in the following 24h (re-derive it with
`python scripts/build_demo_slice.py --pick-window`); at that moment the model flags **4
of 79 assets**, top probability 0.999.

**This is a different moment, not different data or a different model.** Every
rolling/lag feature looks strictly backward over ≤24h, so a 4-day window ending at T
produces the same scored feature vector as the full 90 days evaluated at T — verified
directly: features and predicted probabilities match to within 1e-9 between an 87k-row
slice and the 992k-row full history. Live model inference genuinely runs on the hosted
app; the trained `models/predictive_maintenance.pkl` is committed (6.4MB).

To deploy: point Streamlit Community Cloud at this repo with `dashboard/app.py` as the
entrypoint. Cloud resolves the dependency file **next to the entrypoint first**, so it
installs `dashboard/requirements.txt` (8 packages, ~49 with transitives) rather than the
root `requirements.txt` (22, ~159 with transitives) — a much smaller hosted build. Both
files are pinned, so the deployed app runs the same versions the results were produced
with. Set the entrypoint carefully: pointing it at a module with no Streamlit calls
deploys successfully and renders a blank page.

---

## Architecture

```
nectar-intelligent-facilities-platform/
├── .github/workflows/tests.yml     CI: lint + regenerate data + full test suite
├── PROJECT_STATUS.md               current status + next steps (read this first if resuming)
├── PLAN.md                          original pre-build design doc
├── pyproject.toml                   makes `nectar` pip-installable (`pip install -e .`)
├── data/
│   ├── raw/                      generated: telemetry, metadata, connectivity, weather
│   ├── demo/                      committed 4-day Parquet slice (~1.9MB) so the hosted
│   │                                dashboard runs without the gitignored 177MB telemetry CSV
│   └── processed/                 cleaned/feature-engineered parquet output
├── src/nectar/                    single source of truth -- imported by every notebook,
│   ├── config.py                   scripts/run_pipeline.py, dashboard/app.py
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
│   ├── run_pipeline.py             one-command headless reproduction
│   ├── build_demo_slice.py         writes data/demo/ from a completed full run
│   └── demo_expectations.py        prints the hosted-dashboard acceptance checklist
├── tests/                          76 tests: data generation, feature leakage, graph queries,
│                                    forecasting, anomaly detection, maintenance model,
│                                    demo-slice fallback, FastAPI + GraphQL end-to-end
├── models/                         predictive_maintenance.pkl, asset_graph.pkl
├── dashboard/
│   ├── app.py                      Streamlit ops dashboard (task 6)
│   └── requirements.txt            dashboard-only deps for the hosted deployment
├── api/
│   ├── main.py                     FastAPI (task 7): POST /predict_failure + graph endpoints
│   └── schema.py                   Strawberry GraphQL schema over graph.py (task 7)
├── docs/
│   ├── data_dictionary.md          per-asset-type unit/range/standard reference
│   ├── build_log.md                chronological build log: bugs found, fixes, decisions
│   └── plan/                       README.md (index) + day_1.md ... day_5.md, one file
│                                     per session/branch (branch mapping + ordering rationale)
├── reports/
│   ├── figures/                    17 exported PNGs + 1 interactive HTML graph
│   └── report.md                   ~5-page results write-up (start here if reviewing)
├── LICENSE                          MIT
└── requirements.txt
```

**Data flow:** `data_generation.py` (seeded, deterministic) → 4 CSVs in `data/raw/` →
`preprocessing.py` (load + validate + impute) → `features.py` (shared, leak-free
engineering) → the 5 tasks → saved artifacts → the dashboard consumes those artifacts
directly (rebuilding the connectivity graph live from the CSVs, or loading the pickled
version — both paths work).

**Design principle:** cleaning and feature logic live in `src/nectar/` exactly once.
Every notebook, `scripts/run_pipeline.py`, and `dashboard/app.py` import from there —
no copy-pasted preprocessing anywhere in the repo.

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
10. Manufacturer names are generic placeholder labels, not real-brand claims.

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
- **FastAPI `/predict_failure` runs feature engineering inside the request**, calling the
  same `features.build_maintenance_features()` used in training (train/serve parity),
  rather than accepting a pre-engineered 82-feature vector. `pd.get_dummies()` only
  creates one-hot columns for categories present in whatever window it's given, so a
  single-asset request never produces the full training column set on its own — the
  endpoint reindexes to the trained feature list (`bundle["features"]`) before scoring,
  which correctly zero-fills any category that doesn't apply to that request.
- **Strawberry over Graphene/Ariadne for the GraphQL bonus** — code-first, type-hint-driven
  schema definition (`@strawberry.type` / `@strawberry.field`) that mirrors the rest of the
  codebase's typed style, plus a first-party `strawberry.fastapi.GraphQLRouter` so it
  mounts into the same app as the REST API rather than running a second server. The schema
  is a thin query layer over `graph.py` — no graph logic is duplicated, it's the identical
  `get_connected_assets` / `get_downstream_impact` / `get_assets_by_site` /
  `get_isolated_assets` / `failure_impact` functions Task 5's notebook uses.

---

## Verification

```bash
pytest tests/ -v              # 76 tests: generator output ranges/counts, feature
                               # leakage guards, graph query correctness on a fixture,
                               # forecasting/anomaly/maintenance-model coverage, plus
                               # FastAPI + GraphQL end-to-end (test_api.py/test_graphql.py
                               # skip automatically if the pipeline hasn't been run yet)
python scripts/run_pipeline.py  # full headless run; confirms every artifact the
                                 # dashboard/API need gets produced without Jupyter
ruff check .                     # lint (config in pyproject.toml)
```

`tests/test_dashboard.py` executes `dashboard/app.py` through Streamlit's `AppTest`
against the committed demo slice. That distinction matters: an earlier check started
Streamlit and fetched the HTTP root, which only returns the static shell — the script
doesn't run until a browser connects — so it passed while the app was crashing on load.

Those tests prove the app doesn't raise; they don't prove the numbers on screen are
right. For that:

```bash
python scripts/demo_expectations.py   # expected values for the hosted dashboard
```

It prints the slice window, per-site asset counts, expected day-of-week bar labels,
graph size, isolated assets, and exactly which assets should appear above the failure
threshold and on which site — all computed from the committed slice and model, so the
checklist regenerates itself rather than going stale. Note the failure panel is
site-filtered: a site with zero flagged assets is a correct result, not a failure.

**CI** ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs on every push
and PR: lint, then regenerate the raw datasets from scratch, then the full suite. Data
generation runs in CI rather than being skipped — it takes ~30s and means every run
re-verifies the generator itself, not just the modules downstream of it.

**Dependency policy.** `requirements.txt` is a pinned lock file — the exact versions
every committed number was produced with, because a floating `pandas>=2.2` would
undermine the reproducibility claim this project rests on. `pyproject.toml` carries the
looser ranges you'd depend on as a package, with extras: `pip install -e ".[dashboard]"`,
`[api]`, `[notebooks]`, `[dev]`.

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

---

## License

[MIT](LICENSE) — the code is free to use, learn from, and adapt. The synthetic data is
generated by this repo (`src/nectar/data_generation.py`); no real or proprietary
facilities data is included.
