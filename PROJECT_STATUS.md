# Project Status & Context — Read This First

**Purpose:** a single entry point to restore full context on this project fast — what it
is, how it's built, what's done, and what's left — without re-reading the whole repo or
re-deriving decisions already made. If picking this project back up in a new session,
read this file first.

**Last updated:** 2026-07-22. **Status: build complete, tested, verified, cleaned, pushed;
post-submission bonus work in progress (see §3.1).**
**Deadline:** 2026-07-14 (5 calendar days from receipt on 2026-07-09) — submitted.

---

## 1. What this is

The Nectar Data Scientist Challenge submission (take-home assessment, role: Data
Scientist). An end-to-end IoT analytics pipeline for a fictional facilities-management
platform: 5 required tasks (EDA, Predictive Maintenance, Energy Forecasting, Anomaly
Detection, Multi-Asset Connectivity) plus a bonus Streamlit dashboard, built on a
physics-grounded synthetic dataset generated from scratch — not reused from any other
project. See `docs/build_log.md` §0 if the "not reused" part needs context: an
unrelated, already-built project was found and read early on, then deliberately set
aside as not being this submission's own work.

## 2. Current status — what's actually done

Everything is built, executed for real (not hand-written), and independently verified:

- [x] Domain-grounded synthetic data generator — 151 assets, 1.96M telemetry rows,
      6 fault archetypes, 4 planted data-quality issues, fully seeded (`SEED=42`)
- [x] Task 1 EDA — notebook executed, real findings (1.48× vibration lift before faults,
      age-correlated fault rate, weather/occupancy-driven energy)
- [x] Task 2 Predictive Maintenance — 4 models compared by PR-AUC, RandomForest selected
      (Precision 0.90 / Recall 0.75 / ROC-AUC 0.896), SHAP + error analysis
- [x] Task 3 Energy Forecasting — XGBoost MAPE 9.27% vs. walk-forward Holt-Winters
      baseline 161% (a real, explained finding, not a bug)
- [x] Task 4 Anomaly Detection — 4 methods, 2.5× validated anomaly lift near known faults
- [x] Task 5 Connectivity Analysis — full graph, brief's example queries, all 4 planted
      DQ issues found
- [x] Bonus A: Streamlit dashboard — live model scoring, verified running headless,
      performance-optimized (~2.2x faster first load: ~30s -> ~13.6s, see
      `docs/build_log.md` §11)
- [x] Bonus B: FastAPI `/predict_failure` + graph endpoints, plus a real Strawberry
      GraphQL schema (`/graphql`) over Task 5's graph — restored 2026-07-22 (see §3.1;
      originally built in session 5, removed in `trim-bonus-scope`, now re-added at the
      user's request with an actual GraphQL implementation this time, not just a claim
      that the existing query functions satisfied it)
- [x] 74/74 automated tests passing (`pytest tests/`) — extended post-submission with
      `test_anomaly.py`, `test_forecasting.py`, `test_maintenance_model.py` (22 tests)
      to close a coverage gap: those three modules had shipped in the pipeline with
      zero tests; `test_preprocessing.py` (4 tests) and 2 more in `test_features.py`
      added alongside the dashboard performance work (see `docs/build_log.md` §11);
      `test_api.py` (10) + `test_graphql.py` (6) added with the Bonus B restoration
- [x] All 6 notebooks confirmed executed (checked programmatically: `execution_count`
      populated, zero error outputs — not eyeballed)
- [x] `scripts/run_pipeline.py` — one-command headless reproduction, verified to produce
      metrics matching the notebooks exactly on a full from-scratch regeneration
- [x] Documentation: `README.md`, `docs/data_dictionary.md`, `reports/report.md`
      (~5-page results summary), `docs/build_log.md`, `docs/plan/README.md` +
      `docs/plan/day_1.md`...`day_5.md`
- [x] Repo structure/naming cleaned to standard (see `docs/build_log.md` §10) —
      `pyproject.toml` added, `.gitignore` finalized, no stray/leftover files

## 3. Git — done

All 6 session branches were created, committed, and merged into `main` in order (the
first 5 via direct merge, the 6th via a reviewed GitHub PR), then pushed to the remote:
- `session-1-data-foundation`
- `session-2-eda-features`
- `session-3-maintenance-forecasting`
- `session-4-anomaly-connectivity`
- `session-5-bonuses-docs`
- `session-6-api-tests-docs` — post-submission documentation and test-suite
  maintenance, merged via
  [PR #1](https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform/pull/1)

One further branch was merged after that:
- `trim-bonus-scope` — removed the deployment bonus and the graph-query bonus framing,
  keeping the dashboard as the one bonus deliverable, merged via
  [PR #2](https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform/pull/2)

Remote: `https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform.git`.
The Bonus B restoration (§3.1) was committed on branch `bonus-fastapi-graphql`, pushed,
and merged into `main` via
[PR #4](https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform/pull/4)
on 2026-07-22. Local `main` is up to date with `origin/main` as of the last pull.

### 3.1 Post-submission: Bonus B restored (2026-07-22)

At the user's explicit request, the Model Deployment + GraphQL bonus removed in
`trim-bonus-scope` (§3, PR #2) was rebuilt:
- `api/main.py` — FastAPI service, restored close to its original `ac88cfe` form
  (verified still compatible with the current `features.py`/`graph.py` signatures —
  nothing else changed underneath it since removal).
- `api/schema.py` — **new**, a real GraphQL schema (Strawberry, mounted at `/graphql` via
  `strawberry.fastapi.GraphQLRouter`) implementing the brief's own example queries
  (`connectedAssets`, `downstreamImpact`, `assetsBySite`, `isolatedAssets`, plus
  `failureImpact` and `upstreamDependencies`). Unlike the original submission, this is an
  actual GraphQL implementation, not a re-framing of the existing `graph.py` functions as
  satisfying a GraphQL bonus.
- `tests/test_api.py` (10 tests, restored) + `tests/test_graphql.py` (6 tests, new) — both
  hit the live app via `TestClient`, skip automatically if the pipeline hasn't been run.
- `requirements.txt`: `fastapi`, `uvicorn`, `pydantic`, `strawberry-graphql[fastapi]`
  added back.
- Verified: `pytest tests/` → 72/72 passing; a real `uvicorn api.main:app` process was
  started and `/health`, `/docs`, `/graphql` all responded correctly before being killed.

**Done since:** committed, pushed (branch `bonus-fastapi-graphql`), and merged into `main`
via PR #4; `PLAN.md`, `reports/report.md`, and `docs/build_log.md` were all updated in the
same pass to describe both bonuses (no longer just the dashboard). The confidential brief
PDF and the internal `docs/PROJECT_AUDIT_REPORT.md` were removed from the working tree and
added to `.gitignore` so they can't be re-added by accident.

## 4. Architecture (condensed — full detail in `README.md`)

```
data_generation.py (seeded) -> 4 CSVs in data/raw/
    -> preprocessing.py (validate + impute)
    -> features.py (shared, leak-free rolling/lag features)
    -> Task 2 (maintenance_model.py) / Task 3 (forecasting.py)
    -> Task 4 (anomaly.py) / Task 5 (graph.py)
    -> models/*.pkl + dashboard/anomalies.csv
    -> dashboard/app.py (consumes those artifacts)
    -> api/main.py + api/schema.py (FastAPI + GraphQL, same artifacts/modules)
```

`src/nectar/` is the single source of truth — every notebook, `scripts/run_pipeline.py`,
and `dashboard/app.py` import from it. No copy-pasted logic anywhere. Installable as a
package (`pip install -e .`) via `pyproject.toml`.

## 5. Key things to remember if resuming work

- **This machine has ~8GB RAM.** Training multiple models concurrently (esp.
  RandomForest) can trigger silent process kills. If running heavy notebook executions
  in the background produces a bare "killed" with no error, **switch to foreground
  execution** — that's what actually surfaced the real bugs last time (background was
  masking real Python tracebacks as generic kills). See `docs/build_log.md` §4.
- **Don't trust rolling-window statistics on diurnal signals** (power, temperature)
  without deseasonalizing first — this caused two separate near-90%-false-flag-rate bugs
  in Task 4. See `docs/build_log.md` §6.
- **All data is deterministic** (`SEED=42`) — `python scripts/run_pipeline.py`
  regenerates everything identically. Verified once already; trust it.
- **`data/raw/sensor_telemetry.parquet` and `dashboard/anomalies.parquet` are
  gitignored Parquet caches**, not source data — auto-regenerated from their CSVs
  via an mtime check (see `preprocessing.read_csv_with_parquet_cache()`). Safe to
  delete anytime; they'll rebuild on the next read. See `docs/build_log.md` §11.
- Full bug-by-bug history: `docs/build_log.md`. Full day-by-day plan: `docs/plan/README.md`.

## 6. Quick verification (run these to confirm nothing has drifted)

```bash
pytest tests/ -v                     # expect 74 passed
python scripts/run_pipeline.py       # expect ~5 min, metrics matching README's TL;DR table
```

## 7. Where to look for more detail

| Need | File |
|---|---|
| Setup, architecture, assumptions, design decisions | `README.md` |
| Full results write-up (~5-page report) | `reports/report.md` |
| Data model / units / standards per asset type | `docs/data_dictionary.md` |
| Every real bug found and how it was fixed | `docs/build_log.md` |
| Day-by-day plan, mapped to future git branches | `docs/plan/README.md` + `docs/plan/day_1.md`...`day_5.md` |
| Original pre-build design doc | `PLAN.md` |
