# Project Status & Context — Read This First

**Purpose:** a single entry point to restore full context on this project fast — what it
is, how it's built, what's done, and what's left — without re-reading the whole repo or
re-deriving decisions already made. If picking this project back up in a new session,
read this file first.

**Last updated:** 2026-07-14. **Status: build complete, tested, verified, cleaned, pushed.**
**Deadline:** 2026-07-14 (5 calendar days from receipt on 2026-07-09) — submitted.

---

## 1. What this is

The Nectar Data Scientist Challenge submission (take-home assessment, role: Data
Scientist). An end-to-end IoT analytics pipeline for a fictional facilities-management
platform: 5 required tasks (EDA, Predictive Maintenance, Energy Forecasting, Anomaly
Detection, Multi-Asset Connectivity) plus 2 bonus apps (Streamlit dashboard, FastAPI
service), built on a physics-grounded synthetic dataset generated from scratch — not
reused from any other project. See `docs/build_log.md` §0 if the "not reused" part needs
context: an unrelated, already-built project was found and read early on, then
deliberately set aside as not being this submission's own work.

## 2. Current status — what's actually done

Everything is built, executed for real (not hand-written), and independently verified:

- [x] Domain-grounded synthetic data generator — 151 assets, 1.96M telemetry rows,
      6 fault archetypes, 4 planted data-quality issues, fully seeded (`SEED=42`)
- [x] Task 1 EDA — notebook executed, real findings (1.50× vibration lift before faults,
      age-correlated fault rate, weather/occupancy-driven energy)
- [x] Task 2 Predictive Maintenance — 4 models compared by PR-AUC, RandomForest selected
      (Precision 0.90 / Recall 0.75 / ROC-AUC 0.896), SHAP + error analysis
- [x] Task 3 Energy Forecasting — XGBoost MAPE 9.27% vs. walk-forward Holt-Winters
      baseline 161% (a real, explained finding, not a bug)
- [x] Task 4 Anomaly Detection — 4 methods, 2.5× validated anomaly lift near known faults
- [x] Task 5 Connectivity Analysis — full graph, brief's example queries, all 4 planted
      DQ issues found
- [x] Bonus A: Streamlit dashboard — live model scoring, verified running headless
- [x] Bonus B: FastAPI — `/predict_failure` accepts raw telemetry, verified live
      end-to-end (health, predictions, graph endpoints, error handling)
- [x] 27/27 automated tests passing (`pytest tests/`)
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

All 5 session branches were created, committed, and merged into `main` in order, then
pushed to the remote:
- `session-1-data-foundation`
- `session-2-eda-features`
- `session-3-maintenance-forecasting`
- `session-4-anomaly-connectivity`
- `session-5-bonuses-docs`

Remote: `https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform.git`.
`main` is up to date with `origin/main`. `pytest tests/` → 27/27 passing (last verified
2026-07-14). Nothing outstanding — the submission is complete and pushed.

## 4. Architecture (condensed — full detail in `README.md`)

```
data_generation.py (seeded) -> 4 CSVs in data/raw/
    -> preprocessing.py (validate + impute)
    -> features.py (shared, leak-free rolling/lag features)
    -> Task 2 (maintenance_model.py) / Task 3 (forecasting.py)
    -> Task 4 (anomaly.py) / Task 5 (graph.py)
    -> models/*.pkl + dashboard/anomalies.csv
    -> dashboard/app.py + api/main.py (consume those artifacts)
```

`src/nectar/` is the single source of truth — every notebook, `scripts/run_pipeline.py`,
`dashboard/app.py`, and `api/main.py` import from it. No copy-pasted logic anywhere.
Installable as a package (`pip install -e .`) via `pyproject.toml`.

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
- Full bug-by-bug history: `docs/build_log.md`. Full day-by-day plan: `docs/plan/README.md`.

## 6. Quick verification (run these to confirm nothing has drifted)

```bash
pytest tests/ -v                     # expect 27 passed
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
