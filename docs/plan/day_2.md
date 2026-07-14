# Day 2 — EDA & Shared Feature Engineering

**Branch:** `session-2-eda-features`

**Goal:** understand what's actually in the generated data before building anything on
top of it, and build the *one* feature-engineering module every later model will import
(no copy-pasted preprocessing anywhere in the repo).

## Tasks

- `notebooks/01_data_generation.ipynb` — narrated generation + sanity plots.
- `notebooks/02_eda.ipynb` — distributions, missing-value analysis, temporal patterns,
  cross-site/cross-asset-type comparisons, correlation structure, explicit "what drives
  failures" / "what drives energy consumption" findings.
- Build `features.py` (rolling/lag features for Task 2, calendar+lag features for
  Task 3) — no-leakage guaranteed by construction, not just by convention.
- Test: `tests/test_features.py` (leakage boundary tests using synthetic single-fault
  and single-spike series).

## Files touched

```
notebooks/01_data_generation.ipynb
notebooks/02_eda.ipynb
src/nectar/features.py
tests/test_features.py
reports/figures/00_weather_sanity.png
reports/figures/01_missing_values.png
reports/figures/02_distributions.png
reports/figures/03_temporal_heatmap.png
reports/figures/04_cross_site.png
reports/figures/05_correlation.png
reports/figures/06_failure_factors.png
reports/figures/07_age_vs_fault.png
reports/figures/08_energy_drivers.png
```

## Deliverable at end of day

Documented, visualized understanding of the dataset, plus the shared feature layer both
remaining modeling tasks depend on.

See `docs/build_log.md` §2 for the real bug found and fixed here (a backward-looking
"near fault" window that diluted the vibration-lift finding from 1.50× down to a
barely-there 1.07×).
