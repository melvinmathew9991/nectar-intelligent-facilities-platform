# Day 1 — Problem Understanding, Design & Data Foundation

**Branch:** `session-1-data-foundation`

**Goal:** understand the brief well enough to design a dataset worth building the other
four days on top of, then build and test the generator itself. Nothing downstream can be
more trustworthy than this day's output, so it gets disproportionate care relative to its
own rubric weight.

## Tasks

- Read the brief in full; confirm scope against the rubric (Predictive Modeling 20% and
  Connectivity 15% are the two heaviest single-task weights — plan effort accordingly
  across the rest of the week).
- Design the synthetic data model: per-asset-type physical signal profiles grounded in
  real standards (ISO 10816 vibration bands, real chilled-water/duct/pump ranges), a
  shared weather-and-occupancy-driven cooling-load index coupling every asset in a
  building, asset-type-specific fault archetypes with genuine degradation ramps, and
  deliberately planted data-quality defects for Task 5 to find. Written up in `PLAN.md`.
- Build: `config.py`, `weather.py`, `physics.py`, `data_generation.py`, `preprocessing.py`.
- Test: `tests/test_data_generation.py` (14 tests — row counts, fault-episode counts,
  missingness bands, all 4 planted DQ issues, energy-meter reconciliation).

## Files touched

```
requirements.txt
.gitignore
PLAN.md
src/nectar/__init__.py
src/nectar/config.py
src/nectar/logging_config.py
src/nectar/weather.py
src/nectar/physics.py
src/nectar/data_generation.py
src/nectar/preprocessing.py
tests/test_data_generation.py
```

## Deliverable at end of day

A seeded, deterministic, independently-tested data generator producing 4 CSVs — usable
by every subsequent day without modification.

See `docs/build_log.md` §1 for the real bugs found and fixed while building this (the
`Heating` mode never firing, false-positive orphan assets, and non-monotonic fault-vs-age
rate).
