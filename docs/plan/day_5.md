# Day 5 — Bonus, Documentation & Final QA

**Branch:** `session-5-bonuses-docs`

**Goal:** the bonus dashboard, then everything needed to make the submission reviewable
and reproducible by someone who wasn't there for the first four days.

## Tasks

- `dashboard/app.py` — Streamlit, 6 sections including **live** model scoring against
  each asset's actual current feature vector (not a "model file exists" placeholder).
- `scripts/run_pipeline.py` — one-command headless reproduction of the entire system,
  used to prove reproducibility empirically (its output was checked to match the
  individually-executed notebooks' metrics exactly) rather than just asserting it.
- `docs/data_dictionary.md`, `README.md`, `reports/report.md` — the reference material a
  reviewer actually needs: setup, architecture, assumptions, design decisions, and the
  ~5-page results summary the brief asks for.
- **Final QA pass** (this is the step most plans skip and shouldn't): re-run the full
  test suite, programmatically confirm every notebook's `execution_count`/error state
  (not eyeballed), cross-check artifact timestamps for mutual consistency, and
  re-launch the dashboard live one more time before calling it done.
- Repository cleanup and structure/naming pass, then push.

## Files touched

```
dashboard/app.py
scripts/run_pipeline.py
docs/data_dictionary.md
docs/build_log.md
docs/plan/README.md
docs/plan/day_1.md ... day_5.md
README.md
reports/report.md
```

## Deliverable at end of day

A complete, independently-verified, reviewable submission.

See `docs/build_log.md` §10 for the repository cleanup pass that preceded the push.
