# 5-Day Execution Plan

The brief allows 5 calendar days from receipt. This is the day-by-day breakdown the
build actually followed — one file per day below, each mapping 1:1 to a git branch
(`session-1-...` through `session-5-...`), merged into `main` once that day's slice is
complete, so `main` always has a fully working system at every point in the history, not
just at the very end.

| Day | Focus | Branch | Plan |
|---|---|---|---|
| 1 | Problem understanding, design, data foundation | `session-1-data-foundation` | [day_1.md](day_1.md) |
| 2 | EDA + shared feature engineering | `session-2-eda-features` | [day_2.md](day_2.md) |
| 3 | Predictive Maintenance + Energy Forecasting (Tasks 2-3) | `session-3-maintenance-forecasting` | [day_3.md](day_3.md) |
| 4 | Anomaly Detection + Connectivity Analysis (Tasks 4-5) | `session-4-anomaly-connectivity` | [day_4.md](day_4.md) |
| 5 | Bonus, documentation, final QA | `session-5-bonuses-docs` | [day_5.md](day_5.md) |

Each day's file lists that day's goal, tasks, exact files touched, and end-of-day
deliverable — deliberately scoped so it can be read (and committed) as a self-contained
unit. For what actually went wrong along the way and how it was fixed, see
[`../build_log.md`](../build_log.md).

## Why this ordering, not another one

- **Data foundation first, always** — every other day's numbers are only as trustworthy
  as this day's generator, so it's front-loaded and gets its own dedicated test suite
  before anything else touches it.
- **EDA before modeling** — understanding the data's actual shape (not the intended
  shape) surfaced real issues (e.g. a labeling bug in the "vibration near a fault"
  comparison) before they could quietly undermine a model built on top of them.
- **Tasks 2-3 before Tasks 4-5** — Task 4's validation step and Task 5's narrative both
  reference the fault/degradation understanding built on Day 3; sequencing them after
  means those cross-task references are to *already-verified* results, not forward
  promises.
- **Bonus last, not first** — the bonus app is a thin consumer of artifacts from
  Tasks 2, 4, and 5; building it earlier would have meant building against
  not-yet-finalized model/graph formats and redoing integration work later.
