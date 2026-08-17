# Build Log — How This Project Actually Got Built

A chronological, honest account of the full build: decisions made, bugs found and
fixed, and pivots along the way. Kept separate from `README.md` (which documents the
finished system) because this is the *process* record — the brief explicitly asks for
"thought process, trade-offs, and engineering decisions," not just the final numbers.

---

## 0. Starting point

`D:\Nectar` began the project holding a draft design (`PLAN.md`, a domain-grounded
synthetic-data architecture) but no actual implementation. A separate, already-built
project existed elsewhere on disk from earlier exploration — it was read and verified in
detail, then explicitly set aside: **it was not this project's own work**, and building
the real submission meant starting from `PLAN.md`'s design and writing every module from
scratch. That correction is the actual starting point of the build described below.

The original challenge brief (PDF) was read directly to confirm scope: 5 core tasks
(EDA 15%, Predictive Maintenance 20%, Forecasting 10%, Anomaly Detection 10%,
Connectivity 15%) plus Problem Understanding (10%), Feature Engineering (10%), Code
Quality (5%), Communication (5%) — and a 5-calendar-day submission window from receipt.

---

## 1. Data foundation

Built in order: `config.py` (central parameter tables) → `weather.py` (per-site
seasonal+diurnal synthetic weather) → `physics.py` + `data_generation.py` (the core
generator) → `preprocessing.py`.

**Bugs found and fixed while building the generator** (each caught by actually running
the code and checking real output distributions, not by inspection):

| Bug | Symptom | Fix |
|---|---|---|
| `operating_mode="Heating"` never fired (0% of rows) | Fixed 18°C threshold never overlapped with occupied hours (cold troughs happen overnight, occupancy is 9am-6pm) | Per-site threshold calibrated on the *occupied-hour* temperature distribution instead of a flat global cutoff |
| Main energy meters flagged as false-positive "orphans" in the DQ audit | They legitimately had no connectivity edges (aggregation roots), tripping the same isolated-node check as the *real* planted orphan fixtures | Gave each main meter explicit `Monitors` edges to its building's Chillers/Pumps |
| `test_dq_missing_relationships_detected` failed after the fix above | The random edge-dropper picked one of the new main-meter edges, which has no metadata `parent_asset_id` counterpart to detect | Restricted the droppable-edge pool to edges that actually mirror a metadata parent link |
| Fault rate vs. asset age was non-monotonic despite the EDA narrative claiming otherwise | Fault *count* was sampled uniformly at random per asset — age only affected baseline signal offset, never actual fault probability | Made fault-episode count age-weighted (binomial mean rising from ~1.3 for new assets to ~2.3 for the oldest) |

Result: `tests/test_data_generation.py` — 14 tests (row counts, fault episode counts,
missingness bands, all 4 planted DQ issues, energy-meter reconciliation) — all passing.

---

## 2. Task 1 — EDA

Built `notebooks/01_data_generation.ipynb` and `02_eda.ipynb`, executed for real via
`jupyter nbconvert --execute` (every output in this project is genuine execution output,
never hand-written).

**Bug found:** the "vibration near a fault vs. normal" comparison used a *backward*-looking
24h window (`rolling(144).max()`), which folds in 24h of post-fault recovery alongside
the real pre-fault ramp, diluting the finding to a barely-there 1.07× lift. Fixed to a
genuinely forward-looking window (matching Task 2's actual prediction framing) — the
real lift is 1.48×, a much more honest and useful number.

---

## 3. Feature engineering

`features.py` — shared by Task 2 training and the dashboard's live scoring, so there is
exactly one implementation of the feature contract. `tests/test_features.py`
specifically tests for look-ahead leakage: a synthetic single-fault series confirms the
24h-ahead target flips at the correct forward boundary (not before), and a synthetic
spike placed near the end of a series is confirmed to never affect rolling features
computed near the start.

---

## 4. Task 2 — Predictive Maintenance

This is where the build hit its biggest infrastructure problem: **this machine has only
~8GB of RAM**, and training 4 models (including a 300-tree RandomForest) on ~700k rows
of 82-feature data repeatedly triggered silent process kills.

**The debugging path itself is worth recording** because it wasn't obvious at first:
early attempts ran the training notebook in the *background* and it kept coming back
"killed" with zero error output — looking exactly like an OOM kill. After reducing model
sizes, casting features to `float32`, and adding a stratified training-row cap, it *still*
got killed. The actual breakthrough was switching from background to **foreground**
execution — which immediately surfaced a real Python traceback that had been getting
swallowed by the backgrounding mechanism: a `pandas` `groupby().apply()` version quirk
that silently dropped the target column during subsampling. Fixed, then a second
foreground run surfaced a second real bug: `np.select`'s default value had an
incompatible dtype against a string choicelist under this `numpy` version. Both fixed,
and only then did the notebook complete cleanly — the "OOM" story was real *and* there
were two independent code bugs hiding behind it.

**Final result:** 4 models (LogisticRegression, RandomForest, XGBoost, LightGBM) compared
by **PR-AUC** (deliberately not ROC-AUC alone, which looks deceptively good under ~2%
class imbalance). RandomForest won — Precision 0.90 / Recall 0.75 / ROC-AUC 0.896 — a
genuine model-selection outcome, not a foregone conclusion.

---

## 5. Task 3 — Energy Forecasting

The XGBoost primary model worked cleanly on the first real run (MAPE ≈ 9%). The
Holt-Winters baseline did not.

**Bug found:** a one-shot 336-hour-ahead (14-day) Holt-Winters forecast compounded trend
error catastrophically — MAPE **3318%**, including physically impossible *negative*
energy predictions. The fix required rethinking the baseline's methodology, not just
tuning a parameter: rebuilt as genuine **walk-forward** (refit on an expanding window,
forecast only 24h at a time, advance, repeat) with the trend component dropped entirely
and predictions clipped to non-negative. This brought it down to a stable, still
meaningfully-worse-than-primary MAPE of 161% — a legitimate baseline comparison instead
of a broken one.

---

## 6. Task 4 — Anomaly Detection

Two of the four detection methods broke on first run, both for the **same underlying
reason**: `power_consumption` and `temperature` are strongly diurnal/bimodal signals
(near-zero at night, full load in the day), and a plain time-window rolling statistic
can't tell a normal day/night transition from a genuine anomaly — every transition looks
like an "outlier" relative to whatever's typical in that specific window.

- **Statistical thresholding**: flagged **21.5%** of readings (target was a "high bar,
  actionable" few percent). Fixed by comparing each reading against its own **hour-of-day**
  baseline instead of a raw rolling window — down to 2.21%.
- **CUSUM**: flagged **28-93%** of readings depending on asset type, for the identical
  reason, compounded by the accumulator never resetting after firing (so one early false
  alarm stayed "active" for the rest of the series). Fixed with the same deseasonalizing
  approach plus a proper reset-on-alarm — down to 0.37%.
- **Change-point detection** (`ruptures`) technically worked but took **55 minutes**
  across all rotating assets (`Pelt(jump=1)`'s exact search doesn't scale well here).
  Switched to `Binseg` — same-quality breakpoints, **13.5 seconds**, a ~300× speedup that
  made iteration on the rest of the notebook actually feasible.

**Validation:** Isolation Forest anomaly rate within 24h of a known fault is 2.5× higher
than elsewhere — a real, independently-computed confirmation of the same physical signal
Task 2's model learns.

---

## 7. Task 5 — Connectivity Analysis

Built cleanly against a dedicated fixture graph (`tests/test_graph.py`, 8 tests covering
every query function) before running against the real 151-node graph. No bugs — the
earlier Task 1/4 debugging had already forced a habit of validating against a small,
hand-checkable case before trusting the full dataset.

---

## 8. Bonus deliverable

**Dashboard**: built to do *genuine live scoring* (load the saved model, build real
features from each asset's actual recent history, score it) rather than the weaker
"confirm the model file exists" pattern seen in the other project reviewed earlier.
Verified running headless with no server-side exceptions.

---

## 9. Reproducibility & final verification

`scripts/run_pipeline.py` was written to prove the whole thing is genuinely
reproducible headlessly, not just inside notebooks — running it end-to-end regenerated
every dataset and artifact from scratch and printed metrics that matched the
individually-executed notebooks' numbers **exactly** (same PR-AUC to six decimal places,
same MAPE, same DQ findings), confirming the `SEED=42` determinism claim empirically
rather than just asserting it.

A final verification pass before considering the build "done": full 27-test suite,
every notebook's `execution_count` and error-output checked programmatically (not
eyeballed), all artifact file timestamps cross-checked to confirm they came from the same
generation run, and the dashboard re-launched and hit live one more time.

---

## 10. Repository cleanup

Before pushing, the repo was reviewed for structure and naming standards, which
surfaced a few things worth fixing:

- `notebooks/lib/` — a folder of bundled JS/CSS that `pyvis` had written into the
  notebook source directory as a side effect of `cdn_resources='local'`. Fixed at the
  source (`cdn_resources='remote'`) and the notebook re-executed to confirm it doesn't
  come back.
- Internal meta-scripts that had been used to programmatically assemble the notebook
  `.ipynb` files were removed — useful during the build, but not a standard artifact for
  a finished data-science submission; the notebooks themselves (with real executed
  output) are the actual deliverable.
- `reports/Nectar_DS_Challenge_Report.md` renamed to `reports/report.md` for naming
  consistency with the rest of `docs/`/`reports/` (lowercase snake_case), while
  `README.md`/`PLAN.md` keep the conventional uppercase treatment for top-level docs.
- `.gitignore` finalized to exclude the two generated files that are either over
  GitHub's hard size limit (`sensor_telemetry.csv`, 169MB) or close enough to be risky
  (`dashboard/anomalies.csv`, 95MB) — both fully reproducible via
  `python scripts/run_pipeline.py`.
- Leftover session/reference material unrelated to the actual deliverable (a prior
  session log referencing the separate, unrelated project mentioned in §0, and a
  redundant copy of the brief) was removed entirely.

---

## 11. Dashboard performance optimization

Post-submission, the dashboard's first load after a restart was timed rather than
just described as "a bit slow": ~30s total. Added temporary timing instrumentation
(sidebar/section captions for data load, graph build, live-scoring time) to measure
each step instead of guessing, then fixed the two real bottlenecks it revealed.

| Bottleneck | Root cause | Fix | Measured result |
|---|---|---|---|
| Live feature engineering (~20.6s) | `load_live_failure_scores()` ran `build_maintenance_features()` across the FULL 90-day, ~1.96M-row telemetry history, even though only the LAST row per asset is ever used for live scoring | Trim telemetry to a trailing 36h window before feature engineering (1.5x the largest 24h rolling window -- a safe margin, not the bare minimum) | Proven mathematically exact, not an approximation: every rolling/slope feature there is strictly backward-looking within its own window, so history beyond it cannot affect the last row |
| Unused label computation | The same call also computed `target_24h` (the forward-looking fault label), which live scoring immediately discards | Added `need_target: bool = True` to `build_maintenance_features()`, defaulting to preserve existing training behavior; dashboard passes `need_target=False` | Smaller than expected (~7%) -- the label pass is a single rolling op vs. the main pass's 60 rolling ops per asset, so it was never the dominant cost |
| Data load (~9.3s) | `sensor_telemetry.csv` (177MB) and `dashboard/anomalies.csv` (99MB) were re-parsed from raw CSV text on every server (re)start | Added `preprocessing.read_csv_with_parquet_cache()` -- a transparent Parquet mirror, auto-invalidated via an mtime check against the source CSV so it can never silently serve stale data | ~4x faster overall (9.31s -> 2.35s) on a warm cache -- 8x on telemetry alone, ~30x on anomalies alone |

**Total: ~30.0s -> ~13.6s first-load time (~2.2x faster overall)**, measured against
the real generated dataset, not estimated. Every fix has its own test proving the
claim rather than trusting it: `tests/test_features.py::test_trailing_window_matches_full_history`
(byte-identical output, trimmed window vs. full history) and
`test_need_target_false_skips_label_without_changing_features`, plus 4 new tests in
`tests/test_preprocessing.py` (cache output matches a plain CSV read, a cache hit
serves identical content without rewriting the cache file, a stale cache is
regenerated rather than silently served, and the helper works with no
`parse_dates` too).

**Honest remaining bottleneck, left alone deliberately:** live scoring is still
~10.9s, now dominated by the per-asset Python overhead of
`groupby("asset_id").apply(...)` inside `build_maintenance_features()` (145
assets), not row count -- both fixes above target row count and hit diminishing
returns once that became the real ceiling. A further fix would mean rewriting
the rolling-feature computation to avoid per-group Python `.apply()` entirely --
a materially bigger change to code shared with actual model training, not folded
into this pass.

Result: `pytest tests/` -> 55/55 passing (up from 49).

---

## 12. Bonus B restored: FastAPI deployment + real GraphQL (2026-07-22)

Post-submission, at the user's explicit request, the Model Deployment + GraphQL bonus
scope removed in `trim-bonus-scope` (§9 above did the removal narrative implicitly via
`PROJECT_STATUS.md`; the actual removal commit was `9f2051e`) was rebuilt.

**FastAPI (`api/main.py`):** restored close to its original form from the deleted
`ac88cfe` commit. Checked first whether anything underneath it had changed enough to
break it: `features.build_maintenance_features(rotating_only=, need_target=)` and every
`graph.py` query function used by the API kept their exact signatures across the
removal (confirmed via `git show 9f2051e -- src/nectar/features.py` -- only a docstring
changed), so the restored endpoint code needed no logic changes, only re-adding the file.

**GraphQL (`api/schema.py`) -- new, and deliberately different from the original
submission's approach:** the original submission's "GraphQL bonus" claim (removed in
`9f2051e`) was really just re-framing the existing `graph.py` query functions as
satisfying a GraphQL bonus -- they're plain Python functions, not a GraphQL schema. This
time an actual GraphQL layer was built: `strawberry-graphql`, mounted at `/graphql` via
`strawberry.fastapi.GraphQLRouter` in the same FastAPI app (one process, not a second
server). The schema implements the brief's example queries verbatim --
`connectedAssets`, `downstreamImpact`, `assetsBySite`, `isolatedAssets` -- plus
`upstreamDependencies` and `failureImpact` for completeness, all as thin resolvers over
the unchanged `graph.py` functions (no graph logic duplicated).

**Verification, not just "should work":**
- `pytest tests/test_api.py tests/test_graphql.py` -> 16/16 passing (10 REST + 6
  GraphQL), run against the live app via `TestClient`, same code path a real HTTP
  client hits.
- A real `uvicorn api.main:app` process was started on a throwaway port; `/health`,
  `/docs` (Swagger UI), and `/graphql` (GraphiQL) were each confirmed responding
  `200` via `curl` before the process was killed -- not just the in-process
  `TestClient`, an actual server socket.
- Full suite: `pytest tests/` -> 72/72 passing (up from 55).

**Dependency additions:** `fastapi`, `pydantic`, `strawberry-graphql[fastapi]` (pulls in
`graphql-core`); `uvicorn` and `httpx` were already present as transitive deps of
Streamlit/testing tooling.

`README.md`, `PROJECT_STATUS.md`, `reports/report.md`, and `PLAN.md` were all updated in
this same pass to describe both bonuses. Committed on `bonus-fastapi-graphql`, pushed,
and merged into `main` via PR #4.

---

## 13. Hosted-deployment prep (2026-08-17)

Putting the Bonus A dashboard on Streamlit Community Cloud surfaced a problem the local
build had never had to face: **Cloud deploys from the GitHub repo, and the data isn't
in it.** `sensor_telemetry.csv` is 177MB (over GitHub's 100MB hard limit) and
`dashboard/anomalies.csv` is 99MB -- both gitignored for exactly that reason. Even if
they were committed, a ~1GB hosted container can't feature-engineer 1.96M rows.

**Fix:** a committed 4-day Parquet slice (`data/demo/`, ~1.9MB) plus a
`preprocessing.demo_mode()` fallback that only engages when the full CSV is *absent* --
so a local checkout that has run the pipeline still reads the real 1.96M-row dataset and
is never silently downgraded to the slice. That gating is the part worth testing, and
`tests/test_preprocessing.py` now asserts it in both directions.

**The bug in the obvious version:** the first slice was a plain *trailing* window,
which seemed right -- it makes the hosted scores match a local full run exactly. But
scoring it revealed the final 36h of the dataset contains no imminent faults: all 79
rotating assets scored 0.10-0.27 against a 0.569 threshold, so the dashboard's
failure-prediction panel rendered completely empty. Technically correct, useless as a
demo.

Rather than lower the threshold (which would have been dishonest), the window was moved:
`--pick-window` scans every candidate hour for the most rotating-asset fault onsets in
the following 24h, which selected `2025-02-15 15:00`. There the model flags 4 of 79
assets with a top probability of 0.999.

**Verified this is a different moment, not different data:** every rolling/lag feature
looks strictly backward over <=24h, so a short window ending at T must produce the same
scored feature vector as the full 90 days evaluated at T. Confirmed empirically --
features and predicted probabilities matched to within 1e-9 between the slice
and the 992k-row full history. (Same property `tests/test_features.py::
test_trailing_window_matches_full_history` already asserts for the dashboard's 36h trim.)

**Also:** the app was smoke-tested headless with the full-size files hidden, serving with
zero exceptions; `LICENSE` (MIT) added.

**What the first real deploy corrected (2026-08-17):** two assumptions made here were
wrong, and only deploying surfaced them.

1. `dashboard/requirements.txt` was written on the assumption that Streamlit Cloud
   prefers a dependency file next to the entrypoint. It does not — it resolves at the
   repo root, and the build log shows it installing all 159 packages from
   `requirements.txt` (`WARN: More than one requirements file detected ... Used: uv with
   requirements.txt`). The file is kept for minimal local dashboard installs, and the
   README claim was corrected.
2. The Python version was expected to matter (the pins target 3.13). Cloud used **3.14.7**
   and every pin resolved cleanly, so it didn't.

**And one real failure:** the 14-day slice booted the server but rendered a blank page --
`load_all()` runs at module import, so anything that dies there yields no page at all,
and 304k telemetry rows plus 159k anomaly rows exceeded what the container would carry.
Cut to **4 days** (86,976 telemetry rows, 45,504 anomaly rows, 1.9MB total), which is
still strictly more than anything the dashboard computes: live scoring reads a trailing
36h and the widest rolling feature is 24h. Re-verified after the cut -- the scored
features and probabilities still match a full-history run at the same moment to 1e-9,
same 4 assets flagged, same 0.999 top probability.
