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
real lift is 1.50×, a much more honest and useful number.

---

## 3. Feature engineering

`features.py` — shared by Task 2 training, the dashboard's live scoring, and the API, so
there is exactly one implementation of the feature contract. `tests/test_features.py`
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

## 8. Bonus deliverables

**Dashboard**: built to do *genuine live scoring* (load the saved model, build real
features from each asset's actual recent history, score it) rather than the weaker
"confirm the model file exists" pattern seen in the other project reviewed earlier.
Verified running headless with no server-side exceptions.

**API**: built from the start to accept raw telemetry (`asset_id` or an explicit
`telemetry_window`), not a pre-engineered feature dict — feature engineering runs inside
the endpoint via the same `features.py` used in training. Two bugs surfaced on the first
live test against a running server:

- `pd.get_dummies()` only creates one-hot columns for categories *present* in whatever
  data it's given — scoring a single Chiller never produces `type_AHU`/`type_Pump`/
  `mode_Heating` columns, breaking train/serve parity against the model's expected 82-column
  input. Fixed by reindexing to the trained feature list, missing columns filled with 0.
- A `KeyError` on `timestamp` immediately after — caused by reading `timestamp` *after*
  the reindex above had already dropped it. Fixed by capturing it before reindexing.

Both fixed, then verified end-to-end against a live server: health check, predictions
for two different asset types, the two graph endpoints, a 404 for an unknown asset, and
a 400 with an informative message for a non-rotating asset type.

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
generation run, and both the dashboard and API re-launched and hit live one more time.

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

---

## 11. Post-submission: FastAPI end-to-end tests

After the initial submission (§9-10), the test suite was audited and found to cover
Task 2/5 logic and the data generator thoroughly, but not `api/main.py` itself —
Bonus B had only been verified manually (curl/Swagger). `tests/test_api.py` was added:
9 tests against the live app via FastAPI's `TestClient` (`/health`, `/`,
`/predict_failure` for each rotating asset type, unknown-asset 404, non-rotating-asset
400, and the two graph endpoints incl. their 404 paths) — the same code path a real
HTTP client hits, not a mocked model. Skipped automatically if the full pipeline hasn't
been run yet (needs `data/raw/sensor_telemetry.csv` + `models/*.pkl`). Suite total:
27 -> 37, all passing. `README.md`, `PROJECT_STATUS.md`, and `reports/report.md`
updated to the new count.
- Leftover session/reference material unrelated to the actual deliverable (a prior
  session log referencing the separate, unrelated project mentioned in §0, and a
  redundant copy of the brief) was removed entirely.
