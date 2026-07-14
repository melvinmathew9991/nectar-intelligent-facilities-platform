# Day 4 — Anomaly Detection & Connectivity Analysis (Tasks 4 & 5)

**Branch:** `session-4-anomaly-connectivity`

**Goal:** the second-heaviest single task (Connectivity, 15%) plus the multi-method
anomaly framework — both benefit from being built right after Tasks 2-3, since Task 4's
validation step (anomaly density near known faults) and Task 5's failure-propagation
analysis both lean on artifacts/understanding from the previous day.

## Tasks

- `anomaly.py` + `notebooks/05_anomaly_detection.ipynb` — four methods (statistical
  thresholding, Isolation Forest, CUSUM, change-point detection), each mapped to the
  archetype that actually produces that shape in the data; validated against known
  faults for a genuine lift metric, not just flag counts.
- `graph.py` + `notebooks/06_connectivity_analysis.ipynb` — hierarchy graph, the brief's
  own example queries, failure-impact simulation, full data-quality audit.
- Test: `tests/test_graph.py` (8 tests on an independent fixture graph — query
  correctness verified before trusting the query functions against the real 151-node
  graph).
- Save `models/asset_graph.pkl` and `dashboard/anomalies.csv` for Day 5.

## Files touched

```
src/nectar/anomaly.py
src/nectar/graph.py
notebooks/05_anomaly_detection.ipynb
notebooks/06_connectivity_analysis.ipynb
tests/test_graph.py
models/asset_graph.pkl
dashboard/anomalies.csv
reports/figures/14_task4_anomalies.png
reports/figures/15_task5_hierarchy.png
reports/figures/16_task5_propagation.png
reports/figures/connectivity_interactive.html
```

## Deliverable at end of day

All 5 core tasks complete, tested, and executed with real notebook output.

See `docs/build_log.md` §6 for the real bugs found here: two of the four anomaly-detection
methods initially flagged 20-90%+ of readings because they didn't account for
`power_consumption`/`temperature` being strongly diurnal signals, and a change-point
detection method that technically worked but took 55 minutes until switched to a faster
algorithm (13.5 seconds after the fix).
