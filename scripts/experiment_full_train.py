"""
Issue #6: measure what the 300k-row memory cap costs.

Trains the Task 2 candidates on the FULL training window instead of the capped
stratified subsample, and prints both sets of metrics side by side.

Deliberately a measurement, not a replacement: it does NOT overwrite
`models/predictive_maintenance.pkl`. Every published number (PR-AUC 0.777), the
hosted demo's flagged assets, and `scripts/demo_expectations.py` all derive from
that artifact, so adopting a new model is a separate, explicit decision.

MEMORY: this is the run the 300k cap exists to avoid. `docs/build_log.md` section 4
records repeated silent process kills training 4 models on ~700k x 82 features with
~8GB of RAM. Mitigations here:
  * models are trained and evaluated ONE AT A TIME, then dropped -- the documented
    OOM trigger was holding four fitted models at once;
  * the telemetry frame is freed after feature engineering;
  * run it in the FOREGROUND. Backgrounding is what turned real tracebacks into a
    bare "killed" last time and cost hours of misdiagnosis.

    python scripts/experiment_full_train.py            # full training set
    python scripts/experiment_full_train.py --cap 500000
"""
import argparse
import gc
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import features, preprocessing
from nectar import maintenance_model as mm
from nectar.logging_config import get_logger

log = get_logger(__name__)


def _mem() -> str:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return f"{vm.available / 1e9:.1f}GB free / {vm.total / 1e9:.1f}GB"
    except ImportError:
        return "psutil not installed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=None,
                    help="row cap to test (default: no cap at all)")
    args = ap.parse_args()

    log.info(f"memory before start: {_mem()}")

    raw = preprocessing.load_raw()
    if preprocessing.demo_mode():
        sys.exit("Running on the demo slice -- this experiment needs the full dataset. "
                 "Run scripts/run_pipeline.py first.")

    df, feats = features.build_maintenance_features(raw["telemetry"], raw["metadata"])
    del raw
    gc.collect()
    log.info(f"features built: {len(df):,} rows x {len(feats)} features | {_mem()}")

    Xtr, ytr, Xte, yte, _, _ = mm.time_split(df, feats, max_train_rows=args.cap)
    del df
    gc.collect()
    log.info(f"train={Xtr.shape} test={Xte.shape} | {_mem()}")

    # One model at a time: fit, score, discard. Holding all four simultaneously is
    # the specific thing that triggered the OOM kills documented in build_log 4.
    rows = []
    for name in ["LogisticRegression", "RandomForest", "XGBoost", "LightGBM"]:
        t0 = time.time()
        log.info(f"training {name} ... | {_mem()}")
        one = mm.train_models(Xtr, ytr, feats, only=name)
        metrics = mm.evaluate_all(one, Xte, yte)
        elapsed = time.time() - t0
        metrics["train_seconds"] = round(elapsed, 1)
        rows.append(metrics)
        log.info(f"{name} done in {elapsed:.0f}s | {_mem()}")
        del one
        gc.collect()

    result = pd.concat(rows, ignore_index=True).sort_values("pr_auc", ascending=False)

    cap_label = "no cap (full training window)" if args.cap is None else f"cap {args.cap:,}"
    print("\n" + "=" * 72)
    print(f"FULL-DATA RETRAIN -- {cap_label}, train rows = {len(ytr):,}")
    print("=" * 72)
    print(result.to_string(index=False))

    print("\nCommitted baseline (300,000 train rows), for comparison:")
    print("             model  precision@0.5  recall@0.5   f1@0.5  roc_auc   pr_auc")
    print("      RandomForest       0.884188    0.763625 0.819496 0.895766 0.776589")
    print("          LightGBM       0.896561    0.713003 0.794315 0.877902 0.755892")
    print("           XGBoost       0.870564    0.696060 0.773593 0.879005 0.745132")
    print("LogisticRegression       0.122451    0.802817 0.212491 0.892298 0.566445")

    best = result.iloc[0]
    delta = float(best["pr_auc"]) - 0.776589
    print(f"\nBest PR-AUC now: {best['model']} at {float(best['pr_auc']):.6f} "
          f"({delta:+.6f} vs the committed RandomForest)")
    print("\nNothing was overwritten. models/predictive_maintenance.pkl is unchanged.")
    print("=" * 72)


if __name__ == "__main__":
    main()
