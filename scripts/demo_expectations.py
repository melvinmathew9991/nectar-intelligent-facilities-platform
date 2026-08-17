"""
Print the acceptance checklist for the hosted dashboard, derived from the committed
demo slice.

The hosted app's automated tests (`tests/test_dashboard.py`) prove it renders without
raising. They don't prove the numbers on screen are the right numbers. This script
produces the expected values to check them against -- computed from the same
`data/demo/` slice and the same committed model the deployed app uses, so the checklist
regenerates itself whenever the slice changes instead of going stale in a doc.

Deliberately reads `data/demo/` directly rather than going through
`preprocessing.load_raw()`: the point is to describe what the *hosted* app sees, which
is the slice, regardless of whether the machine running this also has the full dataset.

    python scripts/demo_expectations.py
"""
import os
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import config, features
from nectar import graph as gmod

DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LIVE_SCORING_WINDOW_HOURS = 36   # must match dashboard/app.py


def _rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> None:
    demo_path = os.path.join(config.DATA_DEMO_DIR, "sensor_telemetry.parquet")
    if not os.path.exists(demo_path):
        sys.exit(f"{demo_path} not found -- run scripts/build_demo_slice.py first.")

    tel = pd.read_parquet(demo_path)
    meta = pd.read_csv(os.path.join(config.DATA_RAW_DIR, "asset_metadata.csv"),
                       parse_dates=["installation_date"])
    conn = pd.read_csv(os.path.join(config.DATA_RAW_DIR, "asset_connectivity.csv"))

    print("=" * 72)
    print("HOSTED DASHBOARD -- EXPECTED VALUES")
    print("=" * 72)

    _rule("Demo slice")
    print(f"  window          : {tel.timestamp.min()} -> {tel.timestamp.max()}")
    print(f"  span            : {config.DEMO_DAYS} days (config.DEMO_DAYS)")
    print(f"  telemetry rows  : {len(tel):,}")
    print(f"  size on disk    : {os.path.getsize(demo_path) / 1e6:.1f} MB")

    _rule("Site overview (per site selector)")
    per_site = meta.groupby("site_id").agg(assets=("asset_id", "size"),
                                           buildings=("building_id", "nunique"))
    rot = meta[meta.asset_type.isin(config.ROTATING_TYPES)].groupby("site_id").size()
    per_site["rotating"] = rot
    print(per_site.to_string())

    _rule("Energy trends -- day-of-week chart")
    days = sorted(tel.timestamp.dt.dayofweek.unique())
    print(f"  bars expected   : {len(days)}")
    print(f"  labels expected : {', '.join(DOW_LABELS[d] for d in days)}")
    print("  (three-letter labels, no duplicates -- a repeated label means the")
    print("   categorical-collision bug has regressed; see docs/build_log.md 13.1)")

    _rule("Connectivity graph")
    G = gmod.build_graph(meta, conn)
    print(f"  nodes           : {G.number_of_nodes()}")
    print(f"  edges           : {G.number_of_edges()}")
    print(f"  isolated assets : {', '.join(gmod.get_isolated_assets(G)) or 'none'}")

    _rule("Failure predictions (live model scoring)")
    model_path = os.path.join(config.MODELS_DIR, "predictive_maintenance.pkl")
    if not os.path.exists(model_path):
        print(f"  {model_path} not found -- skipping (run scripts/run_pipeline.py)")
        return

    bundle = joblib.load(model_path)
    model, scaler = bundle["model"], bundle["scaler"]
    feats, thr = bundle["features"], bundle["threshold"]

    cutoff = tel.timestamp.max() - pd.Timedelta(hours=LIVE_SCORING_WINDOW_HOURS)
    df, _ = features.build_maintenance_features(tel[tel.timestamp >= cutoff], meta,
                                                need_target=False)
    latest = df.sort_values("timestamp").groupby("asset_id").tail(1).copy()
    X = latest.reindex(columns=feats, fill_value=0.0).values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    latest["p"] = model.predict_proba(X)[:, 1]
    latest["flagged"] = latest["p"] >= thr

    print(f"  threshold       : {float(thr):.3f}")
    print(f"  assets scored   : {len(latest)} (rotating only)")
    print(f"  probability range: {latest.p.min():.3f} - {latest.p.max():.3f}")
    print("\n  flagged per site (the panel is site-filtered -- a site with 0 is")
    print("  correct, not a failure):")
    by_site = latest.groupby("site_id")["flagged"].sum()
    for site_id, n in by_site.items():
        print(f"    {site_id}: {int(n)}")

    flagged = latest[latest.flagged].sort_values("p", ascending=False)
    if len(flagged):
        print("\n  assets expected above threshold:")
        for r in flagged.itertuples():
            print(f"    {r.asset_id:<18} {r.asset_type:<8} {r.site_id}   {r.p:.1%}")
    else:
        print("\n  NO assets above threshold -- the demo would render an empty panel.")
        print("  Re-pick the window: python scripts/build_demo_slice.py --pick-window")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
