"""
Build the committed demo slice used by hosted deployments (Streamlit Cloud).

The full dataset is 1.96M telemetry rows / 177MB of CSV, gitignored because it
exceeds GitHub's 100MB file limit -- and too large for a 1GB hosted container to
feature-engineer anyway. This script writes a trailing `config.DEMO_DAYS`-day
slice as Parquet into `data/demo/`, small enough to live in git.

The window ends at `config.DEMO_END`, not at the dataset's tail. The dashboard's
live scoring reads each asset's most recent 36h, and the last 36h of the full
dataset contains no imminent faults -- so a tail slice renders an empty
failure-prediction panel. `--pick-window` reports the hour with the most
rotating-asset fault onsets in the following 24h, which is how DEMO_END was
chosen. Feature values for the scored row are unaffected by the window choice:
every rolling/lag feature looks strictly backward over <=24h, so a short window
ending at T yields the same features as the full 90 days evaluated at T
(asserted in tests/test_features.py::test_trailing_window_matches_full_history).

Run after `scripts/run_pipeline.py`:

    python scripts/build_demo_slice.py
    python scripts/build_demo_slice.py --pick-window   # re-derive a good DEMO_END
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import config, preprocessing
from nectar.logging_config import get_logger

log = get_logger(__name__)


def _window(df: pd.DataFrame, end: pd.Timestamp, days: int) -> pd.DataFrame:
    start = end - pd.Timedelta(days=days)
    return df[(df["timestamp"] > start) & (df["timestamp"] <= end)].reset_index(drop=True)


def pick_window(telemetry: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Report the candidate end-timestamps with the most rotating-asset fault
    onsets in the following 24h -- the moments a 24h-ahead predictor should be
    firing, and therefore the ones that make a non-empty hosted demo."""
    rot = set(metadata.loc[metadata["asset_type"].isin(config.ROTATING_TYPES), "asset_id"])
    f = telemetry[(telemetry["asset_id"].isin(rot)) & (telemetry["fault_flag"] == 1)]
    f = f[["asset_id", "timestamp"]].sort_values(["asset_id", "timestamp"])

    # an episode onset is the first fault row, or one >2h after the previous
    gap = f.groupby("asset_id")["timestamp"].diff()
    onsets = f[gap.isna() | (gap > pd.Timedelta(hours=2))]

    earliest = telemetry["timestamp"].min() + pd.Timedelta(days=config.DEMO_DAYS)
    rows = []
    for T in pd.date_range(earliest.ceil("h"), telemetry["timestamp"].max().floor("h"), freq="1h"):
        nxt = onsets[(onsets["timestamp"] > T) & (onsets["timestamp"] <= T + pd.Timedelta(hours=24))]
        rows.append((T, nxt["asset_id"].nunique()))

    res = (pd.DataFrame(rows, columns=["end_timestamp", "assets_faulting_next_24h"])
           .sort_values(["assets_faulting_next_24h", "end_timestamp"], ascending=[False, False]))
    print(f"{len(onsets)} fault episodes across {len(rot)} rotating assets\n")
    print(res.head(10).to_string(index=False))
    print(f"\nCurrent config.DEMO_END = {config.DEMO_END}")


def main() -> None:
    os.makedirs(config.DATA_DEMO_DIR, exist_ok=True)

    telemetry_csv = os.path.join(config.DATA_RAW_DIR, "sensor_telemetry.csv")
    if not os.path.exists(telemetry_csv):
        sys.exit(f"{telemetry_csv} not found -- run scripts/run_pipeline.py first.")

    telemetry = preprocessing.read_csv_with_parquet_cache(telemetry_csv, parse_dates=["timestamp"])

    if "--pick-window" in sys.argv:
        metadata = pd.read_csv(os.path.join(config.DATA_RAW_DIR, "asset_metadata.csv"))
        pick_window(telemetry, metadata)
        return

    end = pd.Timestamp(config.DEMO_END)
    if not (telemetry["timestamp"].min() < end <= telemetry["timestamp"].max()):
        sys.exit(f"config.DEMO_END ({end}) is outside the telemetry range "
                 f"{telemetry['timestamp'].min()} -> {telemetry['timestamp'].max()}")

    demo_tel = _window(telemetry, end, config.DEMO_DAYS)
    out = os.path.join(config.DATA_DEMO_DIR, "sensor_telemetry.parquet")
    demo_tel.to_parquet(out, index=False, compression="snappy")
    log.info(f"telemetry: {len(telemetry):,} rows -> {len(demo_tel):,} rows "
             f"({os.path.getsize(out) / 1e6:.1f} MB), window ending {end}")

    anom_csv = os.path.join(config.ROOT_DIR, "dashboard", "anomalies.csv")
    if os.path.exists(anom_csv):
        anom = preprocessing.read_csv_with_parquet_cache(anom_csv, parse_dates=["timestamp"])
        demo_anom = _window(anom, end, config.DEMO_DAYS)
        out = os.path.join(config.DATA_DEMO_DIR, "anomalies.parquet")
        demo_anom.to_parquet(out, index=False, compression="snappy")
        log.info(f"anomalies: {len(anom):,} rows -> {len(demo_anom):,} rows "
                 f"({os.path.getsize(out) / 1e6:.1f} MB)")
    else:
        log.warning(f"{anom_csv} not found -- skipping the anomalies slice "
                    "(run notebook 05 or the pipeline to produce it)")

    log.info(f"Demo slice written to {config.DATA_DEMO_DIR}")


if __name__ == "__main__":
    main()
