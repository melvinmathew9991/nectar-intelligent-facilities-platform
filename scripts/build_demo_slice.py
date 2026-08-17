"""
Build the committed demo slice used by hosted deployments (Streamlit Cloud).

The full dataset is 1.96M telemetry rows / 177MB of CSV, gitignored because it
exceeds GitHub's 100MB file limit -- and too large for a 1GB hosted container to
feature-engineer anyway. This script writes a trailing `config.DEMO_DAYS`-day
slice as Parquet into `data/demo/`, small enough to live in git.

Trailing (not leading) is deliberate: the dashboard's live failure scoring uses
each asset's most recent 36h, so the slice must end where the full dataset ends
for those scores to match what a local full run produces.

Run after `scripts/run_pipeline.py`:

    python scripts/build_demo_slice.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import config, preprocessing
from nectar.logging_config import get_logger

log = get_logger(__name__)


def _trailing(df: pd.DataFrame, days: int) -> pd.DataFrame:
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    return df[df["timestamp"] >= cutoff].reset_index(drop=True)


def main() -> None:
    os.makedirs(config.DATA_DEMO_DIR, exist_ok=True)

    telemetry_csv = os.path.join(config.DATA_RAW_DIR, "sensor_telemetry.csv")
    if not os.path.exists(telemetry_csv):
        sys.exit(f"{telemetry_csv} not found -- run scripts/run_pipeline.py first.")

    telemetry = preprocessing.read_csv_with_parquet_cache(telemetry_csv, parse_dates=["timestamp"])
    demo_tel = _trailing(telemetry, config.DEMO_DAYS)
    out = os.path.join(config.DATA_DEMO_DIR, "sensor_telemetry.parquet")
    demo_tel.to_parquet(out, index=False, compression="snappy")
    log.info(f"telemetry: {len(telemetry):,} rows -> {len(demo_tel):,} rows "
             f"({os.path.getsize(out) / 1e6:.1f} MB)")

    anom_csv = os.path.join(config.ROOT_DIR, "dashboard", "anomalies.csv")
    if os.path.exists(anom_csv):
        anom = preprocessing.read_csv_with_parquet_cache(anom_csv, parse_dates=["timestamp"])
        demo_anom = _trailing(anom, config.DEMO_DAYS)
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
