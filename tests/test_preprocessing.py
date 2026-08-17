"""Verify the generic CSV-with-Parquet-cache helper
(preprocessing.read_csv_with_parquet_cache, shared by load_raw()'s telemetry
read and the dashboard's anomalies.csv read) returns data identical to a
plain CSV read, uses the cache on a hit, and never serves stale data once
the source CSV changes."""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import config, preprocessing as pp


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cache_matches_plain_csv_read(tmp_path):
    csv_path = str(tmp_path / "sensor_telemetry.csv")
    _write_csv(csv_path, [
        {"timestamp": "2025-01-01 00:00:00", "asset_id": "A1", "temperature": 10.0},
        {"timestamp": "2025-01-01 00:10:00", "asset_id": "A1", "temperature": 11.0},
    ])

    expected = pd.read_csv(csv_path, parse_dates=["timestamp"])
    cached = pp.read_csv_with_parquet_cache(csv_path, parse_dates=["timestamp"])

    pd.testing.assert_frame_equal(cached.reset_index(drop=True), expected.reset_index(drop=True))
    assert os.path.exists(str(tmp_path / "sensor_telemetry.parquet"))


def test_second_call_reads_from_cache_with_identical_content(tmp_path):
    csv_path = str(tmp_path / "sensor_telemetry.csv")
    _write_csv(csv_path, [
        {"timestamp": "2025-01-01 00:00:00", "asset_id": "A1", "temperature": 10.0},
    ])

    first = pp.read_csv_with_parquet_cache(csv_path, parse_dates=["timestamp"])
    parquet_path = str(tmp_path / "sensor_telemetry.parquet")
    parquet_mtime_before = os.path.getmtime(parquet_path)

    second = pp.read_csv_with_parquet_cache(csv_path, parse_dates=["timestamp"])

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    # cache file itself must not have been rewritten on a hit
    assert os.path.getmtime(parquet_path) == parquet_mtime_before


def test_stale_cache_is_regenerated_not_silently_served(tmp_path):
    csv_path = str(tmp_path / "sensor_telemetry.csv")
    parquet_path = str(tmp_path / "sensor_telemetry.parquet")

    _write_csv(csv_path, [
        {"timestamp": "2025-01-01 00:00:00", "asset_id": "A1", "temperature": 10.0},
    ])
    first = pp.read_csv_with_parquet_cache(csv_path, parse_dates=["timestamp"])
    assert first["temperature"].iloc[0] == 10.0

    # rewrite the CSV with different content, then force its mtime safely
    # ahead of the cached parquet's mtime (avoids filesystem mtime-resolution
    # flakiness rather than relying on wall-clock sleep timing)
    _write_csv(csv_path, [
        {"timestamp": "2025-01-01 00:00:00", "asset_id": "A1", "temperature": 999.0},
    ])
    newer = os.path.getmtime(parquet_path) + 10
    os.utime(csv_path, (newer, newer))

    second = pp.read_csv_with_parquet_cache(csv_path, parse_dates=["timestamp"])
    assert second["temperature"].iloc[0] == 999.0, \
        "stale parquet cache was served instead of detecting the newer CSV"


def test_works_without_parse_dates(tmp_path):
    """The dashboard's other CSV reads (metadata, connectivity) don't all
    have a timestamp column -- confirm parse_dates=None is a valid, working
    default, not just an untested code path."""
    csv_path = str(tmp_path / "asset_metadata.csv")
    _write_csv(csv_path, [{"asset_id": "A1", "asset_type": "Chiller"}])

    cached = pp.read_csv_with_parquet_cache(csv_path)
    assert list(cached.columns) == ["asset_id", "asset_type"]
    assert cached["asset_id"].iloc[0] == "A1"


def test_demo_mode_only_when_full_csv_absent_and_slice_present(tmp_path, monkeypatch):
    """demo_mode() gates the hosted-deployment fallback. It must be False
    whenever the full telemetry CSV exists -- otherwise a local checkout that
    has run the pipeline would silently serve the 14-day slice instead of the
    real 1.96M-row dataset."""
    raw_dir, demo_dir = tmp_path / "raw", tmp_path / "demo"
    raw_dir.mkdir()
    demo_dir.mkdir()
    monkeypatch.setattr(config, "DATA_RAW_DIR", str(raw_dir))
    monkeypatch.setattr(config, "DATA_DEMO_DIR", str(demo_dir))

    # neither present
    assert pp.demo_mode() is False

    # slice only -> hosted deployment
    pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01"]), "asset_id": ["A1"]}) \
        .to_parquet(demo_dir / "sensor_telemetry.parquet", index=False)
    assert pp.demo_mode() is True

    # full CSV present as well -> local run must win
    _write_csv(str(raw_dir / "sensor_telemetry.csv"),
               [{"timestamp": "2025-01-01 00:00:00", "asset_id": "A1"}])
    assert pp.demo_mode() is False, \
        "full telemetry CSV present but demo_mode() still True -- a real local run would be downgraded"


def test_load_demo_returns_none_when_slice_missing(tmp_path, monkeypatch):
    """load_demo() is called for anomalies, which may legitimately be absent
    (the dashboard degrades to 'no precomputed anomalies') -- it must return
    None rather than raising."""
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    monkeypatch.setattr(config, "DATA_DEMO_DIR", str(demo_dir))

    assert pp.load_demo("anomalies") is None

    pd.DataFrame({"asset_id": ["A1"], "score": [0.5]}).to_parquet(
        demo_dir / "anomalies.parquet", index=False)
    loaded = pp.load_demo("anomalies")
    assert loaded is not None and loaded["asset_id"].iloc[0] == "A1"
