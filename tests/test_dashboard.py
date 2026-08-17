"""Execute dashboard/app.py headlessly and assert it renders without raising.

Why this exists: the dashboard was previously "verified" by starting Streamlit and
fetching the HTTP root. That only returns Streamlit's static shell -- the script
itself doesn't execute until a browser session connects over the websocket -- so a
crash inside the app passed the check and was only found once deployed. `AppTest`
actually runs the script, which is the difference between checking that a server
listens and checking that an app works.

The specific bug that motivated this: a fixed 7-element day-of-week label list
(`["M","T","W","T","F","S","S"]`) against a groupby whose length depends on how many
distinct weekdays the data window covers. It crashed on the 4-day demo slice, and on
full data it silently collided duplicate labels so Thu drew over Tue and Sun over Sat.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import config, preprocessing

APP_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dashboard", "app.py"))
DEMO_TELEMETRY = os.path.join(config.DATA_DEMO_DIR, "sensor_telemetry.parquet")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEMO_TELEMETRY),
    reason="demo slice not built -- run scripts/build_demo_slice.py")


def _run_app(monkeypatch):
    """Run the app against the committed demo slice, the way a hosted deploy does."""
    from streamlit.testing.v1 import AppTest

    # Force the hosted-deployment path even on a machine that has the full CSV, so
    # this test exercises what actually ships and stays fast (87k rows, not 1.96M).
    monkeypatch.setattr(preprocessing, "demo_mode", lambda: True)
    return AppTest.from_file(APP_PATH, default_timeout=300).run()


def test_dashboard_renders_without_exception(monkeypatch):
    at = _run_app(monkeypatch)
    assert not at.exception, \
        "dashboard raised during render: " + "; ".join(str(e.value) for e in at.exception)


def test_dashboard_renders_core_sections(monkeypatch):
    """A page that renders zero widgets would pass an exception-only check while
    being useless, so assert the real sections actually made it onto the page."""
    at = _run_app(monkeypatch)
    assert not at.exception

    headings = " ".join(str(s.value) for s in at.subheader).lower()
    for section in ["failure predictions", "anomaly alerts"]:
        assert section in headings, f"missing dashboard section: {section} (got: {headings})"

    # live scoring produces a per-asset table, and the demo window is chosen so the
    # model actually fires -- an empty or all-clear board means something regressed
    assert len(at.dataframe) >= 1, "no dataframes rendered -- live scoring likely failed"
