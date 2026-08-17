"""End-to-end tests for the GraphQL bonus service (api/schema.py).

Runs the brief's own example queries verbatim against the live schema via
FastAPI's TestClient hitting POST /graphql -- not a unit test of resolver
functions in isolation, the same HTTP path a real GraphQL client would use.
Requires a full pipeline run first (same data dependency as test_api.py).
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nectar import config, preprocessing  # noqa: E402
from nectar import graph as gmod

TELEMETRY_PATH = os.path.join(config.DATA_RAW_DIR, "sensor_telemetry.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(TELEMETRY_PATH),
    reason="requires a full pipeline run first: python scripts/run_pipeline.py",
)

from api.main import app  # noqa: E402

client = TestClient(app)

CHILLER = "CBE-B1-CHL-01"
AHU = "CBE-B1-AHU-03"
SITE = "CBE"


def _query(q, variables=None):
    resp = client.post("/graphql", json={"query": q, "variables": variables or {}})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "errors" not in body, body.get("errors")
    return body["data"]


def test_show_all_assets_connected_to_chiller():
    """Brief example: 'Show all assets connected to Chiller-01.'"""
    data = _query(f'{{ connectedAssets(assetId: "{CHILLER}") {{ assetId assetType }} }}')
    ids = {a["assetId"] for a in data["connectedAssets"]}
    assert ids  # a real chiller has real connections
    assert CHILLER not in ids  # a node isn't "connected to" itself

    raw = preprocessing.load_raw()
    G = gmod.build_graph(raw["metadata"], raw["connectivity"])
    assert ids == set(gmod.get_connected_assets(G, CHILLER))


def test_downstream_assets_impacted_by_ahu_failure():
    """Brief example: 'Identify all downstream assets impacted by AHU-03 failure.'"""
    data = _query(f'{{ downstreamImpact(assetId: "{AHU}") {{ assetId }} }}')
    ids = [a["assetId"] for a in data["downstreamImpact"]]

    raw = preprocessing.load_raw()
    G = gmod.build_graph(raw["metadata"], raw["connectivity"])
    assert sorted(ids) == gmod.get_downstream_impact(G, AHU)


def test_list_all_assets_under_site():
    """Brief example: 'List all assets under Site A' (this dataset's sites are CBE/CHN/BLR)."""
    data = _query(f'{{ assetsBySite(siteId: "{SITE}") {{ assetId siteId }} }}')
    assets = data["assetsBySite"]
    assert len(assets) > 0
    assert all(a["siteId"] == SITE for a in assets)


def test_find_isolated_assets():
    """Brief example: 'Find isolated assets with no parent or child relationships.'"""
    data = _query("{ isolatedAssets { assetId } }")
    ids = {a["assetId"] for a in data["isolatedAssets"]}

    raw = preprocessing.load_raw()
    G = gmod.build_graph(raw["metadata"], raw["connectivity"])
    assert ids == set(gmod.get_isolated_assets(G))
    assert len(ids) > 0  # this dataset has 2 planted orphan assets


def test_failure_impact_chiller():
    data = _query(f'{{ failureImpact(assetId: "{CHILLER}") {{ '
                   f'downstreamCount operationalImpact mitigation }} }}')
    fi = data["failureImpact"]
    assert fi["downstreamCount"] > 0
    assert "cooling" in fi["operationalImpact"].lower()


def test_unknown_asset_returns_empty_or_null():
    data = _query('{ connectedAssets(assetId: "NOT-A-REAL-ASSET-99") { assetId } '
                   'failureImpact(assetId: "NOT-A-REAL-ASSET-99") { assetId } }')
    assert data["connectedAssets"] == []
    assert data["failureImpact"] is None
