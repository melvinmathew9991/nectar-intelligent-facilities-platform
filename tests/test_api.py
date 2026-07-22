"""End-to-end tests for the FastAPI bonus service (api/main.py).

Exercises the live app via FastAPI's TestClient -- same code path a real
HTTP client hits (uvicorn/curl/Swagger), not a mocked model. Requires the
full pipeline to have been run at least once (`python scripts/run_pipeline.py`)
since it needs data/raw/sensor_telemetry.csv (gitignored, too large for git)
and models/*.pkl (tracked) to exist.
"""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nectar import config  # noqa: E402

TELEMETRY_PATH = os.path.join(config.DATA_RAW_DIR, "sensor_telemetry.csv")

pytestmark = pytest.mark.skipif(
    not os.path.exists(TELEMETRY_PATH),
    reason="requires a full pipeline run first: python scripts/run_pipeline.py",
)

from api.main import app  # noqa: E402

client = TestClient(app)

ROTATING_ASSETS = {
    "Chiller": "CBE-B1-CHL-01",
    "AHU": "CBE-B1-AHU-01",
    "Pump": "CBE-B1-PMP-01",
}
NON_ROTATING_ASSET = "CBE-B1-EVS-01"  # EnvSensor
UNKNOWN_ASSET = "NOT-A-REAL-ASSET-99"


def test_health_reports_model_loaded():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "RandomForest"
    assert body["n_features"] > 0


def test_root_lists_endpoints():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "/predict_failure" in body["endpoints"]
    assert "/graphql" in body["endpoints"]


@pytest.mark.parametrize("asset_type,asset_id", list(ROTATING_ASSETS.items()))
def test_predict_failure_valid_rotating_asset(asset_type, asset_id):
    resp = client.post("/predict_failure", json={"asset_id": asset_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["asset_id"] == asset_id
    assert body["asset_type"] == asset_type
    assert 0.0 <= body["failure_probability_24h"] <= 1.0
    assert body["will_fail"] == (body["failure_probability_24h"] >= body["threshold"])
    assert body["recommendation"] in (
        "Schedule maintenance within 24h", "No action needed",
    )


def test_predict_failure_unknown_asset_is_404():
    resp = client.post("/predict_failure", json={"asset_id": UNKNOWN_ASSET})
    assert resp.status_code == 404


def test_predict_failure_non_rotating_asset_is_400():
    resp = client.post("/predict_failure", json={"asset_id": NON_ROTATING_ASSET})
    assert resp.status_code == 400
    assert "predictive maintenance only applies to" in resp.json()["detail"]


def test_downstream_impact_known_asset():
    resp = client.get(f"/assets/{ROTATING_ASSETS['Chiller']}/downstream_impact")
    assert resp.status_code == 200
    body = resp.json()
    assert body["downstream_count"] == len(body["downstream_assets"])
    assert body["downstream_count"] > 0


def test_upstream_dependencies_known_asset():
    resp = client.get(f"/assets/{ROTATING_ASSETS['Chiller']}/upstream_dependencies")
    assert resp.status_code == 200
    assert isinstance(resp.json()["upstream_assets"], list)


def test_graph_endpoints_unknown_asset_are_404():
    assert client.get(f"/assets/{UNKNOWN_ASSET}/downstream_impact").status_code == 404
    assert client.get(f"/assets/{UNKNOWN_ASSET}/upstream_dependencies").status_code == 404
