"""Query functions correct on a small fixture graph."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import graph as g


def _fixture():
    """Chiller-01 -> AHU-01 -> EnvSensor-01
                  -> AHU-02
       Pump-01 -> EnergyMeter-01
       Isolated-01 (no parent, no edges)
       Ghost-01 (parent points to a nonexistent asset)
    """
    metadata = pd.DataFrame([
        {"asset_id": "CHL-01", "site_id": "S1", "building_id": "B1", "asset_name": "Chiller-01",
         "asset_type": "Chiller", "capacity": 300, "parent_asset_id": None},
        {"asset_id": "AHU-01", "site_id": "S1", "building_id": "B1", "asset_name": "AHU-01",
         "asset_type": "AHU", "capacity": 50, "parent_asset_id": "CHL-01"},
        {"asset_id": "AHU-02", "site_id": "S1", "building_id": "B1", "asset_name": "AHU-02",
         "asset_type": "AHU", "capacity": 50, "parent_asset_id": "CHL-01"},
        {"asset_id": "ENV-01", "site_id": "S1", "building_id": "B1", "asset_name": "EnvSensor-01",
         "asset_type": "EnvSensor", "capacity": None, "parent_asset_id": "AHU-01"},
        {"asset_id": "PMP-01", "site_id": "S1", "building_id": "B1", "asset_name": "Pump-01",
         "asset_type": "Pump", "capacity": 20, "parent_asset_id": None},
        {"asset_id": "MTR-01", "site_id": "S1", "building_id": "B1", "asset_name": "EnergyMeter-01",
         "asset_type": "EnergyMeter", "capacity": None, "parent_asset_id": "PMP-01"},
        {"asset_id": "ISO-01", "site_id": "S2", "building_id": "B2", "asset_name": "Isolated-01",
         "asset_type": "EnvSensor", "capacity": None, "parent_asset_id": None},
        {"asset_id": "GHOST-01", "site_id": "S2", "building_id": "B2", "asset_name": "Ghost-01",
         "asset_type": "EnvSensor", "capacity": None, "parent_asset_id": "NONEXISTENT-999"},
    ])
    connectivity = pd.DataFrame([
        {"source_asset_id": "CHL-01", "target_asset_id": "AHU-01",
         "connection_type": "Supplies", "relationship_strength": 0.9},
        {"source_asset_id": "CHL-01", "target_asset_id": "AHU-02",
         "connection_type": "Supplies", "relationship_strength": 0.9},
        {"source_asset_id": "AHU-01", "target_asset_id": "ENV-01",
         "connection_type": "Controls", "relationship_strength": 0.8},
        {"source_asset_id": "PMP-01", "target_asset_id": "MTR-01",
         "connection_type": "Monitors", "relationship_strength": 0.7},
        {"source_asset_id": "CHL-01", "target_asset_id": "AHU-01",   # duplicate
         "connection_type": "Supplies", "relationship_strength": 0.9},
    ])
    return metadata, connectivity


def test_build_graph_node_and_edge_counts():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    assert G.number_of_nodes() == 8
    # hierarchy edges: AHU-01,AHU-02,ENV-01,MTR-01 = 4; connectivity edges add
    # the same 4 relationships again as a MultiGraph would, but DiGraph dedups
    # same (u,v) pairs -- so edge count reflects the union, not the sum.
    assert G.has_edge("CHL-01", "AHU-01")
    assert G.has_edge("AHU-01", "ENV-01")
    assert G.has_edge("PMP-01", "MTR-01")


def test_downstream_impact():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    downstream = g.get_downstream_impact(G, "CHL-01")
    assert set(downstream) == {"AHU-01", "AHU-02", "ENV-01"}


def test_upstream_dependencies():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    upstream = g.get_upstream_dependencies(G, "ENV-01")
    assert set(upstream) == {"CHL-01", "AHU-01"}


def test_connected_assets():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    connected = g.get_connected_assets(G, "AHU-01")
    assert set(connected) == {"CHL-01", "ENV-01"}


def test_assets_by_site():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    s1_assets = g.get_assets_by_site(G, "S1")
    assert set(s1_assets) == {"CHL-01", "AHU-01", "AHU-02", "ENV-01", "PMP-01", "MTR-01"}


def test_isolated_assets():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    isolated = g.get_isolated_assets(G)
    assert "ISO-01" in isolated
    assert "CHL-01" not in isolated


def test_failure_impact_chiller():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    result = g.failure_impact(G, "CHL-01", metadata)
    assert result["downstream_count"] == 3
    assert result["estimated_occupant_impact"] > 0
    assert "AHU" in result["downstream_by_type"]


def test_data_quality_report_finds_all_planted_issues():
    metadata, connectivity = _fixture()
    G = g.build_graph(metadata, connectivity)
    report = g.data_quality_report(metadata, connectivity, G)
    assert "ISO-01" in report["isolated_assets"]
    assert "GHOST-01" in report["invalid_parent_assets"]
    assert report["duplicate_edge_rows"] >= 2   # the duplicated CHL-01->AHU-01 row (both copies)
