"""
Task 5 -- Multi-Asset Connectivity Analysis. Builds a directed graph from
asset_metadata.parent_asset_id (hierarchy) + asset_connectivity.csv
(typed/weighted relationship edges), exposes query functions matching the
brief's own examples, simulates failure propagation, and audits data quality.
"""
from __future__ import annotations
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx

from . import config
from .logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Build graph
# ---------------------------------------------------------------------------
def build_graph(metadata: pd.DataFrame, connectivity: pd.DataFrame) -> nx.DiGraph:
    G = nx.DiGraph()
    for _, a in metadata.iterrows():
        G.add_node(a["asset_id"], asset_type=a["asset_type"], site_id=a["site_id"],
                   building_id=a["building_id"], asset_name=a["asset_name"],
                   capacity=a["capacity"])

    asset_ids = set(metadata["asset_id"])
    for _, a in metadata.iterrows():
        p = a["parent_asset_id"]
        if isinstance(p, str) and p in asset_ids:
            G.add_edge(p, a["asset_id"], relation="parent_of", weight=1.0)

    valid_edges = 0
    for _, e in connectivity.iterrows():
        if e["source_asset_id"] in asset_ids and e["target_asset_id"] in asset_ids:
            G.add_edge(e["source_asset_id"], e["target_asset_id"],
                       relation=e["connection_type"], weight=e["relationship_strength"])
            valid_edges += 1

    log.info(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
             f"({valid_edges} valid connectivity edges)")
    return G


# ---------------------------------------------------------------------------
# 2. Query functions (match the brief's example queries)
# ---------------------------------------------------------------------------
def get_connected_assets(G: nx.DiGraph, asset_id: str) -> list[str]:
    """All assets connected to `asset_id` (successors + predecessors)."""
    if asset_id not in G:
        return []
    return sorted(set(G.successors(asset_id)) | set(G.predecessors(asset_id)))


def get_downstream_impact(G: nx.DiGraph, asset_id: str) -> list[str]:
    """Every asset reachable downstream from `asset_id` -- what breaks if it fails."""
    if asset_id not in G:
        return []
    return sorted(nx.descendants(G, asset_id))


def get_upstream_dependencies(G: nx.DiGraph, asset_id: str) -> list[str]:
    """Every asset `asset_id` depends on (ancestors)."""
    if asset_id not in G:
        return []
    return sorted(nx.ancestors(G, asset_id))


def get_assets_by_site(G: nx.DiGraph, site_id: str) -> list[str]:
    return sorted(n for n, d in G.nodes(data=True) if d.get("site_id") == site_id)


def get_isolated_assets(G: nx.DiGraph) -> list[str]:
    """No parent AND no connectivity edges at all (fully disconnected)."""
    return sorted(n for n in G.nodes if G.in_degree(n) == 0 and G.out_degree(n) == 0)


# ---------------------------------------------------------------------------
# 3. Failure impact analysis
# ---------------------------------------------------------------------------
def failure_impact(G: nx.DiGraph, asset_id: str, metadata: pd.DataFrame,
                     avg_occupants_per_building: float = 80.0) -> dict:
    downstream = get_downstream_impact(G, asset_id)
    types_affected = pd.Series([G.nodes[n]["asset_type"] for n in downstream]).value_counts().to_dict()
    building_id = G.nodes[asset_id]["building_id"] if asset_id in G else None
    asset_type = G.nodes[asset_id]["asset_type"] if asset_id in G else None

    if asset_type == "Chiller":
        impact = ("Loss of cooling capacity for the building -- comfort/compliance risk, "
                   "and any downstream AHUs relying on this chiller can no longer condition air.")
        mitigation = ("Fail over to a redundant chiller if the building has one; if not, "
                       "prioritize emergency dispatch given the number of downstream AHUs/zones "
                       "affected, and flag all downstream sensors so operators aren't misled by "
                       "secondary anomalies rather than the root cause.")
    elif asset_type == "Pump":
        impact = ("Loss of fluid circulation -- affects any equipment relying on this pump's "
                   "loop (flow/energy metering included), risking indirect overheating "
                   "elsewhere in the loop even though the pump itself isn't a comfort asset.")
        mitigation = ("Check for a redundant pump/loop path; if single point of failure, "
                       "treat as high priority even though downstream count is often small, "
                       "since a stalled loop can cascade into upstream equipment stress.")
    else:
        impact = f"{len(downstream)} downstream assets lose their upstream signal/control input."
        mitigation = "Investigate downstream sensor/meter continuity; low physical urgency otherwise."

    return {
        "asset_id": asset_id, "asset_type": asset_type, "building_id": building_id,
        "downstream_count": len(downstream), "downstream_assets": downstream,
        "downstream_by_type": types_affected,
        "estimated_occupant_impact": (avg_occupants_per_building
                                        if asset_type in ("Chiller", "Pump") and downstream else 0),
        "operational_impact": impact, "mitigation": mitigation,
    }


# ---------------------------------------------------------------------------
# 4. Data-quality audit
# ---------------------------------------------------------------------------
def data_quality_report(metadata: pd.DataFrame, connectivity: pd.DataFrame, G: nx.DiGraph) -> dict:
    asset_ids = set(metadata["asset_id"])
    orphans = metadata.loc[metadata["parent_asset_id"].isna()
                             & ~metadata["asset_id"].isin(
                                 set(connectivity["source_asset_id"]) | set(connectivity["target_asset_id"])
                             ), "asset_id"].tolist()
    invalid_parent = metadata.loc[metadata["parent_asset_id"].notna()
                                    & ~metadata["parent_asset_id"].isin(asset_ids), "asset_id"].tolist()
    dup_edges = connectivity[connectivity.duplicated(["source_asset_id", "target_asset_id"], keep=False)]
    invalid_edges = connectivity[~connectivity["target_asset_id"].isin(asset_ids)
                                   | ~connectivity["source_asset_id"].isin(asset_ids)]
    self_loops = list(nx.selfloop_edges(G))
    cycles = list(nx.simple_cycles(G)) if G.number_of_nodes() < 500 else []

    valid_parent = metadata[metadata["parent_asset_id"].notna() & metadata["parent_asset_id"].isin(asset_ids)]
    has_edge = set(zip(connectivity["source_asset_id"], connectivity["target_asset_id"]))
    missing_relationships = [(r.parent_asset_id, r.asset_id) for r in valid_parent.itertuples()
                              if (r.parent_asset_id, r.asset_id) not in has_edge]

    return {
        "orphan_assets": orphans, "invalid_parent_assets": invalid_parent,
        "duplicate_edge_rows": len(dup_edges), "invalid_edges": invalid_edges[
            ["source_asset_id", "target_asset_id"]].values.tolist(),
        "self_loops": self_loops, "cycles": cycles,
        "missing_relationships": missing_relationships,
        "isolated_assets": get_isolated_assets(G),
    }


# ---------------------------------------------------------------------------
# 5. Persistence
# ---------------------------------------------------------------------------
def save_graph(G: nx.DiGraph, path: str | None = None) -> str:
    path = path or os.path.join(config.MODELS_DIR, "asset_graph.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    log.info(f"Saved graph -> {path}")
    return path


def load_graph(path: str | None = None) -> nx.DiGraph:
    path = path or os.path.join(config.MODELS_DIR, "asset_graph.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)
