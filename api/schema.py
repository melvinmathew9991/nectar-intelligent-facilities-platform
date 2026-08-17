"""
Task 5 bonus -- a real GraphQL schema (Strawberry) over the connectivity
graph built in `src/nectar/graph.py`. Implements the brief's own example
queries verbatim:

    - Show all assets connected to Chiller-01.
    - Identify all downstream assets impacted by AHU-03 failure.
    - List all assets under Site A.
    - Find isolated assets with no parent or child relationships.

The graph is loaded once (module-level cache) from the same
`models/asset_graph.pkl` artifact Task 5's notebook saves, or built fresh
from the raw CSVs if that pickle isn't present -- no duplicated graph logic,
this module only queries `graph.py`.
"""
import os
import sys

import strawberry
from strawberry.fastapi import GraphQLRouter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import config, preprocessing  # noqa: E402
from nectar import graph as gmod

GRAPH_PATH = os.path.join(config.MODELS_DIR, "asset_graph.pkl")

_G = None
_metadata = None


def _get_graph():
    global _G, _metadata
    if _G is None:
        raw = preprocessing.load_raw()
        _metadata = raw["metadata"]
        _G = (gmod.load_graph(GRAPH_PATH) if os.path.exists(GRAPH_PATH)
              else gmod.build_graph(raw["metadata"], raw["connectivity"]))
    return _G


@strawberry.type
class Asset:
    asset_id: str
    asset_type: str | None
    asset_name: str | None
    site_id: str | None
    building_id: str | None
    capacity: float | None


@strawberry.type
class FailureImpact:
    asset_id: str
    asset_type: str | None
    building_id: str | None
    downstream_count: int
    downstream_assets: list[str]
    operational_impact: str
    mitigation: str


def _node_to_asset(G, asset_id: str) -> Asset:
    d = G.nodes[asset_id]
    return Asset(asset_id=asset_id, asset_type=d.get("asset_type"),
                 asset_name=d.get("asset_name"), site_id=d.get("site_id"),
                 building_id=d.get("building_id"), capacity=d.get("capacity"))


@strawberry.type
class Query:

    @strawberry.field(description="All assets directly connected to the given asset "
                                    "(parents, children, and typed connectivity edges "
                                    "in either direction). Example: connectedAssets(assetId: \"CBE-B1-CHL-01\")")
    def connected_assets(self, asset_id: str) -> list[Asset]:
        G = _get_graph()
        if asset_id not in G:
            return []
        return [_node_to_asset(G, a) for a in gmod.get_connected_assets(G, asset_id)]

    @strawberry.field(description="Every asset downstream of the given asset -- what is "
                                    "impacted if it fails. Example: "
                                    "downstreamImpact(assetId: \"CBE-B1-AHU-01\")")
    def downstream_impact(self, asset_id: str) -> list[Asset]:
        G = _get_graph()
        if asset_id not in G:
            return []
        return [_node_to_asset(G, a) for a in gmod.get_downstream_impact(G, asset_id)]

    @strawberry.field(description="Every asset the given asset depends on (ancestors).")
    def upstream_dependencies(self, asset_id: str) -> list[Asset]:
        G = _get_graph()
        if asset_id not in G:
            return []
        return [_node_to_asset(G, a) for a in gmod.get_upstream_dependencies(G, asset_id)]

    @strawberry.field(description="All assets under a given site. Example: "
                                    "assetsBySite(siteId: \"CBE\")")
    def assets_by_site(self, site_id: str) -> list[Asset]:
        G = _get_graph()
        return [_node_to_asset(G, a) for a in gmod.get_assets_by_site(G, site_id)]

    @strawberry.field(description="Assets with no parent AND no connectivity edges at all "
                                    "-- fully isolated / disconnected.")
    def isolated_assets(self) -> list[Asset]:
        G = _get_graph()
        return [_node_to_asset(G, a) for a in gmod.get_isolated_assets(G)]

    @strawberry.field(description="Full failure-propagation analysis for a critical asset: "
                                    "downstream assets affected, operational impact, and "
                                    "recommended mitigation.")
    def failure_impact(self, asset_id: str) -> FailureImpact | None:
        G = _get_graph()
        if asset_id not in G:
            return None
        result = gmod.failure_impact(G, asset_id, _metadata)
        return FailureImpact(
            asset_id=result["asset_id"], asset_type=result["asset_type"],
            building_id=result["building_id"], downstream_count=result["downstream_count"],
            downstream_assets=result["downstream_assets"],
            operational_impact=result["operational_impact"], mitigation=result["mitigation"],
        )


schema = strawberry.Schema(query=Query)
graphql_router = GraphQLRouter(schema)
