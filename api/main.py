"""
Nectar Predictive Maintenance API (Bonus B)
============================================
POST /predict_failure accepts RAW telemetry (an asset_id to score against its
recent history, or an explicit telemetry window) -- feature engineering runs
inside the endpoint via the exact same features.py module used in training
(train/serve parity), not a pre-engineered 82-feature dict.

Run:  uvicorn api.main:app --reload
Docs: http://localhost:8000/docs
"""
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import config, preprocessing, features, graph as gmod

app = FastAPI(title="Nectar Predictive Maintenance API", version="1.0")

MODEL_PATH = os.path.join(config.MODELS_DIR, "predictive_maintenance.pkl")
GRAPH_PATH = os.path.join(config.MODELS_DIR, "asset_graph.pkl")

_bundle = None
_raw = None
_G = None


def get_model():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def get_raw():
    global _raw
    if _raw is None:
        _raw = preprocessing.load_raw()
    return _raw


def get_graph():
    global _G
    if _G is None:
        raw = get_raw()
        _G = (gmod.load_graph(GRAPH_PATH) if os.path.exists(GRAPH_PATH)
              else gmod.build_graph(raw["metadata"], raw["connectivity"]))
    return _G


class TelemetryReading(BaseModel):
    timestamp: datetime
    temperature: float | None = None
    humidity: float | None = None
    pressure: float | None = None
    vibration: float | None = None
    power_consumption: float | None = None
    occupancy_count: float | None = None
    operating_mode: str = "Cooling"
    fault_flag: int = 0


class PredictRequest(BaseModel):
    """Either just `asset_id` (scores against that asset's most recent stored
    history) or `asset_id` + an explicit `telemetry_window` (a raw reading
    sequence, e.g. from a live stream) -- in production the window path is
    what a real deployment would use; the asset_id-only path is a convenience
    for this demo/dataset."""
    asset_id: str
    telemetry_window: list[TelemetryReading] | None = None


MIN_HISTORY_ROWS = 144 + 10   # >= the largest rolling window (24h @ 10min) + margin


@app.get("/")
def root():
    return {"service": "Nectar Predictive Maintenance API",
            "endpoints": ["/predict_failure", "/health",
                          "/assets/{asset_id}/downstream_impact",
                          "/assets/{asset_id}/upstream_dependencies", "/docs"]}


@app.get("/health")
def health():
    try:
        b = get_model()
        return {"status": "ok", "model_name": b.get("model_name"),
                "n_features": len(b["features"]), "threshold": b["threshold"]}
    except Exception as e:
        return {"status": "model_not_found", "detail": str(e)}


@app.post("/predict_failure")
def predict_failure(req: PredictRequest):
    bundle = get_model()
    model, scaler, feats, thr = (bundle["model"], bundle["scaler"],
                                  bundle["features"], bundle["threshold"])
    raw = get_raw()
    metadata = raw["metadata"]

    if req.asset_id not in metadata["asset_id"].values:
        raise HTTPException(404, f"Unknown asset_id: {req.asset_id}")
    asset_meta = metadata[metadata["asset_id"] == req.asset_id].iloc[0]
    if asset_meta["asset_type"] not in config.ROTATING_TYPES:
        raise HTTPException(400, f"{req.asset_id} is a {asset_meta['asset_type']} -- "
                                   f"predictive maintenance only applies to "
                                   f"{config.ROTATING_TYPES}")

    if req.telemetry_window:
        rows = [r.model_dump() for r in req.telemetry_window]
        df = pd.DataFrame(rows)
        df["asset_id"] = req.asset_id
        df["site_id"] = asset_meta["site_id"]
        df["building_id"] = asset_meta["building_id"]
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if len(df) < MIN_HISTORY_ROWS:
            raise HTTPException(400, f"telemetry_window needs >= {MIN_HISTORY_ROWS} rows "
                                       f"(24h of rolling-feature history at 10-min resolution); got {len(df)}")
    else:
        tel = raw["telemetry"]
        df = tel[tel["asset_id"] == req.asset_id].sort_values("timestamp").tail(500)
        if len(df) < MIN_HISTORY_ROWS:
            raise HTTPException(400, f"insufficient stored history for {req.asset_id}")

    feat_df, _ = features.build_maintenance_features(df, metadata, rotating_only=False)
    if feat_df.empty:
        raise HTTPException(400, "feature engineering produced no rows -- check input")
    latest = feat_df.sort_values("timestamp").iloc[[-1]]
    as_of = str(latest["timestamp"].iloc[0])
    # pd.get_dummies() only creates columns for categories PRESENT in this
    # asset's window (e.g. scoring a single Chiller never produces
    # type_AHU/type_Pump/mode_Heating columns) -- reindex to the full
    # trained feature list so train/serve parity holds; a missing one-hot
    # column correctly means "this category doesn't apply here" = 0.
    latest_feats = latest.reindex(columns=feats, fill_value=0.0)

    X = latest_feats.values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    proba = float(model.predict_proba(X)[0, 1])
    will_fail = bool(proba >= thr)

    return {
        "asset_id": req.asset_id,
        "asset_type": asset_meta["asset_type"],
        "as_of": as_of,
        "failure_probability_24h": round(proba, 4),
        "threshold": round(float(thr), 4),
        "will_fail": will_fail,
        "recommendation": ("Schedule maintenance within 24h" if will_fail
                           else "No action needed"),
    }


@app.get("/assets/{asset_id}/downstream_impact")
def downstream_impact(asset_id: str):
    G = get_graph()
    if asset_id not in G:
        raise HTTPException(404, f"Unknown asset_id: {asset_id}")
    downstream = gmod.get_downstream_impact(G, asset_id)
    return {"asset_id": asset_id, "downstream_count": len(downstream),
            "downstream_assets": downstream}


@app.get("/assets/{asset_id}/upstream_dependencies")
def upstream_dependencies(asset_id: str):
    G = get_graph()
    if asset_id not in G:
        raise HTTPException(404, f"Unknown asset_id: {asset_id}")
    return {"asset_id": asset_id, "upstream_assets": gmod.get_upstream_dependencies(G, asset_id)}
