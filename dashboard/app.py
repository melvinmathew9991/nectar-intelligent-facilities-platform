"""
Nectar Intelligent Facilities Platform -- Operations Dashboard (Bonus A)
=========================================================================
Sections (per brief): site overview, asset health status, failure predictions,
energy trends, anomaly alerts, asset connectivity visualization.

Run:  streamlit run dashboard/app.py
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import config, preprocessing, features, graph as gmod

st.set_page_config(page_title="Nectar Facilities Platform", layout="wide")
BASE = os.path.dirname(__file__)


@st.cache_data
def load_all():
    raw = preprocessing.load_raw()
    anom_path = os.path.join(BASE, "anomalies.csv")
    anom = (preprocessing.read_csv_with_parquet_cache(anom_path, parse_dates=["timestamp"])
             if os.path.exists(anom_path) else None)
    return raw["telemetry"], raw["metadata"], raw["connectivity"], raw["weather"], anom


@st.cache_data
def load_live_failure_scores(telemetry: pd.DataFrame, metadata: pd.DataFrame):
    """Genuine live scoring (not just "model present"): builds the same 82
    engineered features used in training on each asset's most recent history,
    scores with the saved model, returns one row per rotating asset."""
    model_path = os.path.join(BASE, "..", "models", "predictive_maintenance.pkl")
    if not os.path.exists(model_path):
        return None
    bundle = joblib.load(model_path)
    model, scaler, feats, thr = bundle["model"], bundle["scaler"], bundle["features"], bundle["threshold"]

    # Only the latest row per asset is ever used below, and every rolling/lag
    # feature in build_maintenance_features looks strictly backward from a
    # fixed number of rows (max window = 24h = 144 rows at 10-min resolution)
    # -- so feeding it 90 days vs. a trailing 36h window (1.5x that, a safe
    # margin around imputation edge effects) produces byte-identical output
    # for the row we actually keep, at a fraction of the compute. Verified
    # empirically in tests/test_features.py::test_trailing_window_matches_full_history.
    cutoff = telemetry["timestamp"].max() - pd.Timedelta(hours=36)
    recent = telemetry[telemetry["timestamp"] >= cutoff]

    df, _ = features.build_maintenance_features(recent, metadata, need_target=False)
    latest = df.sort_values("timestamp").groupby("asset_id").tail(1).copy()
    X = latest.reindex(columns=feats, fill_value=0.0).values.astype(np.float32)
    if scaler is not None:
        X = scaler.transform(X)
    latest["failure_probability_24h"] = model.predict_proba(X)[:, 1]
    latest["will_fail"] = latest["failure_probability_24h"] >= thr
    return latest[["asset_id", "asset_type", "site_id", "building_id",
                    "failure_probability_24h", "will_fail"]].sort_values(
        "failure_probability_24h", ascending=False)


@st.cache_resource
def load_graph_cached(metadata: pd.DataFrame, connectivity: pd.DataFrame) -> nx.DiGraph:
    return gmod.build_graph(metadata, connectivity)


_t0 = time.time()
telemetry, metadata, connectivity, weather, anom = load_all()
_t_data = time.time() - _t0

_t0 = time.time()
G = load_graph_cached(metadata, connectivity)
_t_graph = time.time() - _t0

# ---------------- Sidebar ----------------
st.sidebar.title("Nectar Facilities")
st.sidebar.caption(f"Load times -- data: {_t_data:.2f}s | graph: {_t_graph:.2f}s "
                    "(near-0 on cache hit)")
site = st.sidebar.selectbox("Site", sorted(metadata.site_id.unique()))
site_tel = telemetry[telemetry.site_id == site]
site_meta = metadata[metadata.site_id == site]

st.title("Intelligent Facilities Platform -- Operations Dashboard")
st.caption(f"Site: **{site}** | {len(site_meta)} assets | "
           f"{site_tel.timestamp.min().date()} -> {site_tel.timestamp.max().date()}")

# ---------------- 1. Site overview ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Assets", len(site_meta))
c2.metric("Buildings", site_meta.building_id.nunique())
c3.metric("Total energy (kWh)", f"{site_tel.power_consumption.sum():,.0f}")
c4.metric("Fault events", int(site_tel.fault_flag.sum()))
st.divider()

# ---------------- 2. Asset health status ----------------
st.subheader("Asset health status")
if anom is not None:
    site_anom = anom[anom.asset_id.isin(site_meta.asset_id)]
    health = (site_anom.groupby("asset_id")["iso_anomaly"].mean() * 100).round(1)
    health = health.reset_index().rename(columns={"iso_anomaly": "anomaly_rate_%"})
    health = health.merge(site_meta[["asset_id", "asset_type"]], on="asset_id")

    def status(r):
        return "Critical" if r > 5 else ("Watch" if r > 2 else "Healthy")
    health["status"] = health["anomaly_rate_%"].apply(status)
    health = health.sort_values("anomaly_rate_%", ascending=False)
    st.dataframe(health, width="stretch", hide_index=True)
else:
    st.info("Run notebook 05 to precompute dashboard/anomalies.csv for health scoring.")
st.divider()

# ---------------- 3. Failure predictions (LIVE scoring) ----------------
st.subheader("Failure predictions (next 24h) -- live model scoring")
_t0 = time.time()
scores = load_live_failure_scores(telemetry, metadata)
_t_scores = time.time() - _t0
if scores is not None:
    site_scores = scores[scores.asset_id.isin(site_meta.asset_id)]
    at_risk = site_scores[site_scores.will_fail]
    st.caption(f"Live feature engineering + scoring took {_t_scores:.2f}s "
               "(near-0 on cache hit)")
    st.warning(f"{len(at_risk)} asset(s) currently above the maintenance-alert threshold"
               if len(at_risk) else "No assets currently above the maintenance-alert threshold.")
    st.dataframe(site_scores.style.format({"failure_probability_24h": "{:.1%}"}),
                 width="stretch", hide_index=True)
else:
    st.info("Train the model in notebook 03 to enable live failure scoring.")
st.divider()

# ---------------- 4. Energy trends (with weather overlay) ----------------
st.subheader("Energy trends")
bldg = st.selectbox("Building", sorted(site_tel.building_id.unique()))
b = site_tel[site_tel.building_id == bldg]
energy = b.groupby("timestamp")["power_consumption"].sum()
w = weather[weather.site_id == site].set_index("timestamp")["outdoor_temp"]

fig, ax1 = plt.subplots(figsize=(11, 3.2))
ax1.plot(energy.index, energy.values, color="#e67e22", lw=0.7, label="Energy (kWh)")
ax1.set_ylabel("kWh", color="#e67e22")
ax2 = ax1.twinx()
ax2.plot(w.index, w.values, color="#2980b9", lw=0.6, alpha=0.6, label="Outdoor temp")
ax2.set_ylabel("degC", color="#2980b9")
ax1.set_title(f"{bldg} -- hourly energy vs outdoor temperature")
st.pyplot(fig)

col1, col2 = st.columns(2)
with col1:
    hourly = b.groupby(b.timestamp.dt.hour)["power_consumption"].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.plot(hourly.index, hourly.values, marker="o", color="#d95f0e")
    ax.set(title="Avg by hour of day", xlabel="hour", ylabel="kWh")
    st.pyplot(fig)
with col2:
    dow = b.groupby(b.timestamp.dt.dayofweek)["power_consumption"].mean()
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar(["M", "T", "W", "T", "F", "S", "S"], dow.values, color="#31a354")
    ax.set(title="Avg by day of week", ylabel="kWh")
    st.pyplot(fig)
st.divider()

# ---------------- 5. Anomaly alerts ----------------
st.subheader("Anomaly alerts")
if anom is not None:
    recent = (anom[anom.asset_id.isin(site_meta.asset_id) & (anom.iso_anomaly == 1)]
              .sort_values("timestamp", ascending=False).head(15))
    st.dataframe(recent[["timestamp", "asset_id", "asset_type",
                          "vibration", "power_consumption"]],
                 width="stretch", hide_index=True)
else:
    st.info("No precomputed anomalies available.")
st.divider()

# ---------------- 6. Asset connectivity & failure impact ----------------
st.subheader("Asset connectivity & failure impact")
site_nodes = gmod.get_assets_by_site(G, site)
subG = G.subgraph(site_nodes)

sel = st.selectbox("Trace failure impact from asset", sorted(site_nodes))
downstream = gmod.get_downstream_impact(G, sel)
st.write(f"If **{sel}** fails, **{len(downstream)}** downstream assets are impacted: "
         f"{', '.join(downstream) if downstream else 'none (leaf asset)'}")

tc = {"Chiller": "#e74c3c", "AHU": "#3498db", "Pump": "#2ecc71",
      "EnvSensor": "#95a5a6", "EnergyMeter": "#9b59b6"}
dset = set(downstream)
colors = ["#e74c3c" if n == sel else ("#f39c12" if n in dset
          else tc.get(G.nodes[n]["asset_type"], "#333")) for n in subG.nodes]
try:
    pos = nx.nx_agraph.graphviz_layout(subG, prog="dot")
except Exception:
    pos = nx.spring_layout(subG, k=0.9, seed=42)
fig, ax = plt.subplots(figsize=(12, 7))
nx.draw(subG, pos, ax=ax, node_color=colors, node_size=500, font_size=6,
        with_labels=True, arrows=True, edge_color="#999")
ax.set_title(f"{site} -- connectivity (red=selected, orange=downstream impact)")
st.pyplot(fig)

st.caption("Nectar Data Scientist Challenge -- dashboard demo.")
