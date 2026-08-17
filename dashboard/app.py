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
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from nectar import config, features, preprocessing
from nectar import graph as gmod

st.set_page_config(page_title="Nectar Facilities Platform", layout="wide")
BASE = os.path.dirname(__file__)
DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@st.cache_data
def load_all():
    raw = preprocessing.load_raw()
    anom_path = os.path.join(BASE, "anomalies.csv")
    if os.path.exists(anom_path):
        anom = preprocessing.read_csv_with_parquet_cache(anom_path, parse_dates=["timestamp"])
    else:
        # Hosted deployment: the 99MB anomalies.csv is gitignored, so fall back
        # to the committed demo slice (see scripts/build_demo_slice.py).
        anom = preprocessing.load_demo("anomalies")
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


# Startup runs at module import, before anything is drawn -- so an exception here
# renders a blank page with no visible cause (which is exactly what a first hosted
# deploy did). Surface it on the page instead of failing silently.
try:
    _t0 = time.time()
    telemetry, metadata, connectivity, weather, anom = load_all()
    _t_data = time.time() - _t0

    _t0 = time.time()
    G = load_graph_cached(metadata, connectivity)
    _t_graph = time.time() - _t0
except Exception as exc:                                      # noqa: BLE001
    st.title("Intelligent Facilities Platform")
    st.error("Startup failed while loading data or building the asset graph.")
    st.exception(exc)
    st.caption(
        f"demo_mode={preprocessing.demo_mode()} | "
        f"raw dir exists={os.path.isdir(config.DATA_RAW_DIR)} | "
        f"demo dir exists={os.path.isdir(config.DATA_DEMO_DIR)} | "
        f"demo files={sorted(os.listdir(config.DATA_DEMO_DIR)) if os.path.isdir(config.DATA_DEMO_DIR) else 'n/a'}")
    st.stop()

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

if preprocessing.demo_mode():
    st.info(
        f"**Hosted demo.** Running on a committed {config.DEMO_DAYS}-day slice of the "
        f"full 1.96M-row dataset, ending {config.DEMO_END} -- the 177MB telemetry CSV "
        "is too large for the repo and too heavy for this container. Model inference is "
        "genuinely live: the same trained RandomForest scores the same engineered "
        "features it was trained on. Clone the repo and run `scripts/run_pipeline.py` "
        "for the full 90 days.")

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
    # Label from the actual index, not a fixed 7-element list: the window may not
    # span a full week (the hosted demo slice covers 4 days), and single-letter
    # labels collide -- matplotlib treats bar() x-values as categories, so a
    # duplicated "T"/"S" silently draws Thu on top of Tue and Sun on top of Sat.
    dow = b.groupby(b.timestamp.dt.dayofweek)["power_consumption"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(5.5, 3))
    ax.bar([DOW_LABELS[i] for i in dow.index], dow.values, color="#31a354")
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

ALL_OPTION = "— All assets (full site map) —"
sel = st.selectbox("Trace failure impact from asset",
                    [ALL_OPTION] + sorted(site_nodes))

if sel == ALL_OPTION:
    # Overview mode: the real, full connectivity map for this site, before
    # narrowing to any one asset's trace. No node is "selected" here, so
    # every dot keeps its normal asset-type color -- black/amber only
    # appear once a specific asset is picked below.
    downstream = []
    focus_nodes = set(site_nodes)
    subG = G.subgraph(focus_nodes)
    st.write(f"Showing the full connectivity map for **{site}** -- "
             f"**{subG.number_of_nodes()}** assets, **{subG.number_of_edges()}** connections. "
             f"Pick a specific asset above to trace its failure impact.")
else:
    downstream = gmod.get_downstream_impact(G, sel)
    st.write(f"If **{sel}** fails, **{len(downstream)}** downstream assets are impacted: "
             f"{', '.join(downstream) if downstream else 'none (leaf asset)'}")
    # Draw only what's relevant to this trace -- the selected asset, everything
    # downstream of it, and its immediate parent(s) for context -- rather than
    # the entire site (~50 nodes) every time, which is what made the diagram
    # crowded regardless of layout/spacing tuning.
    focus_nodes = {sel} | set(downstream) | set(G.predecessors(sel))
    subG = G.subgraph(focus_nodes)

tc = {"Chiller": "#e74c3c", "AHU": "#3498db", "Pump": "#2ecc71",
      "EnvSensor": "#95a5a6", "EnergyMeter": "#9b59b6"}
SELECTED_COLOR = "#000000"  # distinct from every asset-type color above, incl. Chiller's red
dset = set(downstream)
colors = [SELECTED_COLOR if n == sel else ("#f39c12" if n in dset
          else tc.get(G.nodes[n]["asset_type"], "#333")) for n in subG.nodes]
try:
    pos = nx.nx_agraph.graphviz_layout(subG, prog="dot")
except Exception:
    # pygraphviz isn't installed in this environment, so this is the layout
    # actually used -- k scaled up from the library default (1/sqrt(n)) and
    # more iterations give real separation instead of a tightly packed clump.
    pos = nx.spring_layout(subG, k=3.5 / (len(subG.nodes) ** 0.5), iterations=200, seed=42)
# Figure scales with how many nodes are actually being shown now that the
# view is scoped to the trace (1-18 nodes typically) instead of the whole
# site (~50) -- a fixed large canvas would look sparse for a 1-3 node trace.
n_nodes = max(len(subG.nodes), 1)
fig_w = max(6.0, min(16.0, 2.5 + 0.7 * n_nodes))
fig_h = max(4.5, min(11.0, 2.0 + 0.5 * n_nodes))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
# Selected node drawn larger so it's the first thing the eye lands on, even
# before reading color.
node_sizes = [700 if n == sel else 450 for n in subG.nodes]
nx.draw_networkx_nodes(subG, pos, ax=ax, node_color=colors, node_size=node_sizes)
# Edges that carry the cascade (selected -> downstream, or downstream ->
# further downstream) are colored amber to match the affected nodes, so the
# failure visibly "flows" outward rather than just lighting up disconnected
# dots. The upstream (parent -> selected) edge stays neutral gray.
edge_colors, edge_widths = [], []
for u, v in subG.edges():
    if (u == sel or u in dset) and v in dset:
        edge_colors.append("#f39c12")
        edge_widths.append(1.8)
    else:
        edge_colors.append("#999")
        edge_widths.append(0.6)
nx.draw_networkx_edges(subG, pos, ax=ax, arrows=True, edge_color=edge_colors, width=edge_widths)
# Labels offset above each node with a white backing box -- always legible
# regardless of node color (in-node black text is invisible on black nodes).
# A thin leader line ties each label back to its own dot, so which name
# belongs to which node stays unambiguous even when two nodes sit close
# together (a fixed uniform offset alone isn't enough at this density).
xs = [x for x, _ in pos.values()]
ys = [y for _, y in pos.values()]
x_span = (max(xs) - min(xs)) or 1
y_span = (max(ys) - min(ys)) or 1
y_offset = 0.04 * y_span
label_pos = {n: (x, y + y_offset) for n, (x, y) in pos.items()}

# Push apart any two labels whose (estimated) text boxes still overlap after
# the uniform offset above -- a fixed offset alone isn't enough once two
# nodes/labels land close together. Leader lines below still point back to
# each node's true position, so a label can move freely without becoming
# ambiguous.
char_w, label_h = 0.011 * x_span, 0.035 * y_span
names = list(label_pos.keys())

def _box(name):
    lx, ly = label_pos[name]
    w = char_w * len(name)
    return lx - w / 2, lx + w / 2, ly - label_h / 2, ly + label_h / 2

for _ in range(80):
    moved = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            ax0, ax1, ay0, ay1 = _box(a)
            bx0, bx1, by0, by1 = _box(b)
            if ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1:
                moved = True
                axp, ayp = label_pos[a]
                bxp, byp = label_pos[b]
                dx, dy = bxp - axp, byp - ayp
                dist = (dx ** 2 + dy ** 2) ** 0.5 or 0.01
                ux, uy = dx / dist, dy / dist
                push = 0.02 * max(x_span, y_span)
                label_pos[a] = (axp - ux * push / 2, ayp - uy * push / 2)
                label_pos[b] = (bxp + ux * push / 2, byp + uy * push / 2)
    if not moved:
        break

for n, (x, y) in pos.items():
    lx, ly = label_pos[n]
    ax.plot([x, lx], [y, ly], color="#aaaaaa", lw=0.5, zorder=1)
nx.draw_networkx_labels(subG, label_pos, ax=ax, font_size=7, font_color="black",
                         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.5})
ax.axis("off")
title = (f"{site} -- full asset connectivity map" if sel == ALL_OPTION
         else f"{sel} -- failure impact trace (black=selected, amber=downstream impact)")
ax.set_title(title)

# On-graph legend -- so the color key is readable directly from the figure
# (e.g. a screenshot or recording) without relying on a separate caption.
legend_entries = list(tc.items())
if sel != ALL_OPTION:
    legend_entries += [("Selected (failed)", SELECTED_COLOR), ("Downstream impact", "#f39c12")]
legend_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                              markersize=9, label=label) for label, color in legend_entries]
ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0, -0.05),
          frameon=True, fontsize=8, ncol=len(legend_entries))

st.pyplot(fig)

st.divider()
st.caption(
    "**Intelligent Facilities Platform** — predictive maintenance, energy forecasting, "
    "anomaly detection and asset connectivity analysis over building sensor telemetry. "
    "Failure probabilities are scored live by a RandomForest (PR-AUC 0.777) on features "
    "engineered in-request, not precomputed. Data is synthetic and generated by this "
    "project. "
    "[Source and methodology on GitHub]"
    "(https://github.com/melvinmathew9991/nectar-intelligent-facilities-platform)")
