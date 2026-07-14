"""
One-command, headless reproduction of the full Nectar pipeline:
generate -> preprocess -> validate -> train Task 2 model -> run Task 3/4 ->
build Task 5 graph -> save every artifact the dashboard/API depend on.

Run:  python scripts/run_pipeline.py
(Does NOT require Jupyter -- the notebooks are the narrated, business-facing
version of the same logic, all imported from src/nectar/.)
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nectar import (config, data_generation, preprocessing, features,
                     maintenance_model as mm, forecasting, anomaly, graph as gmod)
from nectar.logging_config import get_logger

log = get_logger("run_pipeline")


def main():
    t0 = time.time()

    log.info("=== 1. Data generation ===")
    result = data_generation.generate_all(save=True)
    telemetry, metadata, connectivity, weather = (
        result["telemetry"], result["metadata"], result["connectivity"], result["weather"])

    log.info("=== 2. Preprocessing & validation ===")
    report = preprocessing.validate(telemetry, metadata, connectivity)
    log.info(f"Validation: {report}")
    clean = preprocessing.impute_sensors(telemetry)
    clean = preprocessing.attach_weather(clean, weather)
    preprocessing.save_processed(clean, "telemetry_clean")

    log.info("=== 3. Task 2 -- Predictive Maintenance ===")
    df, feats = features.build_maintenance_features(telemetry, metadata)
    Xtr, ytr, Xte, yte, train_df, test_df = mm.time_split(df, feats)
    models = mm.train_models(Xtr, ytr, feats)
    results = mm.evaluate_all(models, Xte, yte)
    log.info(f"Task 2 model comparison:\n{results.to_string(index=False)}")
    best_name = results.iloc[0]["model"]
    best_entry = models[best_name]
    best_proba = mm._proba(best_entry, Xte)
    thr = mm.best_threshold(yte, best_proba)
    mm.save_artifact(best_entry["model"], best_entry["scaler"], feats, thr, best_name)
    log.info(f"Task 2 best model: {best_name} (threshold={thr:.3f})")

    log.info("=== 4. Task 3 -- Energy Forecasting ===")
    hourly, ffeats = features.build_forecast_features(telemetry, weather)
    fc_metrics, _ = forecasting.run_per_building(hourly, ffeats)
    log.info(f"Task 3 portfolio: MAE={fc_metrics.MAE.mean():.2f} "
             f"RMSE={fc_metrics.RMSE.mean():.2f} MAPE={fc_metrics.MAPE.mean():.2f}%")

    log.info("=== 5. Task 4 -- Anomaly Detection ===")
    stat = anomaly.statistical_flags(telemetry)
    iso = anomaly.isolation_forest_flags(telemetry, metadata)
    cusum = anomaly.cusum_flags_all(telemetry)
    lift = anomaly.validate_against_faults(iso, "iso_anomaly")
    log.info(f"Anomaly rate near faults vs elsewhere: {lift.to_dict()}")

    iso_with_type = iso.merge(metadata[["asset_id", "asset_type"]], on="asset_id", how="left")
    combined = iso_with_type[["timestamp", "asset_id", "site_id", "building_id", "asset_type",
                               "temperature", "vibration", "power_consumption", "fault_flag",
                               "iso_anomaly", "iso_score"]].copy()
    combined = combined.merge(stat[["timestamp", "asset_id", "stat_anomaly"]],
                               on=["timestamp", "asset_id"], how="left")
    combined = combined.merge(cusum[["timestamp", "asset_id", "cusum_anomaly"]],
                               on=["timestamp", "asset_id"], how="left")
    dashboard_dir = os.path.join(config.ROOT_DIR, "dashboard")
    os.makedirs(dashboard_dir, exist_ok=True)
    combined.to_csv(os.path.join(dashboard_dir, "anomalies.csv"), index=False)
    log.info(f"Saved dashboard/anomalies.csv ({len(combined):,} rows)")

    log.info("=== 6. Task 5 -- Connectivity Analysis ===")
    G = gmod.build_graph(metadata, connectivity)
    dq = gmod.data_quality_report(metadata, connectivity, G)
    log.info(f"Data-quality audit: {dq}")
    gmod.save_graph(G)

    log.info(f"=== Pipeline complete in {time.time()-t0:.1f}s ===")
    log.info("Artifacts: models/predictive_maintenance.pkl, models/asset_graph.pkl, "
              "dashboard/anomalies.csv, data/raw/*.csv, data/processed/telemetry_clean.parquet")


if __name__ == "__main__":
    main()
