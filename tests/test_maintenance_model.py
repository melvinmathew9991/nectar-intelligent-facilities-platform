"""Sanity-check maintenance_model.py: the time-based split (including the
memory-budget subsample path), that trained models actually beat the base
rate, threshold selection, per-asset-type error breakdown, and artifact
round-tripping."""
import os
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from nectar import maintenance_model as mm, config


def _synthetic_dataset(n_days=100, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(config.DATE_START, periods=n_days * 24, freq="1h")
    n = len(idx)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    target = (x1 + rng.normal(0, 0.3, n) > 1.0).astype(int)
    df = pd.DataFrame({"timestamp": idx, "x1": x1, "x2": x2, "target_24h": target,
                        "asset_type": "Chiller"})
    return df, ["x1", "x2"]


def test_time_split_respects_cutoff_and_no_overlap():
    df, feats = _synthetic_dataset(n_days=100)
    Xtr, ytr, Xte, yte, train, test = mm.time_split(df, feats, train_days=70)

    cutoff = pd.Timestamp(config.DATE_START) + pd.Timedelta(days=70)
    assert train["timestamp"].max() <= cutoff
    assert test["timestamp"].min() > cutoff
    assert len(train) + len(test) == len(df)
    assert Xtr.shape == (len(train), len(feats))
    assert Xte.shape == (len(test), len(feats))
    assert len(ytr) == len(train) and len(yte) == len(test)


def test_time_split_subsamples_when_over_row_cap_and_preserves_positive_rate():
    df, feats = _synthetic_dataset(n_days=100, seed=1)
    cutoff = pd.Timestamp(config.DATE_START) + pd.Timedelta(days=70)
    full_train_pos_rate = df.loc[df["timestamp"] <= cutoff, "target_24h"].mean()

    Xtr, ytr, Xte, yte, train, test = mm.time_split(df, feats, train_days=70, max_train_rows=100)

    assert len(train) <= 100
    assert abs(train["target_24h"].mean() - full_train_pos_rate) < 0.15


def test_train_models_and_evaluate_all_beat_base_rate():
    df, feats = _synthetic_dataset(n_days=100, seed=2)
    Xtr, ytr, Xte, yte, train, test = mm.time_split(df, feats, train_days=70)

    models = mm.train_models(Xtr, ytr, feats)
    assert "LogisticRegression" in models
    assert "RandomForest" in models

    metrics = mm.evaluate_all(models, Xte, yte)
    expected_cols = {"model", "precision@0.5", "recall@0.5", "f1@0.5", "roc_auc", "pr_auc"}
    assert expected_cols.issubset(metrics.columns)
    assert (metrics["pr_auc"].diff().dropna() <= 1e-9).all(), "must be sorted descending by pr_auc"

    base_rate = yte.mean()
    assert (metrics["pr_auc"] > base_rate).all(), \
        "every model should beat the random-guess PR-AUC baseline on separable synthetic data"


def test_best_threshold_finds_a_perfect_separator():
    yte = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = mm.best_threshold(yte, proba)
    pred = (proba >= thr).astype(int)
    assert (pred == yte).all()


def test_error_analysis_confusion_counts_by_asset_type():
    test_df = pd.DataFrame({"asset_type": ["Chiller", "Chiller", "AHU", "AHU"]})
    yte = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.9, 0.1, 0.1])

    out = mm.error_analysis(test_df, yte, proba, threshold=0.5)
    assert out.loc["Chiller", "TP"] == 1
    assert out.loc["Chiller", "FP"] == 1
    assert out.loc["AHU", "FN"] == 1
    assert out.loc["AHU", "TN"] == 1


def test_save_artifact_round_trips(tmp_path):
    df, feats = _synthetic_dataset(n_days=100, seed=3)
    Xtr, ytr, Xte, yte, train, test = mm.time_split(df, feats, train_days=70)
    models = mm.train_models(Xtr, ytr, feats)
    entry = models["LogisticRegression"]

    path = str(tmp_path / "model.pkl")
    out_path = mm.save_artifact(entry["model"], entry["scaler"], feats, 0.42,
                                 "LogisticRegression", path=path)

    assert out_path == path
    loaded = joblib.load(path)
    assert loaded["features"] == feats
    assert loaded["threshold"] == 0.42
    assert loaded["model_name"] == "LogisticRegression"
    assert loaded["scaler"] is not None
