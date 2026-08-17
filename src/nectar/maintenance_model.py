"""
Task 2 -- Predictive Maintenance. Time-based train/test split, class-imbalance
handling, model selection by PR-AUC (more honest than ROC-AUC alone under
this level of imbalance), SHAP explainability, and error analysis broken out
by asset_type. Called from notebooks/03_predictive_maintenance.ipynb and
scripts/run_pipeline.py -- exactly one copy of this logic.
"""
from __future__ import annotations

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from . import config
from .logging_config import get_logger

log = get_logger(__name__)

TRAIN_DAYS = 70   # of the 90-day window; remaining ~20 trailing days = test
MAX_TRAIN_ROWS = 300_000   # memory-budget cap (this machine has ~8GB RAM total;
                            # training 4 models incl. RandomForest on the full
                            # ~700k-row train set risks an OOM kill). A stratified
                            # subsample preserving the positive rate is a standard,
                            # defensible trade-off -- documented explicitly here and
                            # in the notebook rather than silently applied.
                            #
                            # Cost measured, not assumed (issue #6): training on the
                            # full 796,399-row window gains +0.0036 PR-AUC (0.7802 vs
                            # 0.7766), selects the same model, and yields identical
                            # recall, for ~5x the training time. Reproduce with
                            # scripts/experiment_full_train.py.


def time_split(df: pd.DataFrame, feats: list[str], train_days: int = TRAIN_DAYS,
                max_train_rows: int | None = MAX_TRAIN_ROWS, seed: int = config.SEED):
    """`max_train_rows=None` disables the memory-budget subsample and trains on the
    full training window -- the comparison issue #6 asks for. Needs materially more
    RAM than the ~8GB this was built on; see docs/build_log.md section 4."""
    cutoff = pd.Timestamp(config.DATE_START) + pd.Timedelta(days=train_days)
    train = df[df["timestamp"] <= cutoff]
    test = df[df["timestamp"] > cutoff]

    if max_train_rows is not None and len(train) > max_train_rows:
        frac = max_train_rows / len(train)
        parts = [g.sample(n=max(1, round(len(g) * frac)), random_state=seed)
                 for _, g in train.groupby("target_24h")]
        train = pd.concat(parts).sort_values("timestamp").reset_index(drop=True)
        log.info(f"Subsampled train set to {len(train):,} rows (memory budget) "
                 f"-- positive rate preserved at {train['target_24h'].mean()*100:.2f}%")

    Xtr = train[feats].values.astype(np.float32)
    ytr = train["target_24h"].values
    Xte = test[feats].values.astype(np.float32)
    yte = test["target_24h"].values
    log.info(f"Split at {cutoff.date()}: train={len(train):,} (pos {ytr.mean()*100:.2f}%), "
             f"test={len(test):,} (pos {yte.mean()*100:.2f}%)")
    return Xtr, ytr, Xte, yte, train, test


def train_models(Xtr, ytr, feats: list[str], only: str | None = None) -> dict:
    """`only="RandomForest"` fits just that candidate. Used by
    scripts/experiment_full_train.py to train one model at a time on the full
    training window -- holding all four fitted models at once is the specific
    thing that triggered the OOM kills in docs/build_log.md section 4."""
    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    models = {}

    def _want(name: str) -> bool:
        return only is None or only == name

    # Model sizes are deliberately modest (this machine has ~8GB RAM total;
    # a 300-tree depth-14 RandomForest on hundreds of thousands of rows
    # alongside 3 other models risks an OOM kill) -- a documented memory/
    # accuracy trade-off, not an oversight.
    if _want("LogisticRegression"):
        scaler = StandardScaler().fit(Xtr)
        logreg = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)
        logreg.fit(scaler.transform(Xtr).astype(np.float32), ytr)
        models["LogisticRegression"] = {"model": logreg, "scaler": scaler}

    if _want("RandomForest"):
        rf = RandomForestClassifier(n_estimators=150, max_depth=10, class_weight="balanced",
                                     n_jobs=2, random_state=config.SEED)
        rf.fit(Xtr, ytr)
        models["RandomForest"] = {"model": rf, "scaler": None}

    if _want("XGBoost"):
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(n_estimators=300, learning_rate=0.08, max_depth=6,
                                 subsample=0.9, colsample_bytree=0.9,
                                 scale_pos_weight=pos_weight, eval_metric="aucpr",
                                 random_state=config.SEED, n_jobs=2)
            xgb.fit(Xtr, ytr)
            models["XGBoost"] = {"model": xgb, "scaler": None}
        except ImportError:
            log.warning("xgboost not available, skipping")

    if _want("LightGBM"):
        try:
            from lightgbm import LGBMClassifier
            lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.08, max_depth=8,
                                   num_leaves=63, subsample=0.9, colsample_bytree=0.9,
                                   scale_pos_weight=pos_weight, random_state=config.SEED,
                                   n_jobs=2, verbosity=-1)
            lgbm.fit(Xtr, ytr)
            models["LightGBM"] = {"model": lgbm, "scaler": None}
        except ImportError:
            log.warning("lightgbm not available, skipping")

    return models


def _proba(entry: dict, X: np.ndarray) -> np.ndarray:
    Xt = entry["scaler"].transform(X) if entry["scaler"] is not None else X
    # LightGBM's sklearn wrapper always populates feature_names_in_ (auto-named
    # "Column_N" even when fit on a plain ndarray), so predicting from the same
    # ndarray type trips sklearn's fit/predict feature-name mismatch check --
    # a benign library quirk, not a real mismatch (fit and predict use the same
    # positional float32 array throughout, matching `feats` by index).
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return entry["model"].predict_proba(Xt)[:, 1]


def evaluate_all(models: dict, Xte, yte) -> pd.DataFrame:
    rows = []
    for name, entry in models.items():
        proba = _proba(entry, Xte)
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "precision@0.5": precision_score(yte, pred, zero_division=0),
            "recall@0.5": recall_score(yte, pred, zero_division=0),
            "f1@0.5": f1_score(yte, pred, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "pr_auc": average_precision_score(yte, proba),
        })
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)


def best_threshold(yte, proba) -> float:
    prec, rec, thr = precision_recall_curve(yte, proba)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1s[:-1])]) if len(thr) else 0.5


def error_analysis(test_df: pd.DataFrame, yte, proba, threshold: float) -> pd.DataFrame:
    pred = (proba >= threshold).astype(int)
    out = test_df.copy()
    out["y_true"], out["y_pred"], out["proba"] = yte, pred, proba
    out["outcome"] = np.select(
        [(out.y_true == 1) & (out.y_pred == 1), (out.y_true == 1) & (out.y_pred == 0),
         (out.y_true == 0) & (out.y_pred == 1), (out.y_true == 0) & (out.y_pred == 0)],
        ["TP", "FN", "FP", "TN"], default="TN")
    return out.groupby(["asset_type", "outcome"]).size().unstack(fill_value=0)


def save_artifact(model, scaler, feats: list[str], threshold: float, model_name: str,
                    path: str | None = None) -> str:
    path = path or os.path.join(config.MODELS_DIR, "predictive_maintenance.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": feats,
                 "threshold": threshold, "model_name": model_name}, path)
    log.info(f"Saved model artifact -> {path}")
    return path
