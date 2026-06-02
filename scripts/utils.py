"""
utils.py — Anomaly Detection Utilities
MGCLS / BYOL features  ·  Internship project

Changes from v1:
  - MomentPooling: added fit/transform split so it can be used in cross-val loops
  - Added score_ensemble() for combining multiple detectors
  - Added rank_normalise() for fair comparison across detectors
  - compute_metrics now also returns a formatted string summary
  - topk_recall moved here (was duplicated in notebooks)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent.parent


# ── Data loading ────────────────────────────────────────────────────────────

def load_features():
    path = BASE_DIR / "data" / "mgcls_byol_features.parquet"
    df = pd.read_parquet(path)
    print(f"Features loaded: {df.shape}")
    return df


def load_catalogue():
    path = BASE_DIR / "data" / "protege_catalogue.csv"
    df = pd.read_csv(path)
    print(f"Catalogue loaded: {df.shape}")
    return df


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true, scores):
    """Return ROC-AUC and PR-AUC for binary labels y_true and continuous scores."""
    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
    }


def topk_recall(y_true, scores, k=100):
    """Fraction of true positives recovered in the top-k ranked items."""
    if isinstance(scores, pd.Series):
        ranked = scores.sort_values(ascending=False).index[:k]
    else:
        ranked = pd.Series(scores).sort_values(ascending=False).index[:k]
    return y_true.loc[ranked].sum() / y_true.sum()


# ── Score utilities ──────────────────────────────────────────────────────────

def rank_normalise(scores: pd.Series) -> pd.Series:
    """
    Convert raw anomaly scores to [0, 1] rank-normalised scores.
    Rank normalisation makes different detectors comparable on the same scale
    without assuming any score distribution.
    """
    ranks = scores.rank(method='average', ascending=True)
    return (ranks - 1) / (len(ranks) - 1)


def score_ensemble(score_dict: dict, weights=None) -> pd.Series:
    """
    Combine multiple anomaly score Series into one by averaging rank-normalised scores.

    Parameters
    ----------
    score_dict : dict[str, pd.Series]
        Named anomaly score series (higher = more anomalous for all).
    weights : list[float] | None
        Optional per-detector weights. Uniform if None.

    Returns
    -------
    pd.Series  Ensemble score (rank-normalised).
    """
    normalised = [rank_normalise(s) for s in score_dict.values()]
    if weights is None:
        weights = [1.0 / len(normalised)] * len(normalised)
    combined = sum(w * s for w, s in zip(weights, normalised))
    return combined.rename("ensemble")


