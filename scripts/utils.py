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


# ── Moment Pooling ───────────────────────────────────────────────────────────

class MomentPooling:
    """
    Dimensionality reduction via PCA followed by polynomial moment expansion.

    Following arXiv:2403.08854 — compress a high-dimensional feature vector to a
    low-dimensional latent space with PCA, then expand with polynomial features to
    capture cross-moment statistics (mean, variance, skewness proxies, covariances).

    Parameters
    ----------
    latent_dim : int
        Number of PCA components to retain before expansion.
    order : int
        Polynomial degree. order=2 gives means + variances + cross-products.
        order=3 adds cubic terms (skewness proxies).
    include_bias : bool
        Whether to include the constant bias term in the output.

    Usage
    -----
    mp = MomentPooling(latent_dim=8, order=2)
    X_mp = mp.fit_transform(X)          # fit PCA + poly on training data
    X_mp_new = mp.transform(X_new)      # apply to new data (no refit)
    """

    def __init__(self, latent_dim=4, order=3, include_bias=True):
        self.latent_dim = latent_dim
        self.order = order
        self.include_bias = include_bias
        self.pca = PCA(n_components=self.latent_dim)
        self.poly = PolynomialFeatures(
            degree=self.order,
            include_bias=self.include_bias,
            interaction_only=False,
        )
        self._fitted = False

    def _make_feature_names(self, n_features):
        names = []
        if self.include_bias:
            names.append("bias")
        for degree in range(1, self.order + 1):
            for comb in combinations_with_replacement(range(n_features), degree):
                names.append("*".join([f"z{i}" for i in comb]))
        return names

    def fit(self, X):
        """Fit PCA and polynomial transformer on X (does not return transformed data)."""
        Z = self.pca.fit_transform(X)
        self.poly.fit(Z)
        self._fitted = True
        return self

    def transform(self, X):
        """Apply fitted PCA + polynomial expansion to X."""
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_transform() before transform().")
        Z = self.pca.transform(X)
        Z_poly = self.poly.transform(Z)
        feature_names = self._make_feature_names(Z.shape[1])
        index = X.index if hasattr(X, 'index') else None
        return pd.DataFrame(Z_poly, index=index, columns=feature_names)

    def fit_transform(self, X):
        """Fit and transform in one step (equivalent to fit().transform())."""
        Z = self.pca.fit_transform(X)
        Z_poly = self.poly.fit_transform(Z)
        feature_names = self._make_feature_names(Z.shape[1])
        self._fitted = True
        index = X.index if hasattr(X, 'index') else None
        return pd.DataFrame(Z_poly, index=index, columns=feature_names)