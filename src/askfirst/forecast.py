"""Global gradient-boosted demand forecast and normalised split conformal quantiles."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import CATEGORICAL, FEATURES


def fit_forecaster(X: np.ndarray, y: np.ndarray, cfg: dict) -> HistGradientBoostingRegressor:
    rng = np.random.default_rng(cfg["seed"])
    if len(y) > cfg["train_rows"]:
        idx = rng.choice(len(y), cfg["train_rows"], replace=False)
        X, y = X[idx], y[idx]
    cat_mask = [f in CATEGORICAL for f in FEATURES]
    model = HistGradientBoostingRegressor(
        loss="poisson", max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
        max_leaf_nodes=cfg["max_leaf_nodes"], categorical_features=cat_mask,
        early_stopping=False, random_state=cfg["seed"],
    )
    return model.fit(X, y)


def scale_of(std13: np.ndarray) -> np.ndarray:
    """Local scale for normalised residuals, known before the week starts."""
    return 1.0 + std13


def conformal_quantile(z: np.ndarray, tau: float) -> float:
    """Finite-sample split conformal quantile of the signed normalised residuals."""
    n = len(z)
    k = int(np.ceil((n + 1) * tau))
    if k > n:
        return float(np.max(z))
    return float(np.sort(z)[max(k, 1) - 1])
