"""Newsvendor ordering, a simulated planner, and the review-routing policies.

One decision = how many units of one item to order for one store for one week
(single period, no carry-over: a deliberate simplification, see the README).
"""
from __future__ import annotations

import numpy as np

# --- costs ----------------------------------------------------------------------


def unit_costs(price: np.ndarray, over_share: np.ndarray, margin: float):
    """Underage cost (lost margin per unit short) and overage cost (per unit left over)."""
    return margin * price, over_share * price


def cost(S, y, cu, co):
    """Newsvendor cost of ordering S when demand turns out to be y."""
    return cu * np.maximum(y - S, 0.0) + co * np.maximum(S - y, 0.0)


# --- ordering -------------------------------------------------------------------


def auto_order(mu, s, q):
    """Autonomous order: forecast plus the conformal quantile at the critical ratio."""
    return np.maximum(mu + s * q, 0.0)


def human_order(mu, s, q, y, rho):
    """Simulated planner. Removes a share rho of the model error and of the safety stock.

    rho = 0 reproduces the autonomous order; rho = 1 orders the realised demand exactly.
    """
    mu_h = mu + rho * (y - mu)
    return np.maximum(mu_h + (1.0 - rho) * s * q, 0.0)


# --- value of review ------------------------------------------------------------


def monte_carlo_value(mu, s, q, cu, co, z_draws, rho):
    """Expected cost of acting alone (risk) and expected value of a review (EVR).

    z_draws: (n, K) normalised residuals drawn from the calibration set of each row's department.
    """
    rho = np.asarray(rho, float)
    if rho.ndim == 1:
        rho = rho[:, None]
    y = np.maximum(mu[:, None] + s[:, None] * z_draws, 0.0)
    S_a = auto_order(mu, s, q)[:, None]
    c_auto = cost(S_a, y, cu[:, None], co[:, None])
    S_h = human_order(mu[:, None], s[:, None], q[:, None], y, rho)
    c_hum = cost(S_h, y, cu[:, None], co[:, None])
    return c_auto.mean(axis=1), (c_auto - c_hum).mean(axis=1)


# --- routing --------------------------------------------------------------------


def top_fraction(score: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of the top ceil(fraction * n) rows by score (one planning week)."""
    n = len(score)
    k = int(np.ceil(fraction * n))
    mask = np.zeros(n, bool)
    if k > 0:
        mask[np.argsort(-score, kind="stable")[:k]] = True
    return mask
