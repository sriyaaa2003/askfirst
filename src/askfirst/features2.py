"""Extra features for the v2 forecaster. Same rule as features.py: for week w only weeks before w are used, plus the price,
events and SNAP days of week w itself (planned in advance).

Added signals:
  * longer and exponentially weighted history (mean8, mean26, ewm, max13, weeks since the last sale),
  * price context (change since last week, the item's lowest price so far, price relative to its own history),
  * cross-sectional demand: the same item in the other stores of the same state (the item key is the id up to and including the state,
    e.g. HOBBIES_1_001_CA), and the store-department as a whole, one and four weeks ago
    (these capture shocks that hit many items at once, which one series cannot see by itself).
"""
from __future__ import annotations

import warnings

import numpy as np

from .data import WeeklyData

EXTRA = [
    "lag5", "lag6", "lag12", "mean8", "mean26", "ewm", "max13", "since_sale", "trend", "price_chg", "price_min_rel", "price_hist_rel",
    "item_x_lag1", "item_x_mean4", "sd_lag1", "sd_mean4", "item_x_trend", "sd_trend", "price_next_chg",
]


def _group_sums(D: np.ndarray, key: np.ndarray, n_groups: int) -> np.ndarray:
    G = np.zeros((n_groups, D.shape[1]), np.float64)
    np.add.at(G, key, D.astype(np.float64))
    return G


def build_extra(data: WeeklyData, rows: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    D, P = data.D.astype(np.float64), data.P
    n, n_w = D.shape
    item_key = np.array([s.rsplit("_", 2)[0] for s in data.series_ids])      # e.g. HOBBIES_1_001_CA: the item within one state
    _, item_id = np.unique(item_key, return_inverse=True)
    sd_id = data.store.astype(int) * 100 + data.dept.astype(int)
    _, sd_id = np.unique(sd_id, return_inverse=True)
    G_item = _group_sums(D, item_id, item_id.max() + 1)
    G_sd = _group_sums(D, sd_id, sd_id.max() + 1)
    n_item = np.bincount(item_id).astype(float)
    n_sd = np.bincount(sd_id).astype(float)

    pad = np.zeros((n, 1))
    cs = np.concatenate([pad, np.cumsum(D, axis=1)], axis=1)
    # weeks since last sale and EWM, computed forward
    since = np.zeros((n, n_w))
    ewm = np.zeros((n, n_w))
    last = np.full(n, 99.0)
    e = D[:, 0].copy()
    for w in range(n_w):
        since[:, w] = last
        ewm[:, w] = e
        last = np.where(D[:, w] > 0, 0.0, np.minimum(last + 1, 99.0))
        e = 0.3 * D[:, w] + 0.7 * e
    Pmin = np.fmin.accumulate(np.where(np.isnan(P), np.inf, P), axis=1)
    sid, w = rows["sid"].astype(int), rows["w"].astype(int)
    out: dict[str, np.ndarray] = {}
    out["lag5"], out["lag6"], out["lag12"] = D[sid, w - 5], D[sid, w - 6], D[sid, w - 12]
    out["mean8"] = (cs[sid, w] - cs[sid, w - 8]) / 8
    out["mean26"] = (cs[sid, w] - cs[sid, w - 26]) / 26
    out["ewm"] = ewm[sid, w]
    out["since_sale"] = since[sid, w]
    mx = np.stack([D[:, k:n_w - 13 + k + 1] for k in range(13)], axis=0).max(axis=0)   # max over windows [t, t+13)
    out["max13"] = mx[sid, w - 13]
    out["trend"] = (cs[sid, w] - cs[sid, w - 4]) / 4 / (out["mean26"] + 0.1)
    price = P[sid, w]
    prev = P[sid, w - 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        csP = np.concatenate([pad, np.cumsum(np.nan_to_num(P.astype(np.float64)), axis=1)], axis=1)
        csN = np.concatenate([pad, np.cumsum(~np.isnan(P), axis=1)], axis=1)
        pm_hist = csP[sid, w] / np.maximum(csN[sid, w], 1)          # mean price over weeks before w only
        out["price_chg"] = np.where(np.isnan(prev), 1.0, price / prev)
        pmin = Pmin[sid, w - 1]
        out["price_min_rel"] = np.where(np.isinf(pmin), 1.0, price / pmin)
    out["price_hist_rel"] = np.where(pm_hist > 0, price / pm_hist, 1.0)
    nxt = np.where(w + 1 < n_w, P[sid, np.minimum(w + 1, n_w - 1)], np.nan)      # next week's price is known in advance too
    out["price_next_chg"] = np.where(np.isnan(nxt), 1.0, nxt / price)
    out["item_x_lag1"] = (G_item[item_id[sid], w - 1] - D[sid, w - 1]) / np.maximum(n_item[item_id[sid]] - 1, 1)
    out["item_x_mean4"] = ((G_item[item_id[sid], w - 1] + G_item[item_id[sid], w - 2] + G_item[item_id[sid], w - 3] + G_item[item_id[sid], w - 4])
                           - (cs[sid, w] - cs[sid, w - 4])) / np.maximum(n_item[item_id[sid]] - 1, 1) / 4
    out["sd_lag1"] = G_sd[sd_id[sid], w - 1] / n_sd[sd_id[sid]]
    out["sd_mean4"] = sum(G_sd[sd_id[sid], w - k] for k in (1, 2, 3, 4)) / 4 / n_sd[sd_id[sid]]
    sd_m26 = sum(G_sd[sd_id[sid], w - k] for k in range(1, 27)) / 26 / n_sd[sd_id[sid]]
    out["sd_trend"] = out["sd_mean4"] / (sd_m26 + 0.1)
    item_m26 = sum(G_item[item_id[sid], w - k] for k in range(1, 27)) / 26 / n_item[item_id[sid]]
    out["item_x_trend"] = out["item_x_mean4"] / (item_m26 + 0.1)
    out["X2"] = np.column_stack([rows["X"]] + [out[k].astype(np.float32) for k in EXTRA])
    return out
