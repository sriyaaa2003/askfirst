"""Features for the ordering decision made at the start of week w.

Everything used for week w comes from weeks strictly before w, except the price,
event days and SNAP days of week w itself, which are planned in advance.
"""
from __future__ import annotations

import warnings

import numpy as np

from .data import WeeklyData

FEATURES = [
    "lag1", "lag2", "lag3", "lag4", "lag8", "lag52", "mean4", "mean13", "std13", "zero13",
    "price", "price_rel", "age", "events", "snap", "month", "weekofyear",
    "dept", "cat", "store",
]
CATEGORICAL = ["dept", "cat", "store"]


def build(data: WeeklyData, w_min: int = 52) -> dict[str, np.ndarray]:
    D, P = data.D, data.P
    n, n_w = D.shape
    pad = np.zeros((n, 1), np.float64)
    cs = np.concatenate([pad, np.cumsum(D, axis=1, dtype=np.float64)], axis=1)
    cs2 = np.concatenate([pad, np.cumsum(D.astype(np.float64) ** 2, axis=1)], axis=1)
    csz = np.concatenate([pad, np.cumsum(D == 0, axis=1, dtype=np.float64)], axis=1)
    first = np.argmax(~np.isnan(P), axis=1)

    cols: dict[str, list[np.ndarray]] = {k: [] for k in FEATURES + ["sid", "w", "y"]}
    for w in range(w_min, n_w):
        sid = np.flatnonzero(~np.isnan(P[:, w]))
        m = len(sid)
        for k in (1, 2, 3, 4, 8, 52):
            cols[f"lag{k}"].append(D[sid, w - k])
        cols["mean4"].append((cs[sid, w] - cs[sid, w - 4]) / 4)
        mean13 = (cs[sid, w] - cs[sid, w - 13]) / 13
        cols["mean13"].append(mean13)
        var13 = (cs2[sid, w] - cs2[sid, w - 13]) / 13 - mean13**2
        cols["std13"].append(np.sqrt(np.maximum(var13, 0)))
        cols["zero13"].append((csz[sid, w] - csz[sid, w - 13]) / 13)
        price = P[sid, w]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pm = np.nanmean(P[sid, w - 8:w], axis=1)
        cols["price"].append(price)
        cols["price_rel"].append(np.where(np.isnan(pm), 1.0, price / pm))
        cols["age"].append(w - first[sid])
        cols["events"].append(np.full(m, data.events[w]))
        cols["snap"].append(data.snap[data.state[sid], w])
        cols["month"].append(np.full(m, data.month[w]))
        cols["weekofyear"].append(np.full(m, data.weekofyear[w]))
        cols["dept"].append(data.dept[sid])
        cols["cat"].append(data.cat[sid])
        cols["store"].append(data.store[sid])
        cols["sid"].append(sid)
        cols["w"].append(np.full(m, w))
        cols["y"].append(D[sid, w])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["X"] = np.column_stack([out[k].astype(np.float32) for k in FEATURES])
    return out
