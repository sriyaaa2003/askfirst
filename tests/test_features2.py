"""v2 features: same causality rule as v1 (demand of week w or later never reaches the features of week w)."""
import numpy as np

from askfirst.data import WeeklyData
from askfirst.features import build
from askfirst.features2 import EXTRA, build_extra


def _toy(n=6, weeks=80, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.poisson(5, (n, weeks)).astype(np.float32)
    P = rng.uniform(2, 4, (n, weeks)).astype(np.float32)
    P[0, :10] = np.nan
    ids = np.array([f"FOODS_1_00{i // 3}_CA_{i % 3 + 1}_validation" for i in range(n)])
    return WeeklyData(
        D=D, P=P, dept=np.zeros(n, np.int16), cat=np.zeros(n, np.int16), store=np.array([i % 3 for i in range(n)], np.int16),
        state=np.zeros(n, np.int16), dept_names=["D"], cat_names=["C"], events=np.zeros(weeks), snap=np.zeros((1, weeks)),
        month=np.ones(weeks, int), weekofyear=np.arange(weeks) % 52 + 1, series_ids=ids)


def _feats(d):
    rows = build(d)
    rows.update(build_extra(d, rows))
    return rows


def test_future_demand_does_not_change_extra_features():
    a = _feats(_toy())
    alt = _toy()
    alt.D[:, 70:] += 100
    b = _feats(alt)
    sa, sb = a["w"] <= 70, b["w"] <= 70
    assert np.allclose(a["X2"][sa], b["X2"][sb], equal_nan=True)


def test_extra_columns_present_and_finite_enough():
    r = _feats(_toy())
    assert r["X2"].shape[1] == r["X"].shape[1] + len(EXTRA)
    assert np.isfinite(r["X2"][:, r["X"].shape[1]:]).mean() > 0.95
