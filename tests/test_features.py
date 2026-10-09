"""The ordering decision for week w must not see demand of week w or later."""
import numpy as np

from askfirst.data import WeeklyData
from askfirst.features import FEATURES, build


def _toy(n=4, weeks=70, seed=0):
    rng = np.random.default_rng(seed)
    D = rng.poisson(5, (n, weeks)).astype(np.float32)
    P = np.full((n, weeks), 3.0, np.float32)
    P[0, :10] = np.nan                               # item 0 launches in week 10
    return WeeklyData(
        D=D, P=P, dept=np.zeros(n, np.int16), cat=np.zeros(n, np.int16),
        store=np.arange(n, dtype=np.int16), state=np.zeros(n, np.int16),
        dept_names=["D"], cat_names=["C"], events=np.zeros(weeks), snap=np.zeros((1, weeks)),
        month=np.ones(weeks, int), weekofyear=np.arange(weeks) % 52 + 1,
        series_ids=np.array([f"s{i}" for i in range(n)]),
    )


def test_changing_demand_from_week_w_does_not_change_features_of_week_w():
    base = _toy()
    w_cut = 60
    out1 = build(base)
    altered = _toy()
    altered.D[:, w_cut:] += 100                        # change the future, including week w_cut
    out2 = build(altered)
    sel = out1["w"] <= w_cut
    assert np.array_equal(out1["X"][sel], out2["X"][out2["w"] <= w_cut])


def test_only_weeks_with_a_price_are_decisions():
    out = build(_toy())
    assert not ((out["sid"] == 0) & (out["w"] < 52)).any()
    assert len(out["y"]) == len(out["X"]) and out["X"].shape[1] == len(FEATURES)


def test_lag_features_are_the_demand_of_earlier_weeks():
    d = _toy()
    out = build(d)
    i = 0
    assert out["lag1"][i] == d.D[out["sid"][i], out["w"][i] - 1]
    assert out["lag52"][i] == d.D[out["sid"][i], out["w"][i] - 52]
