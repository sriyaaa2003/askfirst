"""Load the M5 files and aggregate to weekly demand per store-item series."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_DAYS = 1913


@dataclass
class WeeklyData:
    D: np.ndarray            # (series, weeks) units sold
    P: np.ndarray            # (series, weeks) price, NaN while the item is not on sale
    dept: np.ndarray         # (series,) int code
    cat: np.ndarray
    store: np.ndarray
    state: np.ndarray
    dept_names: list[str]
    cat_names: list[str]
    events: np.ndarray       # (weeks,) number of event days
    snap: np.ndarray         # (states, weeks) SNAP days
    month: np.ndarray
    weekofyear: np.ndarray
    series_ids: np.ndarray


def load_weekly(raw_dir: str | Path) -> WeeklyData:
    raw = Path(raw_dir)
    cal = pd.read_csv(raw / "calendar.csv")
    sales = pd.read_csv(raw / "sales_train_validation.csv")
    prices = pd.read_csv(raw / "sell_prices.csv")

    n_weeks = N_DAYS // 7
    days = [f"d_{i}" for i in range(1, n_weeks * 7 + 1)]
    D = sales[days].to_numpy(np.float32).reshape(len(sales), n_weeks, 7).sum(axis=2)

    cal = cal.iloc[: n_weeks * 7].reset_index(drop=True)
    blk = cal.index.to_numpy() // 7
    wm = cal.groupby(blk)["wm_yr_wk"].agg(["min", "max"])
    if not (wm["min"] == wm["max"]).all():
        raise ValueError("calendar weeks do not line up with wm_yr_wk blocks of 7 days")
    wm_week = wm["min"].to_numpy()

    has_event = cal["event_name_1"].notna().to_numpy() | cal["event_name_2"].notna().to_numpy()
    events = np.bincount(blk, weights=has_event.astype(float), minlength=n_weeks)
    states = sorted(sales["state_id"].unique())
    snap = np.stack([np.bincount(blk, weights=cal[f"snap_{s}"].to_numpy(float), minlength=n_weeks)
                     for s in states])
    dates = pd.to_datetime(cal["date"])
    mid = dates.iloc[3::7].reset_index(drop=True)
    month = mid.dt.month.to_numpy()
    weekofyear = mid.dt.isocalendar().week.to_numpy().astype(int)

    key = (sales["store_id"] + "|" + sales["item_id"]).to_numpy()
    sid_of = pd.Series(np.arange(len(key)), index=key)
    prices["w"] = prices["wm_yr_wk"].map({v: i for i, v in enumerate(wm_week)})
    prices = prices.dropna(subset=["w"])
    sid = sid_of.reindex((prices["store_id"] + "|" + prices["item_id"]).to_numpy()).to_numpy()
    P = np.full(D.shape, np.nan, np.float32)
    P[sid, prices["w"].to_numpy().astype(int)] = prices["sell_price"].to_numpy(np.float32)

    dept_c = sales["dept_id"].astype("category")
    cat_c = sales["cat_id"].astype("category")
    return WeeklyData(
        D=D, P=P,
        dept=dept_c.cat.codes.to_numpy().astype(np.int16),
        cat=cat_c.cat.codes.to_numpy().astype(np.int16),
        store=sales["store_id"].astype("category").cat.codes.to_numpy().astype(np.int16),
        state=sales["state_id"].astype("category").cat.codes.to_numpy().astype(np.int16),
        dept_names=list(dept_c.cat.categories), cat_names=list(cat_c.cat.categories),
        events=events, snap=snap, month=month, weekofyear=weekofyear,
        series_ids=sales["id"].to_numpy(),
    )
