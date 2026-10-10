"""Total test-period cost of acting alone (no reviews) with the old and the new forecast, same costs and conformal procedure."""
import sys, pickle, json
import numpy as np
sys.path.insert(0, "src")
from askfirst.config import load_config
from askfirst.data import load_weekly
from askfirst.forecast import conformal_quantile, scale_of
from askfirst.policy import auto_order, cost, unit_costs
from askfirst.experiment import OVER_KEY
cfg = load_config("configs/default.toml"); rows = pickle.load(open("data/rows_v2.pkl", "rb")); mus = pickle.load(open("data/mus_v2.pkl", "rb"))
data = load_weekly("data/raw"); c = cfg["costs"]; sp = cfg["split"]; w = rows["w"]
ca = (w >= sp["train_end"]) & (w < sp["calib_end"]); te = w >= sp["calib_end"]; y = rows["y"]
cat_over = np.array([c[OVER_KEY[n]] for n in data.cat_names]); dept_cat = np.array([data.cat[np.flatnonzero(data.dept == d)[0]] for d in range(len(data.dept_names))])
cu, co = unit_costs(rows["price"], cat_over[rows["cat"].astype(int)], c["margin"]); dept = rows["dept"].astype(int); s = scale_of(rows["std13"])
out = {}
for name in ("base", "feat", "feat_big", "ens_feat"):
    mu = mus[name]; z = (y - mu) / s
    q = np.array([conformal_quantile(z[ca & (dept == d)], c["margin"] / (c["margin"] + cat_over[dept_cat[d]])) for d in range(len(data.dept_names))])[dept]
    S = auto_order(mu, s, q); loss = float(cost(S, y, cu, co)[te].sum())
    out[name] = {"wape": float(np.abs(y[te] - mu[te]).sum() / y[te].sum()), "auto_cost": loss}
b = out["base"]["auto_cost"]
for k in out: out[k]["cost_vs_base_pct"] = 100 * (out[k]["auto_cost"] - b) / b
print(json.dumps(out, indent=1)); json.dump(out, open("results/v2/forecast_cost.json", "w"), indent=1)
