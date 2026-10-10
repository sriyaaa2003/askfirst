"""askfirst v2: a learned review router. Trained on calibration weeks only (outcomes known), evaluated on the untouched test weeks.

Compares the existing scores (random, width, value, risk, evr, oracle) with a gradient-boosted model that predicts the realised benefit
of a review from the same inputs plus recent forecast errors of the item and its store-department.
"""
import sys, time, pickle, json
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from sklearn.ensemble import HistGradientBoostingRegressor
from askfirst.config import load_config
from askfirst.data import load_weekly
from askfirst.features import CATEGORICAL
from askfirst.forecast import conformal_quantile, scale_of
from askfirst.policy import auto_order, cost, human_order, monte_carlo_value, unit_costs
from askfirst.experiment import rank_in_week, OVER_KEY

MU = sys.argv[1] if len(sys.argv) > 1 else "base"          # which forecast in data/mus_v2.pkl
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/v2/router_base.json"
cfg = load_config("configs/default.toml")
rows = pickle.load(open("data/rows_v2.pkl", "rb"))
mus = pickle.load(open("data/mus_v2.pkl", "rb"))
data = load_weekly(cfg["data"]["raw_dir"])
sp, c = cfg["split"], cfg["costs"]
w = rows["w"]
tr, ca, te = w < sp["train_end"], (w >= sp["train_end"]) & (w < sp["calib_end"]), w >= sp["calib_end"]
y = rows["y"]
mu_all = mus[MU]
s_all = scale_of(rows["std13"])
z_all = (y - mu_all) / s_all
cat_over = np.array([c[OVER_KEY[n]] for n in data.cat_names])
dept_cat = np.array([data.cat[np.flatnonzero(data.dept == d)[0]] for d in range(len(data.dept_names))])
cu_all, co_all = unit_costs(rows["price"], cat_over[rows["cat"].astype(int)], c["margin"])
n_dept = len(data.dept_names)
tau_d = np.array([c["margin"] / (c["margin"] + cat_over[dept_cat[d]]) for d in range(n_dept)])
dept = rows["dept"].astype(int)
q_d = np.array([conformal_quantile(z_all[ca & (dept == d)], tau_d[d]) for d in range(n_dept)])
q_all = q_d[dept]
rho = cfg["human"]["rho_assumed"]

# ---- recent forecast errors (known at decision time: weeks before w) --------------------------------------
sid, wk = rows["sid"].astype(int), rows["w"].astype(int)
n_series, n_weeks = data.D.shape
Z = np.full((n_series, n_weeks), np.nan, np.float32)
Z[sid, wk] = z_all.astype(np.float32)
A = np.full((n_series, n_weeks), np.nan, np.float32)
A[sid, wk] = (np.abs(y - mu_all)).astype(np.float32)
def lagged(M, k):
    out = np.full(len(sid), np.nan, np.float32)
    ok = wk - k >= 0
    out[ok] = M[sid[ok], wk[ok] - k]
    return out
feat = {"z1": lagged(Z, 1), "z2": lagged(Z, 2), "z3": lagged(Z, 3), "az1": np.abs(lagged(Z, 1)), "az2": np.abs(lagged(Z, 2)), "az3": np.abs(lagged(Z, 3))}
feat["maz"] = np.nanmean(np.stack([feat["az1"], feat["az2"], feat["az3"]]), axis=0)
# store-department mean |z| last week (a shock that hits many items)
sd_key = rows["store"].astype(int) * 100 + dept
az1_series = pd.Series(feat["az1"])
sdm = az1_series.groupby([sd_key, wk]).transform("mean").to_numpy()
feat["sd_az1"] = sdm
base_cols = {"mu": mu_all, "s": s_all, "q": q_all, "cu": cu_all, "co": co_all, "price": rows["price"], "value": rows["price"] * mu_all,
             "width_rel": s_all / (mu_all + 1.0), "price_rel": rows["price_rel"], "events": rows["events"], "snap": rows["snap"],
             "age": rows["age"], "std13": rows["std13"], "zero13": rows["zero13"], "mean13": rows["mean13"], "mean4": rows["mean4"],
             "dept": dept, "store": rows["store"].astype(int), "month": rows["month"]}
mc_risk = np.zeros(len(y)); mc_evr = np.zeros(len(y))
rng = np.random.default_rng(11)
pools = [z_all[ca & (dept == d)] for d in range(n_dept)]
t0 = time.time()
for wkk in np.unique(w[ca | te]):
    for d in range(n_dept):
        idx = np.flatnonzero((w == wkk) & (dept == d))
        if len(idx) == 0: continue
        zd = pools[d][rng.integers(0, len(pools[d]), (len(idx), 200))]
        mc_risk[idx], mc_evr[idx] = monte_carlo_value(mu_all[idx], s_all[idx], q_all[idx], cu_all[idx], co_all[idx], zd, rho)
print("MC done", round(time.time() - t0), flush=True)
base_cols["risk"], base_cols["evr"] = mc_risk, mc_evr
F = pd.DataFrame({**base_cols, **feat})

S_a = auto_order(mu_all, s_all, q_all)
cost_auto = cost(S_a, y, cu_all, co_all)
benefit = cost_auto - cost(human_order(mu_all, s_all, q_all, y, rho), y, cu_all, co_all)

cols = list(F.columns)
cat_mask = [cn in ("dept", "store") for cn in cols]
t = ca & (wk >= sp["train_end"] + 4)          # skip the first calibration weeks (lag features missing)
results = {}
Xtr, Xte = F[t].to_numpy(np.float32), F[te].to_numpy(np.float32)
scale = (rows["price"] * (mu_all + 1.0))      # normaliser so that big-ticket items do not swamp the loss
variants = {
  "learn_raw": (benefit[t], 1.0),
  "learn_norm": (benefit[t] / scale[t], scale[te]),
}
week_t = wk[te]; n_week = pd.Series(week_t).map(pd.Series(week_t).value_counts()).to_numpy()
ben_t = benefit[te]
scores = {"random": rng.random(te.sum()), "width": base_cols["width_rel"][te], "value": base_cols["value"][te], "risk": mc_risk[te], "evr": mc_evr[te]}
for name, (target, back) in variants.items():
    m = HistGradientBoostingRegressor(loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                      l2_regularization=1.0, categorical_features=cat_mask, early_stopping=False, random_state=7)
    m.fit(Xtr, target)
    pred = m.predict(Xte)
    scores[name] = pred * back if np.isscalar(back) else pred * back
# online variant: refit every 8 test weeks on all rows whose outcome is already known (calibration weeks + earlier test weeks)
pred_on = np.zeros(te.sum()); te_idx = np.flatnonzero(te); wk_te = wk[te]
blocks = np.arange(wk_te.min(), wk_te.max() + 1, 8)
for b0 in blocks:
    sel = (wk_te >= b0) & (wk_te < b0 + 8)
    tr_mask = (t | (te & (wk < b0))) & (wk < b0)
    m = HistGradientBoostingRegressor(loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                                      l2_regularization=1.0, categorical_features=cat_mask, early_stopping=False, random_state=7)
    m.fit(F[tr_mask].to_numpy(np.float32), (benefit / scale)[tr_mask])
    pred_on[sel] = m.predict(F.iloc[te_idx[sel]].to_numpy(np.float32)) * scale[te][sel]
scores["learn_online"] = pred_on
scores["oracle"] = ben_t
def capture(score, f):
    r = rank_in_week(score, week_t)
    sel = r < np.ceil(f * n_week)
    return float(ben_t[sel].sum() / ben_t.sum())
out = {k: {f"{int(f*100)}%": round(capture(v, f), 4) for f in (0.02, 0.05, 0.10, 0.20, 0.30)} for k, v in scores.items()}
wape = float(np.abs(y[te] - mu_all[te]).sum() / y[te].sum())
print(MU, "WAPE", round(wape, 4))
for k, v in out.items(): print(k, v)
json.dump({"mu": MU, "wape": wape, "capture": out}, open(OUT, "w"), indent=1)
