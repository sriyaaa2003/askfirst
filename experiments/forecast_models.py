"""askfirst v2: does a richer feature set and an ensemble improve the demand forecast? (test-week WAPE)"""
import sys, time, pickle, os
import numpy as np
sys.path.insert(0, "src")
from sklearn.ensemble import HistGradientBoostingRegressor
from askfirst.config import load_config
from askfirst.data import load_weekly
from askfirst import features as F
from askfirst.features import CATEGORICAL, FEATURES
from askfirst.features2 import build_extra, EXTRA

cfg = load_config("configs/default.toml")
CACHE = "data/rows_v2.pkl"
t0 = time.time()
if os.path.exists(CACHE):
    rows = pickle.load(open(CACHE, "rb"))
else:
    data = load_weekly(cfg["data"]["raw_dir"])
    rows = F.build(data, cfg["split"]["first_week"])
    rows.update(build_extra(data, rows))
    pickle.dump(rows, open(CACHE, "wb"), protocol=4)
print("rows", len(rows["y"]), round(time.time() - t0), flush=True)
sp = cfg["split"]; w = rows["w"]
tr, ca, te = w < sp["train_end"], (w >= sp["train_end"]) & (w < sp["calib_end"]), w >= sp["calib_end"]
y = rows["y"]
names2 = FEATURES + EXTRA
cat2 = [n in CATEGORICAL for n in names2]

def wape(mu, m=te): return float(np.abs(y[m] - mu[m]).sum() / y[m].sum())

def fit_hgb(X, mask, cat_mask, n_rows, **kw):
    rng = np.random.default_rng(7)
    idx = np.flatnonzero(mask)
    if len(idx) > n_rows: idx = rng.choice(idx, n_rows, replace=False)
    p = dict(loss="poisson", max_iter=300, learning_rate=0.08, max_leaf_nodes=63, categorical_features=cat_mask, early_stopping=False, random_state=7)
    p.update(kw)
    return HistGradientBoostingRegressor(**p).fit(X[idx], y[idx])

res, mus = {}, {}
Xa, Xb = rows["X"], rows["X2"]
m = fit_hgb(Xa, tr, [n in CATEGORICAL for n in FEATURES], 1_500_000); mus["base"] = np.maximum(m.predict(Xa), 0); print("base", wape(mus["base"]), round(time.time()-t0), flush=True)
m = fit_hgb(Xb, tr, cat2, 1_500_000); mus["feat"] = np.maximum(m.predict(Xb), 0); print("feat", wape(mus["feat"]), round(time.time()-t0), flush=True)
m = fit_hgb(Xb, tr, cat2, 3_000_000, max_iter=500, learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=50, l2_regularization=1.0)
mus["feat_big"] = np.maximum(m.predict(Xb), 0); print("feat_big", wape(mus["feat_big"]), round(time.time()-t0), flush=True)
m = fit_hgb(Xb, tr, cat2, 1_500_000, loss="gamma" if False else "poisson", max_iter=300, max_leaf_nodes=31, random_state=11)
mus["feat_s11"] = np.maximum(m.predict(Xb), 0); print("feat_s11", wape(mus["feat_s11"]), flush=True)
mus["ens_feat"] = (mus["feat"] + mus["feat_big"] + mus["feat_s11"]) / 3; print("ens_feat", wape(mus["ens_feat"]), flush=True)
pickle.dump(mus, open("data/mus_v2.pkl", "wb"))
print("done", round(time.time() - t0))
