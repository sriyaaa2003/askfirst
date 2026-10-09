"""Run the whole study: forecast, calibrate, route reviews, trace the autonomy frontier.

    python -m askfirst.experiment --config configs/default.toml

Every number in the README comes from results/summary.json, written here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as F
from .config import load_config
from .data import load_weekly
from .forecast import conformal_quantile, fit_forecaster, scale_of
from .policy import auto_order, cost, human_order, monte_carlo_value, unit_costs

OVER_KEY = {"FOODS": "over_foods", "HOUSEHOLD": "over_household", "HOBBIES": "over_hobbies"}
SCORES = ["random", "width", "value", "risk", "evr", "evr_ctx", "oracle"]


def _log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def rank_in_week(score: np.ndarray, week: np.ndarray) -> np.ndarray:
    """0 = highest score within its planning week."""
    rank = np.empty(len(score), np.int64)
    for w in np.unique(week):
        idx = np.flatnonzero(week == w)
        order = np.argsort(-score[idx], kind="stable")
        r = np.empty(len(idx), np.int64)
        r[order] = np.arange(len(idx))
        rank[idx] = r
    return rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.toml")
    ap.add_argument("--out", default="results")
    ap.add_argument("--margin-scale", type=float, default=1.0)
    ap.add_argument("--over-scale", type=float, default=1.0)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_config(args.config)
    cfg["costs"]["margin"] *= args.margin_scale
    for k in ("over_foods", "over_household", "over_hobbies"):
        cfg["costs"][k] *= args.over_scale
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = load_weekly(cfg["data"]["raw_dir"])
    _log(f"loaded {data.D.shape[0]} series x {data.D.shape[1]} weeks", t0)
    rows = F.build(data, cfg["split"]["first_week"])
    _log(f"built features: {len(rows['y'])} eligible decisions", t0)

    sp = cfg["split"]
    w = rows["w"]
    tr = w < sp["train_end"]
    ca = (w >= sp["train_end"]) & (w < sp["calib_end"])
    te = w >= sp["calib_end"]
    model = fit_forecaster(rows["X"][tr], rows["y"][tr], cfg["model"])
    _log("trained forecaster", t0)
    mu = np.maximum(model.predict(rows["X"][ca | te]), 0.0)
    ca_in = ca[ca | te]
    te_in = te[ca | te]

    # ---- accuracy on the test weeks -------------------------------------------
    y_te = rows["y"][te]
    acc = {}
    acc["model"] = float(np.abs(y_te - mu[te_in]).sum() / y_te.sum())
    acc["mean4_naive"] = float(np.abs(y_te - rows["mean4"][te]).sum() / y_te.sum())
    acc["lag52_seasonal_naive"] = float(np.abs(y_te - rows["lag52"][te]).sum() / y_te.sum())
    acc["lag1_naive"] = float(np.abs(y_te - rows["lag1"][te]).sum() / y_te.sum())

    # ---- costs ------------------------------------------------------------------
    c = cfg["costs"]
    cat_over = np.array([c[OVER_KEY[n]] for n in data.cat_names])
    dept_cat = np.zeros(len(data.dept_names), int)
    for d in range(len(data.dept_names)):
        dept_cat[d] = data.cat[np.flatnonzero(data.dept == d)[0]]
    over_share_row = cat_over[rows["cat"].astype(int)]
    cu_all, co_all = unit_costs(rows["price"], over_share_row, c["margin"])

    # ---- conformal quantiles per department -------------------------------------
    s_all = np.empty(len(rows["y"]))
    s_all[:] = scale_of(rows["std13"])
    mu_full = np.zeros(len(rows["y"]))
    mu_full[ca | te] = mu
    z_all = (rows["y"] - mu_full) / s_all
    n_dept = len(data.dept_names)
    tau_d = np.array([c["margin"] / (c["margin"] + cat_over[dept_cat[d]]) for d in range(n_dept)])
    q_d = np.zeros(n_dept)
    q10_d = np.zeros(n_dept)
    q90_d = np.zeros(n_dept)
    pools = []
    for d in range(n_dept):
        zc = z_all[ca & (rows["dept"] == d)]
        pools.append(zc)
        q_d[d] = conformal_quantile(zc, tau_d[d])
        q10_d[d] = conformal_quantile(zc, 0.10)
        q90_d[d] = conformal_quantile(zc, 0.90)

    T = {k: (v[te] if isinstance(v, np.ndarray) else v) for k, v in rows.items()}
    mu_t, s_t = mu_full[te], s_all[te]
    dept_t = T["dept"].astype(int)
    week_t = T["w"]
    q_t = q_d[dept_t]
    cu_t, co_t = cu_all[te], co_all[te]
    y_t = T["y"]
    S_a = auto_order(mu_t, s_t, q_t)
    lo, hi = np.maximum(mu_t + s_t * q10_d[dept_t], 0), mu_t + s_t * q90_d[dept_t]

    cov80 = float(((y_t >= lo) & (y_t <= hi)).mean())
    cov_by_dept = {data.dept_names[d]: float(((y_t >= lo) & (y_t <= hi))[dept_t == d].mean())
                   for d in range(n_dept)}
    service = {data.dept_names[d]: {"target": float(tau_d[d]),
                                    "achieved": float((y_t <= S_a)[dept_t == d].mean())}
               for d in range(n_dept)}

    # ---- value of review by Monte Carlo -----------------------------------------
    K = cfg["routing"]["mc_draws"]
    rho_a = cfg["human"]["rho_assumed"]
    rng = np.random.default_rng(cfg["routing"]["seed"])
    ctx_cfg = cfg["human"]["context"]
    ctx = (T["price_rel"] < ctx_cfg["price_drop"]) | (T["events"] > 0)
    rho_ctx = np.where(ctx, ctx_cfg["rho_with_context"], ctx_cfg["rho_without_context"])
    risk = np.zeros(len(y_t))
    evr = np.zeros(len(y_t))
    evr_ctx = np.zeros(len(y_t))
    for wk in np.unique(week_t):
        for d in range(n_dept):
            idx = np.flatnonzero((week_t == wk) & (dept_t == d))
            if len(idx) == 0:
                continue
            zd = pools[d][rng.integers(0, len(pools[d]), (len(idx), K))]
            risk[idx], evr[idx] = monte_carlo_value(
                mu_t[idx], s_t[idx], q_t[idx], cu_t[idx], co_t[idx], zd, rho_a)
            _, evr_ctx[idx] = monte_carlo_value(
                mu_t[idx], s_t[idx], q_t[idx], cu_t[idx], co_t[idx], zd, rho_ctx[idx])
    _log("Monte Carlo value of review done", t0)

    # ---- realised outcomes ------------------------------------------------------
    cost_auto = cost(S_a, y_t, cu_t, co_t)
    loss_auto = float(cost_auto.sum())
    scenarios = {str(r): r for r in cfg["human"]["rho_true"]}
    scenarios["context"] = rho_ctx
    benefit = {}
    for key, rho in scenarios.items():
        S_h = human_order(mu_t, s_t, q_t, y_t, rho)
        benefit[key] = cost_auto - cost(S_h, y_t, cu_t, co_t)

    width = s_t * (q90_d[dept_t] - q10_d[dept_t]) / (mu_t + 1.0)
    value = T["price"] * mu_t
    base_scores = {"random": rng.random(len(y_t)), "width": width, "value": value,
                   "risk": risk, "evr": evr, "evr_ctx": evr_ctx}
    ranks = {k: rank_in_week(v, week_t) for k, v in base_scores.items()}
    n_week = pd.Series(week_t).map(pd.Series(week_t).value_counts()).to_numpy()

    fine = np.round(np.arange(0.01, 1.0001, 0.01), 2)
    review_cost = c["review_cost"]
    frontier = []
    for rho in scenarios:
        rk = dict(ranks)
        rk["oracle"] = rank_in_week(benefit[rho], week_t)
        loss_human = loss_auto - float(benefit[rho].sum())
        for name in SCORES:
            for f in np.concatenate([[0.0], fine]):
                k_w = np.ceil(f * n_week)
                m = rk[name] < k_w
                loss = loss_auto - float(benefit[rho][m].sum())
                frontier.append({
                    "rho_true": rho, "policy": name, "fraction": float(f),
                    "reviews": int(m.sum()), "loss_cost": loss,
                    "net_cost": loss + review_cost * int(m.sum()),
                    "savings_capture": (loss_auto - loss) / (loss_auto - loss_human),
                })
    fr = pd.DataFrame(frontier)
    fr.to_csv(out / "frontier.csv", index=False)
    _log("frontier computed", t0)

    # ---- headline numbers -------------------------------------------------------
    n_dec = len(y_t)
    n_weeks_test = int(len(np.unique(week_t)))
    headline = {}
    for rho in scenarios:
        sub = fr[fr["rho_true"] == rho]
        loss_human = loss_auto - float(benefit[rho].sum())
        per = {"loss_all_auto": loss_auto, "loss_all_human": loss_human,
               "net_all_human": loss_human + review_cost * n_dec,
               "savings_all_human_pct": 100 * (loss_auto - loss_human) / loss_auto}
        for name in SCORES:
            s = sub[sub["policy"] == name].sort_values("fraction")
            for target in (0.5, 0.8):
                hit = s[s["savings_capture"] >= target]
                per[f"{name}_fraction_for_{int(target * 100)}pct"] = (
                    float(hit["fraction"].iloc[0]) if len(hit) else None)
            best = s.loc[s["net_cost"].idxmin()]
            per[f"{name}_best_net_fraction"] = float(best["fraction"])
            per[f"{name}_best_net_cost"] = float(best["net_cost"])
            for f in (0.05, 0.10, 0.20):
                per[f"{name}_capture_at_{int(f * 100)}pct"] = float(
                    s[np.isclose(s["fraction"], f)]["savings_capture"].iloc[0])
        thr = evr > review_cost
        per["threshold_fraction_reviewed"] = float(thr.mean())
        per["threshold_net_cost"] = float(loss_auto - benefit[rho][thr].sum()
                                          + review_cost * thr.sum())
        per["net_all_auto"] = loss_auto
        headline[rho] = per

    # ---- review-cost sweep (assumed planner at rho_assumed, evaluated at the same rho) ----
    sweep = {}
    sub = fr[(fr["rho_true"] == str(rho_a)) & (fr["policy"] == "evr")].sort_values("fraction")
    for mult in (0.25, 0.5, 1, 2, 4, 8, 16):
        cr = review_cost * mult
        net = sub["loss_cost"].to_numpy() + cr * sub["reviews"].to_numpy()
        j = int(np.argmin(net))
        sweep[str(mult)] = {"review_cost": cr, "best_fraction": float(sub["fraction"].iloc[j]),
                            "net_cost": float(net[j]),
                            "net_saving_vs_all_auto_pct": float(100 * (loss_auto - net[j]) / loss_auto)}

    # ---- bootstrap over test weeks: capture of all-human savings at a 10% budget ----------
    wk_idx = (week_t - week_t.min()).astype(int)
    n_w = int(wk_idx.max()) + 1
    boot = {}
    brng = np.random.default_rng(cfg["routing"]["seed"] + 1)
    draws = brng.integers(0, n_w, (2000, n_w))
    ben = benefit[str(rho_a)]
    tot_w = np.bincount(wk_idx, weights=ben, minlength=n_w)
    per_policy = {}
    for name in ("evr", "risk", "value", "width", "random"):
        m = ranks[name] < np.ceil(0.10 * n_week)
        sel_w = np.bincount(wk_idx, weights=ben * m, minlength=n_w)
        per_policy[name] = sel_w
        cap = sel_w[draws].sum(axis=1) / tot_w[draws].sum(axis=1)
        boot[name] = {"capture_10pct": float(sel_w.sum() / tot_w.sum()),
                      "ci95": [float(np.percentile(cap, 2.5)), float(np.percentile(cap, 97.5))]}
    d = (per_policy["evr"][draws].sum(axis=1) - per_policy["value"][draws].sum(axis=1)) \
        / tot_w[draws].sum(axis=1)
    boot["evr_minus_value_capture_10pct"] = {
        "diff": float((per_policy["evr"].sum() - per_policy["value"].sum()) / tot_w.sum()),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    top_evr = ranks["evr"] < np.ceil(0.10 * n_week)
    cov_top = float(((y_t >= lo) & (y_t <= hi))[top_evr].mean())
    cov_rest = float(((y_t >= lo) & (y_t <= hi))[~top_evr].mean())
    workload = {f"{int(f * 100)}pct": {
        "reviews_per_week": float(np.ceil(f * n_dec / n_weeks_test)),
        "planner_hours_per_week": float(np.ceil(f * n_dec / n_weeks_test) / 60.0)}
        for f in (0.01, 0.05, 0.10, 0.20)}

    summary = {
        "data": {"series": int(data.D.shape[0]), "weeks": int(data.D.shape[1]),
                 "test_weeks": n_weeks_test, "test_decisions": int(n_dec),
                 "train_decisions": int(tr.sum()), "calibration_decisions": int(ca.sum()),
                 "zero_demand_share_test": float((y_t == 0).mean())},
        "wape_test": acc,
        "interval_80_coverage": {"overall": cov80, "by_dept": cov_by_dept,
                                 "top10pct_evr": cov_top, "rest": cov_rest},
        "order_up_to_service_level": service,
        "assumptions": cfg["costs"] | {"rho_assumed": rho_a, "context_planner": ctx_cfg},
        "context_rows_share": float(ctx.mean()),
        "spearman_evr_ctx_vs_risk": float(pd.Series(evr_ctx).corr(pd.Series(risk), method="spearman")),
        "headline": headline,
        "workload": workload,
        "review_cost_sweep": sweep,
        "bootstrap_over_test_weeks": boot,
        "spearman_evr_vs_risk": float(pd.Series(evr).corr(pd.Series(risk), method="spearman")),
        "seconds": time.time() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot(fr, out, cfg)
    _log("done", t0)


def _plot(fr: pd.DataFrame, out: Path, cfg: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"evr": ("#1f4e79", "-", "askfirst (expected value of review)"),
             "risk": ("#2a9d8f", "--", "expected cost of acting alone"),
             "width": ("#e9a23b", "-.", "widest interval first"),
             "value": ("#8d6e63", ":", "highest value at stake first"),
             "random": ("#999999", ":", "random"),
             "oracle": ("#c0392b", "-", "oracle (upper bound)")}
    rhos = [str(r) for r in cfg["human"]["rho_true"]] + ["context"]
    style = dict(style, evr_ctx=("#6a1b9a", "-", "askfirst, planner profile known"))
    fig, axes = plt.subplots(1, len(rhos), figsize=(3.9 * len(rhos), 4.6), sharey=True)
    for ax, rho in zip(np.atleast_1d(axes), rhos):
        for name, (col, ls, lab) in style.items():
            s = fr[(fr["rho_true"] == rho) & (fr["policy"] == name)].sort_values("fraction")
            ax.plot(100 * s["fraction"], 100 * s["savings_capture"], color=col, ls=ls, label=lab)
        ax.set_title("context-aware planner" if rho == "context"
                     else f"planner removes {int(float(rho) * 100)}% of the error", fontsize=9)
        ax.set_xlabel("share of decisions sent to a planner (%)")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("share of all-human savings captured (%)")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "autonomy_frontier.png", dpi=160)


if __name__ == "__main__":
    main()
