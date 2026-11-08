"""Plan B r3, E6: city-departure learners with a purged chronology, years <= 1983.

usage: uv run python docs/plans/creative/analysis/planB_e6_departures.py
Target: y(t+h) - ref30(t) - mean over the 15 nearest cities of the same quantity, leads 4-7.
Training rows: origins 1930-1961 with target year <= TRAIN_TARGET_MAX (1965). Test: origins 1966-1976 (targets <= 1983).
MLP (exactly the plan's v2.7 spec): scaler and target scale fitted on the early-stopping fit partition only
(targets <= 1953); validation rows: origins 1954-1958 with targets <= 1965; at most 200 epochs, patience 10 on
validation MAE; the best epoch count is refit on all training rows with the same preprocessing; 5 seeds averaged.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planB_core as C  # noqa: E402

ROOT = HERE.parents[3]
CUTOFF, TRAIN_TARGET_MAX, FIT_TARGET_MAX = 1983, 1965, 1953
TRAIN_ORIGINS, VAL_ORIGINS, TEST_ORIGINS, FIT_ORIGINS = range(1930, 1962), range(1954, 1959), range(1966, 1977), range(1930, 1950)
MAX_EPOCHS, PATIENCE, SEEDS, CLIP = 200, 10, 5, 3.0
METRICS = ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"]

d = pd.read_csv(ROOT / "data" / "annual_valid.csv", dtype={"city_id": str})
d = d[d.year <= CUTOFF]
assert d.year.max() <= CUTOFF
coords = pd.read_csv(ROOT / "data" / "city_info.csv", dtype={"city_id": str}).set_index("city_id").sort_index()
cities = sorted(d.city_id.unique())
coords = coords.loc[cities]
Y0 = int(d.year.min())
YEARS = np.arange(Y0, CUTOFF + 1)
T = {m: d.pivot(index="city_id", columns="year", values=m).reindex(index=cities, columns=YEARS).to_numpy(float) for m in METRICS}
NB = C.neighbours(coords.lat, coords.lon)


def features(M, t):
    """Departure features at origin t (columns <= t): 15 yearly departures from the regional mean (NaN -> 0),
    their 5- and 15-year means, the city's OLS-30 slope minus its neighbours' mean slope, the 30-year sd."""
    r = C.ref30(M, Y0, t)
    hist = C.window(M, Y0, t - 14, t, t) - r[:, None]
    dep = hist - C.nanmean(hist[NB], 1, 4)
    w30 = C.window(M, Y0, t - 29, t, t)
    s, _ = C.ols_slope(w30, np.arange(t - 29, t + 1, dtype=float)[None, :])
    f = np.column_stack([np.nan_to_num(dep), np.nan_to_num(C.nanmean(dep[:, -5:], 1)), np.nan_to_num(C.nanmean(dep, 1)),
                         np.nan_to_num(s - C.nanmean(s[NB], 1)), np.nan_to_num(np.nanstd(w30, 1))])
    return f, r, np.isnan(dep).sum(1)


def build(M, origins, max_target):
    X, Y = [], []
    for t in origins:
        f, r, nmiss = features(M, t)
        for h in range(4, 8):
            if t + h > max_target:
                continue
            assert t + h <= max_target <= CUTOFF
            y = M[:, t + h - Y0] - r
            tgt = y - C.nanmean(y[NB], 1, 4)
            keep = ~np.isnan(tgt) & ~np.isnan(r) & (nmiss <= 5)
            X.append(np.column_stack([f, np.full(len(r), h)])[keep]); Y.append(tgt[keep])
    return np.vstack(X), np.concatenate(Y)


def robust_scale(y):
    return 1.4826 * float(np.median(np.abs(y - np.median(y))))


def mlp_ensemble(Xf, yf, Xv, yv, Xall, yall, Xte):
    sc = StandardScaler().fit(Xf)            # preprocessing from the fit partition only
    ys = robust_scale(yf)
    z = lambda v: np.clip(v / ys, -CLIP, CLIP)  # noqa: E731
    preds, epochs = [], []
    with threadpool_limits(1):
        for seed in range(42, 42 + SEEDS):     # the declared deployment seeds 42-46
            new = lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-2, learning_rate_init=1e-3, random_state=seed)  # noqa: E731
            m = new(); best, best_ep, bad = np.inf, 1, 0
            for ep in range(1, MAX_EPOCHS + 1):
                m.partial_fit(sc.transform(Xf), z(yf))
                v = float(np.abs(m.predict(sc.transform(Xv)) * ys - yv).mean())
                if v < best - 1e-9:
                    best, best_ep, bad = v, ep, 0
                else:
                    bad += 1
                    if bad >= PATIENCE:
                        break
            final = new()
            for _ in range(best_ep):
                final.partial_fit(sc.transform(Xall), z(yall))
            preds.append(final.predict(sc.transform(Xte)) * ys)
            epochs.append(best_ep)
    return np.mean(preds, 0), epochs


print(f"E6: train targets <= {TRAIN_TARGET_MAX}, test origins {TEST_ORIGINS.start}-{TEST_ORIGINS.stop - 1} (targets <= {CUTOFF})")
for m in METRICS:
    M = T[m]
    Xtr, ytr = build(M, TRAIN_ORIGINS, TRAIN_TARGET_MAX)
    Xte, yte = build(M, TEST_ORIGINS, CUTOFF)
    Xf, yf = build(M, FIT_ORIGINS, FIT_TARGET_MAX)
    Xv, yv = build(M, VAL_ORIGINS, TRAIN_TARGET_MAX)
    base = float(np.abs(yte).mean())
    sc = StandardScaler().fit(Xtr)
    ys = robust_scale(ytr)
    out = {}
    hub = HuberRegressor(alpha=1.0, max_iter=500).fit(sc.transform(Xtr), ytr / ys)
    out["huber"] = np.abs(hub.predict(sc.transform(Xte)) * ys - yte).mean()
    gbm = HistGradientBoostingRegressor(loss="absolute_error", max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=500,
                                        l2_regularization=5.0, early_stopping=False, random_state=0).fit(Xtr, ytr)
    out["gbm"] = np.abs(gbm.predict(Xte) - yte).mean()
    pm, eps = mlp_ensemble(Xf, yf, Xv, yv, Xtr, ytr, Xte)
    out["mlp_ens5"] = np.abs(pm - yte).mean()
    out["half_mean15_dep"] = np.abs(0.5 * Xte[:, 16] - yte).mean()
    print(f"  {m:18s} n_train={len(ytr)} n_test={len(yte)} MAE(0)={base:.4f} | "
          + "  ".join(f"{k}={1 - v / base:+.6f}" for k, v in out.items()) + f" | best epochs {eps}")
