"""Plan B r3: design tables E1, E2, E3 and E5, from data/annual_valid.csv and data/city_info.csv, years <= 1983.

usage: uv run python docs/plans/creative/analysis/planB_design_tables.py
Every pseudo-origin t uses columns <= t; every scored target year t+h is asserted <= CUTOFF.
The state-space rows use the plan's estimator (planB_core: anchored bounded MLE + fixed-gain windowed level).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import planB_core as C  # noqa: E402

ROOT = HERE.parents[3]
CUTOFF = 1983
ORIGINS = range(1915, 1977)
ERAS = (("1915-1945", range(1915, 1946)), ("1946-1976", range(1946, 1977)))
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
NBS = np.column_stack([np.arange(len(cities)), NB])


def target(M, y):
    assert y <= CUTOFF, f"target year {y} after cutoff"
    return M[:, y - Y0]


class Scorer:
    """Pooled MAE of candidates against ref30 over (origin, city, lead 4-7)."""

    def __init__(self):
        self.e = {}

    def add(self, name, t, err):
        self.e.setdefault(name, {}).setdefault(t, []).append(err)

    def table(self, names):
        rows = []
        for name in names:
            row = {"method": name}
            for era, rg in ERAS + (("all", ORIGINS),):
                a = sum(np.concatenate(self.e[name][t]).sum() for t in rg)
                b = sum(np.concatenate(self.e["ref30"][t]).sum() for t in rg)
                wins = sum(np.concatenate(self.e[name][t]).sum() < np.concatenate(self.e["ref30"][t]).sum() for t in rg)
                row[era] = f"{1 - a / b:+.6f}"
                row[f"{era} wins"] = f"{wins}/{len(rg)}"
            rows.append(row)
        return pd.DataFrame(rows).set_index("method")


def e1_e2():
    print("E1. variance shares, 1951-1983, anomaly vs city mean 1951-1980")
    for m in METRICS:
        M = T[m]
        base = C.ref30(M, Y0, 1980)
        an = M[:, 1951 - Y0:1983 - Y0 + 1] - base[:, None]
        nat = C.nanmean(an, 0)[None, :]
        regx = C.nanmean(an[NB], 1, 4)
        ok = ~np.isnan(an) & ~np.isnan(regx)
        v = np.var(an[ok])
        beta = np.sum(an[ok] * regx[ok]) / np.sum(regx[ok] ** 2)
        print(f"  {m:18s} sd={math.sqrt(v):8.3f} national={1 - np.var((an - nat)[ok]) / v:.2f} "
              f"regional(excl self)={1 - np.var((an - beta * regx)[ok]) / v:.2f}")
    print("E2. lag correlations 1, 3, 7 (1931-1983), anomaly vs trailing 30-yr mean ending y-1")
    for m in METRICS:
        M = T[m]
        an = np.full_like(M, np.nan)
        for y in range(1931, CUTOFF + 1):
            an[:, y - Y0] = M[:, y - Y0] - C.ref30(M, Y0, y - 1)
        sub = an[:, 1931 - Y0:]
        dep = sub - C.nanmean(sub[NB], 1, 4)
        nat = C.nanmean(sub, 0)
        out = []
        for X in (sub, dep, nat[None, :]):
            r = []
            for k in (1, 3, 7):
                x, y = X[:, :-k].ravel(), X[:, k:].ravel()
                ok = ~np.isnan(x) & ~np.isnan(y)
                r.append(np.corrcoef(x[ok], y[ok])[0, 1])
            out.append(" ".join(f"{v:+.2f}" for v in r))
        print(f"  {m:18s} anomaly {out[0]} | departure {out[1]} | national {out[2]}")


def llt_r1(c, logpar):
    """r1's local linear trend (2 states, diffuse 1e4/1e2 start, first two terms dropped). Kept only for its E3 row."""
    r, ql, qs = np.exp(logpar)
    n, TT = c.shape
    lv = np.zeros(n); sl = np.zeros(n); P11 = np.full(n, 1e4); P12 = np.zeros(n); P22 = np.full(n, 1e2)
    ll = 0.0; started = np.zeros(n, bool); nobs = np.zeros(n)
    for k in range(TT):
        if k > 0:
            lv = lv + sl
            P11, P12, P22 = P11 + 2 * P12 + P22 + ql, P12 + P22, P22 + qs
        y = c[:, k]; ok = ~np.isnan(y); fresh = ok & ~started
        lv = np.where(fresh, y, lv); P11 = np.where(fresh, 1e4, P11); P12 = np.where(fresh, 0, P12); P22 = np.where(fresh, 1e2, P22)
        started |= ok
        v = np.where(ok, y - lv, 0.0); F = P11 + r; K1 = P11 / F; K2 = P12 / F
        nobs = nobs + ok; use = ok & (nobs > 2)
        ll += np.sum(np.where(use, -0.5 * (np.log(2 * np.pi * F) + v * v / F), 0.0))
        lv = lv + np.where(ok, K1 * v, 0); sl = sl + np.where(ok, K2 * v, 0)
        P11, P12, P22 = (np.where(ok, P11 - K1 * P11, P11), np.where(ok, P12 - K1 * P12, P12), np.where(ok, P22 - K2 * P12, P22))
    return ll, lv, sl


def hl(w):
    out = np.full(len(w), np.nan)
    for i, row in enumerate(w):
        x = row[~np.isnan(row)]
        if len(x) >= 10:
            s = (x[:, None] + x[None, :]) / 2
            out[i] = np.median(s[np.triu_indices(len(x))])
    return out


def v03_cdd_warming_rate(t):
    """v0.3's CDD rate (placeholder_v0_5.warming_rates): temp rate x response, each a self+15 mean floored at 0.
    Response = per-city OLS slope of CDD on annual mean temperature over t-29..t (>= 15 paired years)."""
    temp = C.window(T["annual_avg_temp"], Y0, t - 29, t, t)
    cdd = C.window(T["cdd"], Y0, t - 29, t, t)
    rate = C.drift_slope(T["annual_avg_temp"], Y0, t, NB)
    ok = ~np.isnan(temp) & ~np.isnan(cdd)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(ok, temp, 0).sum(1) / n
        ym = np.where(ok, cdd, 0).sum(1) / n
        dx = np.where(ok, temp - xm[:, None], 0.0)
        dy = np.where(ok, cdd - ym[:, None], 0.0)
        resp = np.where(n >= 15, (dx * dy).sum(1) / (dx ** 2).sum(1), np.nan)
    resp = np.maximum(np.nan_to_num(C.nanmean(np.column_stack([resp, resp[NB]]), 1)), 0.0)
    return rate * resp


def e3_e5():
    cache = C.RhoCache()
    sc = {m: Scorer() for m in METRICS}
    for t in ORIGINS:
        assert t + 7 <= CUTOFF
        for m in METRICS:
            M = T[m]
            r = C.ref30(M, Y0, t)
            w30 = C.window(M, Y0, t - 29, t, t)
            w10 = C.window(M, Y0, t - 9, t, t)
            ok30 = (~np.isnan(w30)).sum(1) >= 10
            m10 = np.where((~np.isnan(w10)).sum(1) >= 5, C.nanmean(w10, 1), r)
            rho = cache.get(m, M, Y0, t, NBS, Y0)
            cand = {}
            if m in ("annual_avg_temp", "cdd"):
                s_city, xm = C.ols_slope(w30, np.arange(t - 29, t + 1, dtype=float)[None, :])
                s_reg = C.drift_slope(M, Y0, t, NB)
                if m == "cdd":
                    v03_cdd_rate = v03_cdd_warming_rate(t)
                W0, _ = C.core_W(M, Y0, t, rho, 0.0, m, NBS, NB)
                W1, _ = C.core_W(M, Y0, t, rho, 1.0, m, NBS, NB)
                kap = C.KAPPA[m]
                # r1's local linear trend, for its E3 row only
                start = max(Y0, t - 99)
                cfull = C.regional(M, Y0, t, start, NBS)
                good = (~np.isnan(cfull)).sum(1) >= 30
                cc = np.where(good[:, None], cfull, np.nan)
                sd = np.nanstd(cc)
                x0 = np.log(np.array([sd ** 2 * 0.8, sd ** 2 * 0.02, sd ** 2 * 1e-4]))
                res = minimize(lambda p: -llt_r1(cc[good], p)[0], x0, method="Nelder-Mead",
                               options={"maxiter": 300, "xatol": 1e-3, "fatol": 1e-3})
                _, lv, sl = llt_r1(cc, res.x)
                cw = C.window(M, Y0, t - 29, t, t)
                cbar_llt = C.nanmean(np.where(~np.isnan(cw), cfull[:, -30:], np.nan), 1)
            for h in range(4, 8):
                y = target(M, t + h)
                keep = ~np.isnan(y) & ~np.isnan(r)
                c = {"ref30": r, "mean10": m10}
                if m == "annual_avg_temp":
                    c["v0.3 W (OLS-30 floored, from window centre)"] = r + np.nan_to_num(s_reg * (t + h - xm))
                if m == "cdd":
                    # proxy: the regional CDD slope used directly (what r1-r3 labelled "v0.3 W")
                    c["direct regional CDD-slope proxy (not v0.3)"] = r + np.nan_to_num(s_reg * (t + h - xm))
                    # actual v0.3/v0.5 CDD warming: floored regional temperature rate x floored regional CDD-on-temperature response
                    c["v0.3 W actual (temp rate x CDD response)"] = r + np.nan_to_num(v03_cdd_rate * (t + h - xm))
                if m in ("annual_avg_temp", "cdd"):   # r4 had these under the CDD branch only (Sol, r4 review)
                    tc = np.where(np.isnan(s_city), r, C.nanmean(w30, 1) + s_city * (t + h - xm))
                    c["combo blend"] = 0.5 * m10 + 0.5 * np.where(np.isnan(tc), r, tc)
                    c["r1 local linear trend, slope floored"] = r + np.nan_to_num(lv + np.maximum(sl, 0) * h - cbar_llt)
                    c["core level, no drift, kappa 1"] = r + W0[:, h - 1] / kap
                    c["core level, no drift, kappa 0.5"] = r + 0.5 * W0[:, h - 1] / kap
                    c["core level + drift (phi 1), kappa 1"] = r + W1[:, h - 1] / kap
                    c["core level + drift (phi 1), kappa 0.5"] = r + 0.5 * W1[:, h - 1] / kap
                if m in ("total_precip_in", "hottest_day_temp"):
                    c["median30"] = np.where(ok30, np.nanmedian(np.where(ok30[:, None], w30, 0), 1), r)
                    c["median10"] = np.where((~np.isnan(w10)).sum(1) >= 5, np.nanmedian(np.where(((~np.isnan(w10)).sum(1) >= 5)[:, None], w10, 0), 1), r)
                    hlv = hl(w30)
                    c["Hodges-Lehmann 30"] = np.where(np.isnan(hlv), r, hlv)
                    c["half ref30 + half all-history mean (>=40)"] = r + C.precip_W(M, Y0, t)[:, h - 1]
                    Wl, _ = C.core_W(M, Y0, t, rho, 0.0, "hottest_day_temp", NBS, NB)  # level, kappa 0.5
                    c["core level, kappa 1"] = r + 2 * Wl[:, h - 1]
                    c["core level, kappa 0.5"] = r + Wl[:, h - 1]
                for k, v in c.items():
                    vv = np.where(np.isnan(v), r, v)
                    sc[m].add(k, t, np.abs(vv[keep] - y[keep]))
    pd.set_option("display.width", 250)
    for m in METRICS:
        names = [k for k in sc[m].e if k != "ref30"]
        print(f"\nE3/E5 {m}: skill vs ref30, pseudo-origins 1915-1976, leads 4-7 (targets <= {CUTOFF})")
        print(sc[m].table(names).to_string())
    print("\nrho by anchor (plan estimator):")
    for m in METRICS:
        vals = {a: v for (mm, a), v in sorted(cache.values.items()) if mm == m}
        print(f"  {m}: " + " ".join(f"{a}:{'none' if v is None else f'{v:.4f}'}" for a, v in vals.items()))


if __name__ == "__main__":
    e1_e2()
    e3_e5()
