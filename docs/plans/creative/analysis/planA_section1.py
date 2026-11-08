"""Plan A (graph), section 1 design tables: correlations, relative noise, decomposition, in-era tables, kernel horse-race,
and the rejected alternatives (empirical-Bayes reliability weights, correlation-learned graph, break-aware level).

Run from the repository root:  uv run python docs/plans/creative/analysis/planA_section1.py   (about 2-4 minutes)

Data: `planA_walkforward.load` parses the whole data/annual_valid.csv and data/city_info.csv, then keeps only years <= 1982
(asserted). Every table here fits and scores on the same pre-1983 era: descriptive by design, with no forward cutoff.
The out-of-sample evidence is planA_walkforward.py. Plan: docs/plans/creative/plan-A-graph.md (section 1, appendix A).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planA_walkforward as wf  # noqa: E402

METRICS = wf.METRICS
ERAS = {"1915-35": range(1915, 1936), "1950-62": range(1950, 1963), "1963-75": range(1963, 1976)}
LEADS = (4, 5, 6, 7)
GRID = wf.GRID
GRID_WIDE = np.round(np.arange(0.0, 1.5 + 1e-9, 0.05), 2)


def nanmean(x, axis):
    return wf.nmean(x, axis)


def centre(v):
    defined = np.isfinite(v)
    return np.where(defined, v - np.nanmedian(np.where(defined, v, np.nan), axis=-1, keepdims=True), 0.0)


# ------------------------------------------------------------------------------ (a)-(d): r1/r2 tables
def table_correlation(tab, d, order):
    print("== (a) median distance to the k-th nearest city (km) ==")
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    print("   " + "  ".join(f"k={k}: {np.median(dn[np.arange(len(d)), order[:, k - 1]]):.0f}" for k in (1, 5, 10, 15, 30)))
    print("== (b) median pairwise correlation of annual anomalies by distance, 1941-1982 (>= 25 shared years) ==")
    bins = [0, 250, 500, 750, 1000, 1500, 2500, 6000]
    efold = {}
    iu = np.triu_indices(len(d), 1)
    for m in METRICS:
        x = tab[m][:, wf.col(1941): wf.col(1982) + 1]
        a = x - nanmean(x, 1)[:, None]
        ok = np.isfinite(a)
        az = np.where(ok, a, 0.0)
        n = ok.astype(float) @ ok.T.astype(float)
        s2 = (az ** 2) @ ok.T.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (az @ az.T) / np.sqrt(s2 * s2.T)
        dd, cc, nn = d[iu], c[iu], n[iu]
        keep = (nn >= 25) & np.isfinite(cc)
        row = [f"{lo}-{hi}: {np.median(cc[keep & (dd >= lo) & (dd < hi)]):.2f}" for lo, hi in zip(bins[:-1], bins[1:])]
        s = keep & (cc > 0.05) & (dd < 2000)
        efold[m] = -np.sum(dd[s] ** 2) / np.sum(dd[s] * np.log(cc[s]))
        print(f"   {m:17s} " + "  ".join(row) + f"  e-folding ~{efold[m]:.0f} km")
    return efold


def table_relative_noise(tab, order):
    print("== (c) relative anomaly R = anomaly - mean of 15 nearest, 1941-1982 ==")
    for m in METRICS:
        x = tab[m][:, wf.col(1941): wf.col(1982) + 1]
        a = x - nanmean(x, 1)[:, None]
        r = a - nanmean(a[order[:, :15]], 1)

        def lag(z, k):
            p, q = z[:, :-k].ravel(), z[:, k:].ravel()
            s = np.isfinite(p) & np.isfinite(q)
            return np.corrcoef(p[s], q[s])[0, 1]
        print(f"   {m:17s} sd(anom) {np.nanstd(a):.3f}  sd(R) {np.nanstd(r):.3f}  lag1 anom {lag(a, 1):.2f}  "
              f"lag1 R {lag(r, 1):.2f}  lag5 R {lag(r, 5):.2f}")


def table_decomposition(tab, order):
    print("== (d) OLS (no intercept) of the lead 4-7 departure on national / regional / local parts of the 10-year gap ==")
    for m in METRICS:
        arr = tab[m]
        recs = []
        for t in range(1950, 1976):
            w = arr[:, wf.col(t - 29): wf.col(t) + 1]
            ref = np.where(np.isfinite(w).sum(1) >= 10, nanmean(w, 1), np.nan)
            g10 = nanmean(w[:, -10:], 1) - ref
            nat = np.nanmedian(g10)
            reg = nanmean(g10[order[:, :15]], 1)
            for h in LEADS:
                y = arr[:, wf.col(t + h)] - ref
                recs.append(np.column_stack([np.full(len(y), t), y, np.full(len(y), nat), reg, g10]))
        z = np.vstack(recs)
        z = z[np.isfinite(z).all(1)]
        t_, y, nat, reg, own = z.T
        full = np.linalg.lstsq(np.column_stack([nat, reg - nat, own - reg]), y, rcond=None)[0]
        parts = []
        for lo, hi in ((1950, 1962), (1963, 1975)):
            s = (t_ >= lo) & (t_ <= hi)
            b = np.linalg.lstsq(np.column_stack([reg[s] - nat[s], own[s] - reg[s]]), y[s] - nat[s], rcond=None)[0]
            parts.append(f"{lo}-{hi} regional {b[0]:+.2f} local {b[1]:+.2f}")
        print(f"   {m:17s} 1950-75 [national, regional, local] {np.round(full, 3)} | " + " | ".join(parts))


# ------------------------------------------------------------------------- (e)-(i): centred estimator tables
def era_rows(tab, m, ts, lfun):
    ys, refs, ls = [], [], []
    for t in ts:
        ref, l_raw = lfun(t)
        for h in LEADS:
            ys.append(tab[m][:, wf.col(t + h)])
            refs.append(ref)
            ls.append(l_raw)
    return np.array(ys), np.array(refs), np.array(ls)


def best_skill(m, y, ref, l, grid, base=None, extra=None):
    """In-era best b: skill of clip(base + centre(b*l) [+ centre(g*extra)]) against ref30, pooled over rows."""
    base = ref if base is None else base
    ok0 = np.isfinite(y) & np.isfinite(ref)
    out = (-np.inf, None)
    for b in grid:
        f = wf.deploy(m, base, centre(b * l) if extra is None else centre(b * np.nan_to_num(l) + extra))
        ok = ok0 & np.isfinite(f)
        s = 1 - np.abs(y - f)[ok].sum() / np.abs(y - ref)[ok].sum()
        if s > out[0] + 1e-15:
            out = (s, b)
    return out


def table_in_era(tab, pre):
    print("== (e) in-era best b, declared centred estimator, avg30 base; gain over the v0.5-like proxy ==")
    for m in METRICS:
        parts = []
        for era, ts in ERAS.items():
            y, ref, l = era_rows(tab, m, ts, lambda t: pre[t][m][:2])
            s, b = best_skill(m, y, ref, l, GRID)
            proxy = np.array([pre[t][m][3][h]["v05proxy"] for t in ts for h in LEADS])
            s0 = best_skill(m, y, ref, l, [0.0], base=proxy)[0]
            s1, b1 = best_skill(m, y, ref, l, GRID, base=proxy)
            parts.append(f"{era}: b={b:.2f} sk={s:+.5f} | over proxy b={b1:.2f} gain={s1 - s0:+.5f}")
        print(f"   {m:17s} " + " || ".join(parts))
    print("== (e2) v1.2 in-era best (b, g) on a 0.1 grid, combined centring ==")
    g01 = np.round(np.arange(0, 1.0 + 1e-9, 0.1), 1)
    for m in METRICS:
        parts = []
        for era, ts in ERAS.items():
            y, ref, l = era_rows(tab, m, ts, lambda t: pre[t][m][:2])
            ll = np.array([pre[t][m][2] for t in ts for h in LEADS])
            best = (-np.inf, 0, 0)
            for g in g01:
                defined = np.isfinite(l) | np.isfinite(ll)
                for b in g01:
                    v = np.where(defined, b * np.nan_to_num(l) + g * np.nan_to_num(ll), np.nan)
                    f = wf.deploy(m, ref, centre(v))
                    ok = np.isfinite(y) & np.isfinite(ref)
                    s = 1 - np.abs(y - f)[ok].sum() / np.abs(y - ref)[ok].sum()
                    if s > best[0] + 1e-15:
                        best = (s, b, g)
            parts.append(f"{era}: b={best[1]:.1f} g={best[2]:.1f} sk={best[0]:+.5f}")
        print(f"   {m:17s} " + " || ".join(parts))


def kriging_kernel(d, order, length):
    c = np.exp(-d / length)
    k = np.zeros_like(d)
    for i in range(len(d)):
        nb = order[i, :30]
        w = np.maximum(np.linalg.solve(c[np.ix_(nb, nb)] + 0.25 * np.eye(30), c[nb, i]), 0.0)
        k[i, nb] = w / w.sum()
    return k


def corr_kernel(arr, t, order):
    """Top 10 of the 30 nearest by first-difference correlation over t-29..t (>= 10 shared), weights corr^2, rows sum 1."""
    w = arr[:, wf.col(t - 29): wf.col(t) + 1]
    dif = np.diff(w, axis=1)
    msk = np.isfinite(dif).astype(float)
    z = np.where(msk > 0, dif, 0.0)
    n = msk @ msk.T
    sx = z @ msk.T
    sxx = (z ** 2) @ msk.T
    sxy = z @ z.T
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (n * sxy - sx * sx.T) / np.sqrt((n * sxx - sx ** 2) * (n * sxx.T - sx.T ** 2))
    corr = np.where((n >= 10) & np.isfinite(corr), corr, -1.0)
    k = np.zeros_like(corr)
    for i in range(len(corr)):
        cand = order[i, :30]
        top = cand[np.argsort(-corr[i, cand], kind="stable")[:10]]
        wt = np.maximum(corr[i, top], 0.0) ** 2
        if wt.sum() > 0:
            k[i, top] = wt / wt.sum()
    return k


def table_kernels(tab, d, order, efold):
    print("== (f) kernel horse-race: mean over 3 eras of in-era best skill (centred L, avg30 base, clipped) ==")
    kernels = {"knn5": wf.knn_kernel(order, 5), "knn15": wf.knn_kernel(order, 15), "knn25": wf.knn_kernel(order, 25),
               "gauss300": None, "gauss600": wf.gauss_kernel(d)}
    k300 = np.exp(-(d / 300.0) ** 2)
    np.fill_diagonal(k300, 0.0)
    kernels["gauss300"] = k300
    rows = {}
    for name, k in kernels.items():
        rows[name] = {m: np.mean([best_skill(m, *era_rows(tab, m, ts, lambda t: wf.terms_at(tab[m], t, k, m)[:2]),
                                             GRID)[0] for ts in ERAS.values()]) for m in METRICS}
    old = wf.MIN_KERNEL_MASS
    wf.MIN_KERNEL_MASS = 0.5                     # row-normalised weights sum to 1: half the weight must be valid
    try:
        rows["kriging"] = {m: np.mean([best_skill(m, *era_rows(tab, m, ts, lambda t, kk=kriging_kernel(d, order, efold[m]):
                                                              wf.terms_at(tab[m], t, kk, m)[:2]), GRID)[0]
                                       for ts in ERAS.values()]) for m in METRICS}
        rows["corr top-10"] = {m: np.mean([best_skill(m, *era_rows(tab, m, ts, lambda t: wf.terms_at(
            tab[m], t, corr_kernel(tab[m], t, order), m)[:2]), GRID)[0] for ts in ERAS.values()]) for m in METRICS}
    finally:
        wf.MIN_KERNEL_MASS = old
    for name, r in rows.items():
        print(f"   {name:12s} " + "  ".join(f"{m} {r[m]:+.4f}" for m in METRICS) + f"  mean {np.mean(list(r.values())):+.4f}")


def relative_30(arr, t, k):
    hist = arr[:, : wf.col(t) + 1]
    ref = wf.ref30(hist, t)
    x = hist[:, wf.col(t - 29): wf.col(t) + 1]
    a = x - ref[:, None]
    return ref, np.where(np.isfinite(x), a - wf.neighbour_mean(k, a), np.nan)


def eb_term(arr, t, k, m):
    ref, l_raw, _ = wf.terms_at(arr, t, k, m)
    _, r = relative_30(arr, t, k)
    ok = np.isfinite(r)
    w = np.where(ok, 0.5 ** (np.arange(29, -1, -1) / wf.HALF_LIFE[m]), 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        noise = np.nanvar(r, axis=1) * (w ** 2).sum(1) / w.sum(1) ** 2
    tau2 = max(np.nanmean(l_raw ** 2) - np.nanmean(np.where(np.isfinite(l_raw), noise, np.nan)), 1e-9)
    return ref, l_raw * tau2 / (tau2 + noise)


def break_term(arr, t, k, m, thr=4.0):
    ref, l_raw, _ = wf.terms_at(arr, t, k, m)
    _, r = relative_30(arr, t, k)
    ok = np.isfinite(r)
    rz = np.where(ok, r, 0.0)
    allm = np.where(ok, r, 0.0).sum(1) / np.maximum(ok.sum(1), 1)
    tstat, lvl = np.zeros(len(r)), np.full(len(r), np.nan)
    for s in range(3, 28):
        n1, n2 = ok[:, :s].sum(1), ok[:, s:].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            m1, m2 = rz[:, :s].sum(1) / n1, rz[:, s:].sum(1) / n2
            ss = (np.where(ok[:, :s], (r[:, :s] - m1[:, None]) ** 2, 0).sum(1)
                  + np.where(ok[:, s:], (r[:, s:] - m2[:, None]) ** 2, 0).sum(1))
            ts = np.abs(m2 - m1) / (np.sqrt(ss / (n1 + n2 - 2)) * np.sqrt(1 / n1 + 1 / n2))
        ts = np.where((n1 >= 3) & (n2 >= 3) & np.isfinite(ts), ts, 0.0)
        better = ts > tstat
        tstat, lvl = np.where(better, ts, tstat), np.where(better, m2 - allm, lvl)
    return ref, np.where(np.isfinite(l_raw), np.where(tstat > thr, lvl, l_raw), np.nan)


def table_rejected(tab, k):
    print("== (g)-(i) rejected alternatives vs centred L, in-era best b (grid 0-1.5), avg30 base ==")
    for m in METRICS:
        parts = []
        for era, ts in ERAS.items():
            s_l = best_skill(m, *era_rows(tab, m, ts, lambda t: wf.terms_at(tab[m], t, k, m)[:2]), GRID_WIDE)[0]
            s_eb = best_skill(m, *era_rows(tab, m, ts, lambda t: eb_term(tab[m], t, k, m)), GRID_WIDE)[0]
            s_br = best_skill(m, *era_rows(tab, m, ts, lambda t: break_term(tab[m], t, k, m)), GRID_WIDE)[0]
            parts.append(f"{era}: L {s_l:+.4f} EB {s_eb:+.4f} break {s_br:+.4f}")
        print(f"   {m:17s} " + " || ".join(parts))


def main():
    cities, coords, tab = wf.load()
    d = wf.distances(coords)
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    order = np.argsort(dn, axis=1, kind="stable")
    efold = table_correlation(tab, d, order)
    table_relative_noise(tab, order)
    table_decomposition(tab, order)
    k = wf.gauss_kernel(d)
    pre = wf.precompute(tab, k, order, range(1915, 1976))
    table_in_era(tab, pre)
    table_kernels(tab, d, order, efold)
    table_rejected(tab, k)


if __name__ == "__main__":
    main()
