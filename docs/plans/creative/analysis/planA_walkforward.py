"""Plan A (graph) chronological evidence: walk-forward, promotion analogue, dependence diagnostic, runtime, determinism.

Run from the repository root:  uv run python docs/plans/creative/analysis/planA_walkforward.py

Data rule and cutoffs (audit points):
- Reads only data/annual_valid.csv and data/city_info.csv, and drops every year > 1982 right after reading (assert).
- At each fit origin F the coefficient rows are pseudo-origins t in [F-45, F-4] and leads h in 4..7 with t + h <= F.
  `fit_coefs` asserts max(training target year) <= F before every fit.
- Graph terms at a pseudo-origin t read only columns <= t (`terms_at` slices the table at t; asserted).
- `end_to_end` repeats one origin from a table with every year > F set to NaN and checks that it reproduces the
  walk-forward coefficients and forecast exactly, which shows the walk-forward used nothing after F.
Plan: docs/plans/creative/plan-A-graph.md (section 3.4 and appendix A).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
LAST = 1982
METRICS = ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"]
NONNEG = {"total_precip_in", "cdd"}
HALF_LIFE = {"annual_avg_temp": 3, "hottest_day_temp": 3, "cdd": 3, "total_precip_in": 6}
KERNEL_KM = 600.0
MIN_KERNEL_MASS = 1.0
MIN_REL_YEARS = 10
MIN_LONG_YEARS = 15
GRID = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 2)
FIT_LEADS = (4, 5, 6, 7)
TRAIN_YEARS = 45
MIN_COEF_ROWS = 2000
Y0 = 1780                       # NaN padding so 60-year windows never index before the array
FIT_ORIGINS = list(range(1940, 1976))
BLOCKS = {"1940-49": range(1940, 1950), "1950-62": range(1950, 1963), "1963-75": range(1963, 1976)}
SEED = 20261108
DRAWS = 2000


# ------------------------------------------------------------------------------------------------- data
def load(cut: int = LAST):
    annual = pd.read_csv(ROOT / "data" / "annual_valid.csv", dtype={"city_id": str})
    annual = annual[annual.year <= LAST]
    assert annual.year.max() <= LAST, "design analysis must not read years after 1982"
    info = pd.read_csv(ROOT / "data" / "city_info.csv", dtype={"city_id": str}).set_index("city_id")
    cities = sorted(annual.city_id.unique())
    tab = {}
    for m in METRICS:
        wide = annual.pivot(index="city_id", columns="year", values=m).reindex(index=cities)
        arr = wide.reindex(columns=range(Y0, LAST + 1)).to_numpy(float)
        arr[:, cut - Y0 + 1:] = np.nan                    # years > cut are invisible
        tab[m] = arr
    return cities, info.reindex(cities)[["lat", "lon"]], tab


def col(year: int) -> int:
    return year - Y0


def distances(coords: pd.DataFrame) -> np.ndarray:
    lat, lon = np.radians(coords.lat.to_numpy(float)), np.radians(coords.lon.to_numpy(float))
    a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.minimum(1.0, np.sqrt(a)))


def gauss_kernel(d: np.ndarray) -> np.ndarray:
    k = np.exp(-(d / KERNEL_KM) ** 2)
    np.fill_diagonal(k, 0.0)
    return k


def nmean(x, axis=1):
    ok = np.isfinite(x)
    c = ok.sum(axis)
    return np.where(c > 0, np.where(ok, x, 0.0).sum(axis) / np.maximum(c, 1), np.nan)


def ref30(arr, t):
    w = arr[:, col(t - 29): col(t) + 1]
    return np.where(np.isfinite(w).sum(1) >= 10, nmean(w), np.nan)


def neighbour_mean(k, a):
    ok = np.isfinite(a)
    num = (k[:, :, None] * np.where(ok, a, 0.0)[None]).sum(1)      # broadcast sums: thread-independent order
    den = (k[:, :, None] * ok[None]).sum(1)
    return np.where(den >= MIN_KERNEL_MASS, num / np.where(den > 0, den, 1.0), np.nan)


# ------------------------------------------------------------------------------- graph terms, W, combo
def terms_at(arr, t, k, metric):
    """ref30, L_raw and L_long_raw at pseudo-origin t, from columns <= t only."""
    hist = arr[:, : col(t) + 1]                                       # nothing after t is visible
    assert hist.shape[1] == col(t) + 1
    ref = ref30(hist, t)
    x = hist[:, col(t - 59): col(t) + 1]
    a = x - ref[:, None]
    r = np.where(np.isfinite(x), a - neighbour_mean(k, a), np.nan)
    r30 = r[:, 30:]
    ok = np.isfinite(r30)
    nv = ok.sum(1)
    w = 0.5 ** (np.arange(29, -1, -1) / HALF_LIFE[metric])
    ewma = (np.where(ok, r30, 0.0) * w).sum(1) / np.maximum((ok * w).sum(1), 1e-12)
    m30 = np.where(ok, r30, 0.0).sum(1) / np.maximum(nv, 1)
    l_raw = np.where(nv >= MIN_REL_YEARS, ewma - m30, np.nan)
    ok_all = np.isfinite(r)
    m60 = np.where(ok_all, r, 0.0).sum(1) / np.maximum(ok_all.sum(1), 1)
    l_long = np.where((ok_all[:, :30].sum(1) >= MIN_LONG_YEARS) & (nv >= MIN_REL_YEARS), m60 - m30, np.nan)
    return ref, l_raw, l_long


def slope_state(arr, t):
    w = arr[:, col(t - 29): col(t) + 1]
    yrs = np.arange(t - 29, t + 1, dtype=float)
    ok = np.isfinite(w)
    nv = ok.sum(1)
    xm = np.where(ok, yrs, 0).sum(1) / np.maximum(nv, 1)
    ym = np.where(ok, w, 0).sum(1) / np.maximum(nv, 1)
    dx = np.where(ok, yrs - xm[:, None], 0)
    dy = np.where(ok, w - ym[:, None], 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        sl = (dx * dy).sum(1) / (dx ** 2).sum(1)
    return np.where((nv >= 10) & np.isfinite(sl), sl, np.nan), np.where(nv > 0, xm, np.nan), ym


def warming_rates(tab, t, order):
    """v0.5's W rates at t: neighbour-averaged 30-year temperature slope (floored at 0); CDD via its temperature response."""
    def with_self(v):
        return nmean(np.column_stack([v, v[order[:, :15]]]))
    rate_t = np.maximum(np.nan_to_num(with_self(slope_state(tab["annual_avg_temp"], t)[0])), 0.0)
    tw = tab["annual_avg_temp"][:, col(t - 29): col(t) + 1]
    cw = tab["cdd"][:, col(t - 29): col(t) + 1]
    ok = np.isfinite(tw) & np.isfinite(cw)
    nj = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(ok, tw, 0).sum(1) / nj
        ym = np.where(ok, cw, 0).sum(1) / nj
        dx = np.where(ok, tw - xm[:, None], 0)
        dy = np.where(ok, cw - ym[:, None], 0)
        resp = (dx * dy).sum(1) / (dx ** 2).sum(1)
    resp = np.where((nj >= 15) & np.isfinite(resp), resp, np.nan)
    rate_c = rate_t * np.maximum(np.nan_to_num(with_self(resp)), 0.0)
    zero = np.zeros_like(rate_t)
    return {"annual_avg_temp": rate_t, "cdd": rate_c, "total_precip_in": zero, "hottest_day_temp": zero}


def precompute(tab, k, order, ts):
    pre = {}
    for t in ts:
        rates = warming_rates(tab, t, order)
        pre[t] = {"wpos": float((rates["annual_avg_temp"] > 0).mean())}
        for m in METRICS:
            ref, l_raw, l_long = terms_at(tab[m], t, k, m)
            sl, xm, ym = slope_state(tab[m], t)
            avg = np.where(np.isfinite(ref), ref, ym)
            last10 = nmean(tab[m][:, col(t - 9): col(t) + 1])
            last10 = np.where(np.isfinite(last10), last10, avg)
            base = {}
            for h in range(1, 8):
                w = np.nan_to_num(rates[m] * (t + h - xm))
                if m in ("annual_avg_temp", "cdd"):
                    combo = 0.5 * np.where(np.isfinite(sl), ym + sl * (t + h - xm), avg) + 0.5 * last10
                else:
                    combo = avg
                base[h] = {"avg30": ref, "avg30_halfW": ref + 0.5 * w, "v05proxy": 0.5 * (ref + w) + 0.5 * combo}
            pre[t][m] = (ref, l_raw, l_long, base)
    return pre


# ---------------------------------------------------------------------------------- G, deploy, fitting
def graph_shift(beta, gamma, l_raw, l_long, use_long):
    """G = centre(beta L_raw [+ gamma L_long_raw]): median over defined cities removed, undefined -> 0. Row-wise.

    v1.1 (use_long False): defined where L_raw is defined.  v1.2 (use_long True): defined where either part is.
    """
    defined = np.isfinite(l_raw) | (np.isfinite(l_long) if use_long else False)
    v = beta * np.nan_to_num(l_raw) + (gamma * np.nan_to_num(l_long) if use_long else 0.0)
    v = np.where(defined, v, np.nan)
    med = np.nanmedian(v, axis=-1, keepdims=True)
    return np.where(defined, v - med, 0.0)


def deploy(metric, base, g):
    f = base + g
    return np.maximum(f, 0.0) if metric in NONNEG else f


def coef_rows(tab, pre, metric, fit_origin, base_kind):
    ys, bs, ls, lls, target_years = [], [], [], [], []
    for t in range(fit_origin - TRAIN_YEARS, fit_origin - FIT_LEADS[0] + 1):
        if t not in pre:
            continue
        ref, l_raw, l_long, base = pre[t][metric]
        for h in FIT_LEADS:
            if t + h > fit_origin:                         # the cutoff: no training target after the fit origin
                continue
            ys.append(tab[metric][:, col(t + h)])
            bs.append(base[h][base_kind])
            ls.append(l_raw)
            lls.append(l_long)
            target_years.append(t + h)
    return np.array(ys), np.array(bs), np.array(ls), np.array(lls), target_years


def fit_coefs(tab, pre, metric, fit_origin, base_kind, use_long):
    y, b, l_raw, l_long, target_years = coef_rows(tab, pre, metric, fit_origin, base_kind)
    assert max(target_years) <= fit_origin, f"training target {max(target_years)} after fit origin {fit_origin}"
    y = np.where(np.isfinite(b), y, np.nan)
    ok = np.isfinite(y)
    if ok.sum() < MIN_COEF_ROWS:
        return 0.0, 0.0
    best = (np.inf, 0.0, 0.0)
    for gamma in (GRID if use_long else [0.0]):
        for beta in GRID:
            e = np.abs(y - deploy(metric, b, graph_shift(beta, gamma, l_raw, l_long, use_long)))
            loss = np.where(ok, e, 0.0).sum() / ok.sum()
            if loss < best[0] - 1e-15:                     # ascending scan: ties keep the smallest coefficients
                best = (loss, beta, gamma)
    return best[1], best[2]


CANDIDATES = {                      # name: (base, use_long, fitted)
    "v1.1": ("avg30", False, True),
    "v1.2": ("avg30", True, True),
    "diag avg30+G+0.5W": ("avg30_halfW", False, True),
    "ref v0.5-like proxy (no G)": ("v05proxy", False, False),
}


def forecast_at(pre, metric, fit_origin, cand, coefs):
    base_kind, use_long, fitted = CANDIDATES[cand]
    ref, l_raw, l_long, base = pre[fit_origin][metric]
    beta, gamma = coefs if fitted else (0.0, 0.0)
    g = graph_shift(beta, gamma, l_raw, l_long, use_long)
    return {h: deploy(metric, base[h][base_kind], g) for h in range(1, 8)}, ref


# -------------------------------------------------------------------------------------------- scoring
def city_sums(tab, metric, fit_origin, fc, ref):
    out = {}
    for split, leads in (("live", (1, 2, 3)), ("final", (4, 5, 6, 7))):
        e = np.zeros(len(ref))
        r = np.zeros(len(ref))
        for h in leads:
            y = tab[metric][:, col(fit_origin + h)]
            ok = np.isfinite(y) & np.isfinite(ref)
            e += np.where(ok, np.abs(y - fc[h]), 0.0)
            r += np.where(ok, np.abs(y - ref), 0.0)
        out[split] = (e, r)
    return out


def verdict(sums, cand, trip, weights):
    """reward.py-style comparison against average30 (reference skill is 0 by definition)."""
    def pts(split, w):
        return 100 * np.mean([[1 - (w @ sums[(cand, o, m)][split][0]) / (w @ sums[(cand, o, m)][split][1])
                                for m in METRICS] for o in trip], axis=(0, 1))
    fin, liv = pts("final", weights), pts("live", weights)
    p_better = float((fin[1:] > 0).mean())
    lb = float(np.quantile(fin[1:], 0.05))
    by_origin = [100 * np.mean([1 - sums[(cand, o, m)]["final"][0].sum() / sums[(cand, o, m)]["final"][1].sum()
                                for m in METRICS]) for o in trip]
    floor = min(100 * np.mean([1 - sums[(cand, o, m)]["final"][0].sum() / sums[(cand, o, m)]["final"][1].sum()
                               for o in trip]) for m in METRICS)
    live_p_worse = float((liv[1:] < 0).mean())
    promote = (p_better >= 0.95 and fin[0] > 0 and sum(v > 0 for v in by_origin) >= 2
               and live_p_worse < 0.95 and floor >= -2.0)
    return fin[0], p_better, lb, floor, promote


def city_weights(n, rng):
    return np.vstack([np.ones(n), rng.multinomial(n, np.full(n, 1.0 / n), size=DRAWS).astype(float)])


def block_weights(labels, rng):
    """Spatial-block bootstrap: resample blocks with replacement; every city in a drawn block gets the block's count."""
    blocks = np.unique(labels)
    idx = np.searchsorted(blocks, labels)
    counts = rng.multinomial(len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=DRAWS).astype(float)
    return np.vstack([np.ones(len(labels)), counts[:, idx]])


# ------------------------------------------------------------------------------------------ main
def main():
    cities, coords, tab = load()
    n = len(cities)
    d = distances(coords)
    k = gauss_kernel(d)
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    order = np.argsort(dn, axis=1, kind="stable")
    t0 = time.perf_counter()
    pre = precompute(tab, k, order, range(FIT_ORIGINS[0] - TRAIN_YEARS, FIT_ORIGINS[-1] + 1))
    print(f"precompute: {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    coefs, sums, forecasts = {}, {}, {}
    for fo in FIT_ORIGINS:
        for m in METRICS:
            for cand, (base_kind, use_long, fitted) in CANDIDATES.items():
                c = fit_coefs(tab, pre, m, fo, base_kind, use_long) if fitted else (0.0, 0.0)
                coefs[(cand, fo, m)] = c
                fc, ref = forecast_at(pre, m, fo, cand, c)
                forecasts[(cand, fo, m)] = fc
                sums[(cand, fo, m)] = city_sums(tab, m, fo, fc, ref)
    print(f"walk-forward ({len(FIT_ORIGINS)} fit origins x {len(CANDIDATES)} candidates x 4 metrics): "
          f"{time.perf_counter() - t0:.1f} s")

    def skill(cand, fo, m, split):
        e, r = sums[(cand, fo, m)][split]
        return 1 - e.sum() / r.sum()

    print("\n== Held-out points against average30, by block (final = leads 4-7, live = leads 1-3) ==")
    for cand in CANDIDATES:
        print(cand)
        for name, fos in BLOCKS.items():
            fin = [100 * np.mean([skill(cand, fo, m, "final") for m in METRICS]) for fo in fos]
            live = [100 * np.mean([skill(cand, fo, m, "live") for m in METRICS]) for fo in fos]
            per_m = [100 * np.mean([skill(cand, fo, m, "final") for fo in fos]) for m in METRICS]
            print(f"   {name}: final {np.mean(fin):+.4f}  positive {np.mean(np.array(fin) > 0):.0%}  "
                  f"min {min(fin):+.2f}  live {np.mean(live):+.4f}  | temp {per_m[0]:+.2f} precip {per_m[1]:+.2f} "
                  f"hot {per_m[2]:+.2f} cdd {per_m[3]:+.2f}")

    print("\n== Coefficients (beta, gamma) by fit origin ==")
    show = (1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975)
    for cand in ("v1.1", "v1.2", "diag avg30+G+0.5W"):
        for m in METRICS:
            row = " ".join(f"{fo}:({coefs[(cand, fo, m)][0]:.2f},{coefs[(cand, fo, m)][1]:.2f})" for fo in show)
            b = np.median([coefs[(cand, fo, m)][0] for fo in FIT_ORIGINS])
            g = np.median([coefs[(cand, fo, m)][1] for fo in FIT_ORIGINS])
            print(f"   {cand:18s} {m:17s} {row}  | median over 36: ({b:.3f},{g:.3f})")
    print("   share of cities with positive warming rate, mean per block:",
          {name: round(float(np.mean([pre[fo]["wpos"] for fo in fos])), 2) for name, fos in BLOCKS.items()})

    # Promotion analogue: triplets spaced like 1983/1990/1997, three resampling schemes
    lat, lon = coords.lat.to_numpy(float), coords.lon.to_numpy(float)
    schemes = {"city (official scheme)": None,
               "5-degree blocks": (np.floor(lat / 5) * 1000 + np.floor(lon / 5)).astype(int),
               "10-degree blocks": (np.floor(lat / 10) * 1000 + np.floor(lon / 10)).astype(int)}
    trips = [(f, f + 7, f + 14) for f in range(1940, 1962)]
    print(f"\n== Promotion analogue: {len(trips)} held-out triplets (F, F+7, F+14), F = 1940..1961, {DRAWS} draws ==")
    for sname, labels in schemes.items():
        nblocks = n if labels is None else len(np.unique(labels))
        print(f"-- {sname}: {nblocks} resampling units")
        for cand in CANDIDATES:
            rng = np.random.default_rng(SEED)
            res = []
            for trip in trips:
                w = city_weights(n, rng) if labels is None else block_weights(labels, rng)
                res.append(verdict(sums, cand, trip, w))
            r = pd.DataFrame(res, columns=["pts", "p", "lb", "floor", "promote"])
            print(f"   {cand:26s} median reward {r.pts.median():+.4f}  median P {r.p.median():.3f}  "
                  f"min P {r.p.min():.3f}  median LB90 {r.lb.median():+.4f}  min LB90 {r.lb.min():+.4f}  "
                  f"worst floor {r.floor.min():+.4f}  promote {int(r.promote.sum())}/{len(r)}")

    # Runtime and determinism of one complete origin from a table cut at F (what an adapter would see)
    print("\n== End-to-end single origin from a table cut at F (load, pivot, graph terms, fit v1.1 + v1.2, forecast) ==")
    for fo in (1940, 1960, 1975):
        digests, times = [], []
        for _ in range(2):
            t0 = time.perf_counter()
            _, coords_f, tab_f = load(cut=fo)
            k_f = gauss_kernel(distances(coords_f))
            pre_f = precompute(tab_f, k_f, order, range(fo - TRAIN_YEARS, fo + 1))
            out = []
            for m in METRICS:
                for cand in ("v1.1", "v1.2"):
                    base_kind, use_long, _ = CANDIDATES[cand]
                    c = fit_coefs(tab_f, pre_f, m, fo, base_kind, use_long)
                    assert c == coefs[(cand, fo, m)], f"cut-table coefficients differ at {fo} {m} {cand}"
                    fc, _ = forecast_at(pre_f, m, fo, cand, c)
                    for h in range(1, 8):
                        assert np.array_equal(fc[h], forecasts[(cand, fo, m)][h], equal_nan=True)
                        out.append(fc[h])
            times.append(time.perf_counter() - t0)
            digests.append(hashlib.sha256(np.ascontiguousarray(np.array(out)).tobytes()).hexdigest()[:16])
        print(f"   F={fo}: {times[0]:.1f} s and {times[1]:.1f} s; sha256 {digests[0]} / {digests[1]}; "
              f"identical={digests[0] == digests[1]}; matches walk-forward: yes")


if __name__ == "__main__":
    main()
