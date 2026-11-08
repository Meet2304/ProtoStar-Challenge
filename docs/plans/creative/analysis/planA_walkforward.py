"""Plan A (graph) chronological evidence: walk-forward, promotion analogue, dependence diagnostic, runtime, determinism.

Run from the repository root:  uv run python docs/plans/creative/analysis/planA_walkforward.py   (about 5 minutes)

Data rule and cutoffs (audit points):
- `load` parses the whole data/annual_valid.csv and data/city_info.csv, then keeps only years <= 1982 (asserted)
  before anything else is computed.
- At each fit origin F the coefficient rows are pseudo-origins t in [F-45, F-4] and leads h in 4..7 with t + h <= F.
  `fit_coefs` asserts max(training target year) <= F before every fit.
- Graph terms and the national increment at a pseudo-origin t read only columns <= t (`terms_at`, `national_shift`).
- The end-to-end check repeats three origins from a table with every year > F set to NaN and checks that it reproduces the
  walk-forward coefficients and forecasts exactly, which shows the walk-forward used nothing after F.
- The promotion analogue follows code/evaluation/reward.py: seed 20261108 reset for every comparison (triplet), and only
  cities with at least one scored row in that triplet are resampled.
Plan: docs/plans/creative/plan-A-graph.md (sections 3.4, 3.5 and appendix A).

r6 adds the early-era diagnostic: fit origins 1915-1939 with the same fit, the same cutoff asserts and the unchanged
parameters (reused design-era years), plus a cut-table check at F = 1925. Fits for F >= 1940 are unchanged.

r5 adds v1.3 (v1.1 + a national warm increment; KAPPA and CAP below, which the author reports fixing before this script
first computed any v1.3 score; there is no timestamped declaration), the reward.py-exact bootstrap, Monte Carlo seeds for the weakest triplet, overlap counts of
the fit sample, the fallback-city counts, and two diagnostic rows (25-NN graph, raw national shift).
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
# v1.3 national warm increment, declared in plan r5 before any v1.3 number was computed (not tuned):
KAPPA = 0.5                     # shrink applied to the floored, capped national 10-year gap
CAP_F = 1.0                     # cap on the national 10-year gap (F); so the temperature increment is at most 0.5 F
GAP_MIN_VALID = 5               # valid years needed in t-9..t for a city's 10-year gap (as v0.5's gap10)
Y0 = 1780                       # NaN padding so 60-year windows never index before the array
FIT_ORIGINS = list(range(1940, 1976))
BLOCKS = {"1940-49": range(1940, 1950), "1950-62": range(1950, 1963), "1963-75": range(1963, 1976)}
# r6: early-era diagnostic (reused design-era years; same fit, cutoff asserts and frozen parameters)
EARLY_ORIGINS = list(range(1915, 1940))
EARLY_BLOCKS = {"1915-35": range(1915, 1936), "1925-39": range(1925, 1940)}
ALL_ORIGINS = sorted(set(EARLY_ORIGINS) | set(FIT_ORIGINS))
SEED = 20261108
DRAWS = 2000


# ------------------------------------------------------------------------------------------------- data
def load(cut: int = LAST):
    annual = pd.read_csv(ROOT / "data" / "annual_valid.csv", dtype={"city_id": str})
    annual = annual[annual.year <= LAST]
    assert annual.year.max() <= LAST, "design analysis must not use years after 1982"
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


def knn_kernel(order: np.ndarray, kn: int) -> np.ndarray:
    k = np.zeros((len(order), len(order)))
    for i in range(len(order)):
        k[i, order[i, :kn]] = 1.0
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
    """v0.5's W rates at t, plus rho: the regional CDD-on-temperature response (self + 15 nearest), floored at 0."""
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
    rho = np.maximum(np.nan_to_num(with_self(resp)), 0.0)
    zero = np.zeros_like(rate_t)
    return {"annual_avg_temp": rate_t, "cdd": rate_t * rho, "total_precip_in": zero, "hottest_day_temp": zero}, rho


def national_gap(arr, t, ref):
    """Cross-city median of (mean of valid years in t-9..t) - ref30(t); cities need >= 5 valid years and a ref30."""
    w = arr[:, col(t - 9): col(t) + 1]
    gap = np.where((np.isfinite(w).sum(1) >= GAP_MIN_VALID) & np.isfinite(ref), nmean(w) - ref, np.nan)
    return float(np.nanmedian(gap)) if np.isfinite(gap).any() else 0.0


def national_shift(n_gap, rho, kappa, cap, floor):
    """Uniform temperature increment and its CDD conversion through rho. Returns per-metric arrays."""
    g = n_gap
    if floor:
        g = max(g, 0.0)
    if cap is not None:
        g = min(g, cap)
    inc = kappa * g
    zero = np.zeros_like(rho)
    return {"annual_avg_temp": np.full_like(rho, inc), "cdd": inc * rho, "total_precip_in": zero,
            "hottest_day_temp": zero}, inc


def precompute(tab, k, order, ts):
    pre = {}
    for t in ts:
        rates, rho = warming_rates(tab, t, order)
        refs = {}
        pre[t] = {"wpos": float((rates["annual_avg_temp"] > 0).mean())}
        for m in METRICS:
            refs[m] = terms_at(tab[m], t, k, m)
        n_gap = national_gap(tab["annual_avg_temp"], t, refs["annual_avg_temp"][0])
        nat, inc = national_shift(n_gap, rho, KAPPA, CAP_F, floor=True)              # v1.3, declared
        raw, _ = national_shift(n_gap, rho, 1.0, None, floor=False)                  # diagnostic only
        pre[t].update(n_gap=n_gap, inc=inc)
        for m in METRICS:
            ref, l_raw, l_long = refs[m]
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
                base[h] = {"avg30": ref, "avg30_halfW": ref + 0.5 * w, "v05proxy": 0.5 * (ref + w) + 0.5 * combo,
                           "avg30_nat": ref + nat[m], "avg30_natraw": ref + raw[m]}
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


def overlap_counts(tab, pre, metric, fit_origin):
    """Examples in the v1.1 fit sample, unique city-years among their targets, and median reuse of a target."""
    y, b, _, _, target_years = coef_rows(tab, pre, metric, fit_origin, "avg30")
    ok = np.isfinite(y) & np.isfinite(b)
    years = np.array(target_years)[:, None].repeat(y.shape[1], axis=1)[ok]
    cities = np.arange(y.shape[1])[None, :].repeat(y.shape[0], axis=0)[ok]
    _, reuse = np.unique(cities * 10000 + years, return_counts=True)
    return int(ok.sum()), int(len(reuse)), float(np.median(reuse))


CANDIDATES = {                      # name: (kernel, base, use_long, fitted)
    "v1.1": ("gauss", "avg30", False, True),
    "v1.2": ("gauss", "avg30", True, True),
    "v1.3": ("gauss", "avg30_nat", False, True),
    "diag v1.1 graph=knn25": ("knn25", "avg30", False, True),
    "diag v1.1 + raw national shift": ("gauss", "avg30_natraw", False, True),
    "diag avg30+G+0.5W": ("gauss", "avg30_halfW", False, True),
    "ref v0.5-like proxy (no G)": ("gauss", "v05proxy", False, False),
}


def forecast_at(pre, metric, fit_origin, cand, coefs):
    _, base_kind, use_long, fitted = CANDIDATES[cand]
    ref, l_raw, l_long, base = pre[fit_origin][metric]
    beta, gamma = coefs if fitted else (0.0, 0.0)
    g = graph_shift(beta, gamma, l_raw, l_long, use_long)
    return {h: deploy(metric, base[h][base_kind], g) for h in range(1, 8)}, ref, g


# -------------------------------------------------------------------------------------------- scoring
def city_sums(tab, metric, fit_origin, fc, ref):
    """Per city: (absolute error sum, reference error sum, scored-row count) for live and final."""
    out = {}
    for split, leads in (("live", (1, 2, 3)), ("final", (4, 5, 6, 7))):
        e, r, c = np.zeros(len(ref)), np.zeros(len(ref)), np.zeros(len(ref))
        for h in leads:
            y = tab[metric][:, col(fit_origin + h)]
            ok = np.isfinite(y) & np.isfinite(ref)
            e += np.where(ok, np.abs(y - fc[h]), 0.0)
            r += np.where(ok, np.abs(y - ref), 0.0)
            c += ok
        out[split] = (e, r, c)
    return out


def compare(sums, cand, champ, trip, seed=SEED, labels=None):
    """reward.py's compare() on an analogue triplet. champ = a candidate name or 'average30' (skill 0 by definition).

    Seed reset per call; only cities with a scored row in the triplet are resampled (reward.py's Run.cities).
    labels: optional spatial-block ids for the dependence diagnostic (blocks resampled, cities get their block's count).
    """
    cells = [(o, s, m) for o in trip for s in ("live", "final") for m in METRICS]
    counts = sum(sums[(cand, o, m)][s][2] for o, s, m in cells)
    idx = np.where(counts > 0)[0]
    n = len(idx)
    rng = np.random.default_rng(seed)
    if labels is None:
        w = rng.multinomial(n, np.full(n, 1.0 / n), size=DRAWS).astype(float)
    else:
        blocks, inv = np.unique(labels[idx], return_inverse=True)
        w = rng.multinomial(len(blocks), np.full(len(blocks), 1.0 / len(blocks)), size=DRAWS).astype(float)[:, inv]
    w = np.vstack([np.ones(n), w])

    def sk(name, o, s, m):
        e, r, _ = sums[(name, o, m)][s]
        return 1.0 - (w @ e[idx]) / (w @ r[idx])

    delta = {s: np.array([[100 * (sk(cand, o, s, m) - (0.0 if champ == "average30" else sk(champ, o, s, m)))
                           for m in METRICS] for o in trip]).mean(axis=1) for s in ("live", "final")}   # (origins, 1+B)
    fin, liv = delta["final"].mean(axis=0), delta["live"].mean(axis=0)
    p_better = float((fin[1:] > 0).mean())
    wins = int((delta["final"][:, 0] > 0).sum())
    floor = min(100 * np.mean([1 - sums[(cand, o, m)]["final"][0].sum() / sums[(cand, o, m)]["final"][1].sum()
                               for o in trip]) for m in METRICS)
    live_p_worse = float((liv[1:] < 0).mean())
    promote = p_better >= 0.95 and fin[0] > 0 and wins >= 2 and live_p_worse < 0.95 and floor >= -2.0
    return {"pts": fin[0], "p": p_better, "lb": float(np.quantile(fin[1:], 0.05)), "floor": floor, "wins": wins,
            "live_p_worse": live_p_worse, "promote": promote, "n_cities": n,
            "by_origin": delta["final"][:, 0]}


# ------------------------------------------------------------------------------------------ main
def main():
    cities, coords, tab = load()
    n = len(cities)
    d = distances(coords)
    dn = d.copy()
    np.fill_diagonal(dn, np.inf)
    order = np.argsort(dn, axis=1, kind="stable")
    ts = range(ALL_ORIGINS[0] - TRAIN_YEARS, ALL_ORIGINS[-1] + 1)
    t0 = time.perf_counter()
    pres = {"gauss": precompute(tab, gauss_kernel(d), order, ts), "knn25": precompute(tab, knn_kernel(order, 25), order, ts)}
    pre = pres["gauss"]
    print(f"precompute (two kernels): {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    coefs, sums, forecasts, shifts = {}, {}, {}, {}
    for fo in ALL_ORIGINS:
        for m in METRICS:
            for cand, (kern, base_kind, use_long, fitted) in CANDIDATES.items():
                c = fit_coefs(tab, pres[kern], m, fo, base_kind, use_long) if fitted else (0.0, 0.0)
                coefs[(cand, fo, m)] = c
                fc, ref, g = forecast_at(pres[kern], m, fo, cand, c)
                forecasts[(cand, fo, m)], shifts[(cand, fo, m)] = fc, g
                sums[(cand, fo, m)] = city_sums(tab, m, fo, fc, ref)
    # precip_off: v1.1 with G = 0 for precipitation (forecast = average30 there)
    for fo in ALL_ORIGINS:
        for m in METRICS:
            src = "ref v0.5-like proxy (no G)" if m == "total_precip_in" else "v1.1"   # proxy = average30 for precip
            sums[("diag v1.1 precip=off", fo, m)] = sums[(src, fo, m)]
    names = list(CANDIDATES) + ["diag v1.1 precip=off"]
    print(f"walk-forward ({len(FIT_ORIGINS)} fit origins x {len(CANDIDATES)} candidates x 4 metrics): "
          f"{time.perf_counter() - t0:.1f} s")

    def skill(cand, fo, m, split):
        e, r, _ = sums[(cand, fo, m)][split]
        return 1 - e.sum() / r.sum()

    print("\n== Held-out points against average30, by block (final = leads 4-7, live = leads 1-3) ==")
    for cand in names:
        print(cand)
        for bname, fos in BLOCKS.items():
            fin = [100 * np.mean([skill(cand, fo, m, "final") for m in METRICS]) for fo in fos]
            live = [100 * np.mean([skill(cand, fo, m, "live") for m in METRICS]) for fo in fos]
            per_m = [100 * np.mean([skill(cand, fo, m, "final") for fo in fos]) for m in METRICS]
            print(f"   {bname}: final {np.mean(fin):+.4f}  positive {np.mean(np.array(fin) > 0):.0%}  "
                  f"min {min(fin):+.2f}  live {np.mean(live):+.4f}  | temp {per_m[0]:+.2f} precip {per_m[1]:+.2f} "
                  f"hot {per_m[2]:+.2f} cdd {per_m[3]:+.2f}")

    print("\n== National increment (v1.3) by block: national 10-year gap N, increment I = 0.5*min(max(N,0),1) ==")
    for bname, fos in BLOCKS.items():
        ng = np.array([pre[fo]["n_gap"] for fo in fos])
        inc = np.array([pre[fo]["inc"] for fo in fos])
        print(f"   {bname}: N mean {ng.mean():+.3f} F (min {ng.min():+.3f}, max {ng.max():+.3f}); I mean {inc.mean():.3f} F; "
              f"I > 0 at {np.mean(inc > 0):.0%} of origins; cap binding at {np.mean(ng > CAP_F):.0%}")
    print("   v1.3 minus v1.1, final points by block:",
          {b: round(float(np.mean([100 * np.mean([skill("v1.3", fo, m, "final") - skill("v1.1", fo, m, "final")
                                                    for m in METRICS]) for fo in fos])), 4) for b, fos in BLOCKS.items()})

    print("\n== r6 early-era diagnostic: fit origins 1915-1939 (reused design-era years; same cutoff asserts) ==")
    for bname, fos in EARLY_BLOCKS.items():
        ng = np.array([pre[fo]["n_gap"] for fo in fos])
        inc = np.array([pre[fo]["inc"] for fo in fos])
        nex = [overlap_counts(tab, pre, "annual_avg_temp", fo)[0] for fo in fos]
        nexp = [overlap_counts(tab, pre, "total_precip_in", fo)[0] for fo in fos]
        print(f"   {bname}: N mean {ng.mean():+.3f} F (min {ng.min():+.3f}, max {ng.max():+.3f}); N > 0 at {np.mean(ng > 0):.0%}; "
              f"I mean {inc.mean():.3f} F; cap binding at {np.mean(ng > CAP_F):.0%}; "
              f"fit examples per origin: temp min {min(nex)}, precip min {min(nexp)}")
        for cand in ("v1.1", "v1.2", "v1.3", "diag v1.1 + raw national shift", "diag avg30+G+0.5W",
                     "ref v0.5-like proxy (no G)"):
            fin = [100 * np.mean([skill(cand, fo, m, "final") for m in METRICS]) for fo in fos]
            live = [100 * np.mean([skill(cand, fo, m, "live") for m in METRICS]) for fo in fos]
            per_m = [100 * np.mean([skill(cand, fo, m, "final") for fo in fos]) for m in METRICS]
            print(f"      {cand:30s} final {np.mean(fin):+.4f}  positive {np.mean(np.array(fin) > 0):.0%}  "
                  f"min {min(fin):+.2f}  live {np.mean(live):+.4f}  | temp {per_m[0]:+.2f} precip {per_m[1]:+.2f} "
                  f"hot {per_m[2]:+.2f} cdd {per_m[3]:+.2f}")
        diff = [100 * np.mean([skill("v1.3", fo, m, "final") - skill("v1.1", fo, m, "final") for m in METRICS])
                for fo in fos]
        dlive = [100 * np.mean([skill("v1.3", fo, m, "live") - skill("v1.1", fo, m, "live") for m in METRICS])
                 for fo in fos]
        print(f"      v1.3 minus v1.1: final {np.mean(diff):+.4f} (positive at {np.mean(np.array(diff) > 0):.0%} of origins, "
              f"min {min(diff):+.2f}), live {np.mean(dlive):+.4f}")
    early_trips = [(f, f + 7, f + 14) for f in range(1915, 1926)]
    for cand, champ in (("v1.1", "average30"), ("v1.3", "average30"), ("v1.3", "v1.1")):
        res = pd.DataFrame([compare(sums, cand, champ, trip, SEED) for trip in early_trips])
        print(f"   early triplets F=1915..1925: {cand} vs {champ}: median reward {res.pts.median():+.4f}  "
              f"median P {res.p.median():.3f}  min P {res.p.min():.4f}  median LB90 {res.lb.median():+.4f}  "
              f"worst floor {res.floor.min():+.4f}  promote {int(res.promote.sum())}/{len(res)}")
    print("   all 61 fit origins 1915-1975, v1.3 minus v1.1 final:",
          f"{np.mean([100 * np.mean([skill('v1.3', fo, m, 'final') - skill('v1.1', fo, m, 'final') for m in METRICS]) for fo in ALL_ORIGINS]):+.4f}")

    print("\n== Coefficients (beta, gamma) by fit origin ==")
    show = (1940, 1945, 1950, 1955, 1960, 1965, 1970, 1975)
    for cand in ("v1.1", "v1.2", "v1.3", "diag avg30+G+0.5W"):
        for m in METRICS:
            row = " ".join(f"{fo}:({coefs[(cand, fo, m)][0]:.2f},{coefs[(cand, fo, m)][1]:.2f})" for fo in show)
            b = np.median([coefs[(cand, fo, m)][0] for fo in FIT_ORIGINS])
            g = np.median([coefs[(cand, fo, m)][1] for fo in FIT_ORIGINS])
            print(f"   {cand:18s} {m:17s} {row}  | median over 36: ({b:.3f},{g:.3f})")
    print("   mean share of cities with a positive floored 30-year warming rate, per block:",
          {b: round(float(np.mean([pre[fo]["wpos"] for fo in fos])), 2) for b, fos in BLOCKS.items()})

    print("\n== Fit sample (v1.1): examples, unique target city-years, median reuse of a target city-year ==")
    for fo in (1940, 1975):
        print(f"   F={fo}: " + "; ".join(f"{m} {overlap_counts(tab, pre, m, fo)}" for m in METRICS))

    print("\n== Cities without a 30-year mean (G must be 0 there; adapter uses reference() fallbacks) ==")
    for fo in (1940, 1960, 1975):
        parts = []
        for m in METRICS:
            ref = pre[fo][m][0]
            miss = ~np.isfinite(ref)
            g_ok = all(np.all(shifts[(c, fo, m)][miss] == 0.0) for c in ("v1.1", "v1.2", "v1.3"))
            parts.append(f"{m} {int(miss.sum())} (G=0 there: {g_ok})")
        print(f"   F={fo}: " + "; ".join(parts))

    # Promotion analogue: triplets spaced like 1983/1990/1997, reward.py procedure, three resampling schemes
    lat, lon = coords.lat.to_numpy(float), coords.lon.to_numpy(float)
    schemes = {"city (official scheme)": None,
               "5-degree blocks": (np.floor(lat / 5) * 1000 + np.floor(lon / 5)).astype(int),
               "10-degree blocks": (np.floor(lat / 10) * 1000 + np.floor(lon / 10)).astype(int)}
    trips = [(f, f + 7, f + 14) for f in range(1940, 1962)]
    pairs = [(c, "average30") for c in names] + [("v1.2", "v1.1"), ("v1.3", "v1.1")]
    print(f"\n== Promotion analogue: {len(trips)} held-out triplets (F, F+7, F+14), F = 1940..1961, {DRAWS} draws, "
          f"seed {SEED} reset per triplet ==")
    for sname, labels in schemes.items():
        print(f"-- {sname}")
        for cand, champ in pairs:
            if labels is not None and cand.startswith(("diag", "ref")) and cand != "ref v0.5-like proxy (no G)":
                continue
            res = pd.DataFrame([compare(sums, cand, champ, trip, SEED, labels) for trip in trips])
            weakest = trips[int(res.p.idxmin())]
            print(f"   {cand:30s} vs {champ:9s} median reward {res.pts.median():+.4f}  median P {res.p.median():.3f}  "
                  f"min P {res.p.min():.4f} at {weakest}  median LB90 {res.lb.median():+.4f}  min LB90 {res.lb.min():+.4f}  "
                  f"worst floor {res.floor.min():+.4f}  2+ wins {int((res.wins >= 2).sum())}/{len(res)}  "
                  f"promote {int(res.promote.sum())}/{len(res)}  cities {res.n_cities.min()}-{res.n_cities.max()}")

    print("\n== Monte Carlo seeds for the weakest v1.1 triplet (1960, 1967, 1974), city scheme ==")
    trip = (1960, 1967, 1974)
    base = compare(sums, "v1.1", "average30", trip, SEED)
    half = 1.6449 * np.sqrt(base["p"] * (1 - base["p"]) / DRAWS)
    ps = np.array([compare(sums, "v1.1", "average30", trip, s)["p"] for s in range(1, 31)])
    print(f"   seed {SEED}: P {base['p']:.4f}, binomial 90% interval [{base['p'] - half:.4f}, {base['p'] + half:.4f}]; "
          f"seeds 1..30: min {ps.min():.4f} max {ps.max():.4f} mean {ps.mean():.4f}; P < 0.95 in {(ps < 0.95).sum()} of 30")
    for cand in ("v1.2", "v1.3"):
        res = pd.DataFrame([compare(sums, cand, "average30", t, SEED) for t in trips])
        tw = trips[int(res.p.idxmin())]
        ps = np.array([compare(sums, cand, "average30", tw, s)["p"] for s in range(1, 31)])
        print(f"   {cand} weakest {tw}: P {res.p.min():.4f}; seeds 1..30 min {ps.min():.4f}, P < 0.95 in {(ps < 0.95).sum()} of 30")

    # Runtime and determinism of one complete origin from a table cut at F (what an adapter would see)
    print("\n== End-to-end single origin from a table cut at F (load, pivot, graph terms, fits, forecast) ==")
    for fo in (1925, 1940, 1960, 1975):
        digests, digests13, times = [], [], []
        for _ in range(2):
            t0 = time.perf_counter()
            _, coords_f, tab_f = load(cut=fo)
            pre_f = precompute(tab_f, gauss_kernel(distances(coords_f)), order, range(fo - TRAIN_YEARS, fo + 1))
            out, out13 = [], []
            for m in METRICS:
                for cand in ("v1.1", "v1.2", "v1.3"):
                    _, base_kind, use_long, _ = CANDIDATES[cand]
                    c = fit_coefs(tab_f, pre_f, m, fo, base_kind, use_long)
                    assert c == coefs[(cand, fo, m)], f"cut-table coefficients differ at {fo} {m} {cand}"
                    fc, _, _ = forecast_at(pre_f, m, fo, cand, c)
                    for h in range(1, 8):
                        assert np.array_equal(fc[h], forecasts[(cand, fo, m)][h], equal_nan=True)
                        (out13 if cand == "v1.3" else out).append(fc[h])
            times.append(time.perf_counter() - t0)
            digests.append(hashlib.sha256(np.ascontiguousarray(np.array(out)).tobytes()).hexdigest()[:16])
            digests13.append(hashlib.sha256(np.ascontiguousarray(np.array(out13)).tobytes()).hexdigest()[:16])
        print(f"   F={fo}: {times[0]:.1f} s and {times[1]:.1f} s (v1.1+v1.2+v1.3); sha256 v1.1+v1.2 {digests[0]} / "
              f"{digests[1]}; v1.3 {digests13[0]} / {digests13[1]}; identical={digests == digests[::-1] and digests13 == digests13[::-1]}; "
              f"matches walk-forward: yes")


if __name__ == "__main__":
    main()
