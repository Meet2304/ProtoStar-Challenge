"""Plan B r3: the regional state-space core shared by every analysis script (and specified for the adapter).

Arrays are city x year panels `M` with first column year `y0`. Every function that takes a pseudo-origin `t`
reads columns <= t only (asserted by `window`). Constants are the plan's fixed-in-advance values (plan section 3.1).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar

NB_K = 15
BASE_YEARS, BASE_MIN = 60, 20
REG_MIN = 4
WINDOW, MIN_WINDOW_OBS = 30, 10
RECENT_YEARS, MIN_RECENT_OBS = 5, 3
MLE_YEARS, MIN_SERIES_YEARS, MIN_SERIES = 100, 30, 20
ANCHOR_STEP = 5
ANCHOR_BACKOFF = (0, 1, 2)          # multiples of ANCHOR_STEP
RHO_BOUNDS = (1e-5, 0.25)
KAPPA = {"annual_avg_temp": 1.0, "cdd": 0.5, "hottest_day_temp": 0.5}
PRCP_WEIGHT, PRCP_MIN_YEARS = 0.5, 40
LEADS = tuple(range(1, 8))


def nanmean(x, axis, minimum=1):
    ok = ~np.isnan(x)
    n = ok.sum(axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= minimum, np.where(ok, x, 0.0).sum(axis) / n, np.nan)


def neighbours(lat_deg, lon_deg, k=NB_K):
    """k nearest other cities by haversine distance; inputs sorted by city_id so ties break by city_id."""
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    a = np.sin((lat[:, None] - lat[None]) / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None] * np.sin((lon[:, None] - lon[None]) / 2) ** 2
    dist = 2 * np.arcsin(np.sqrt(np.minimum(1.0, a)))
    np.fill_diagonal(dist, np.inf)
    return np.argsort(dist, axis=1, kind="stable")[:, :k]


def window(M, y0, lo, hi, t):
    """Columns lo..hi (inclusive) as an array, NaN-padded; refuses to read past the pseudo-origin t."""
    assert hi <= t, f"read of year {hi} at pseudo-origin {t}"
    out = np.full((M.shape[0], hi - lo + 1), np.nan)
    a, b = max(lo, y0), min(hi, y0 + M.shape[1] - 1)
    if b >= a:
        out[:, a - lo:b - lo + 1] = M[:, a - y0:b - y0 + 1]
    return out


def ref30(M, y0, t):
    w = window(M, y0, t - 29, t, t)
    return np.where((~np.isnan(w)).sum(1) >= 10, nanmean(w, 1), np.nan)


def regional(M, y0, t, start, nbs):
    """c[i, start..t]: mean over self + 15 nearest of value - city mean over [t-59, t] (>= 20 valid), >= 4 members."""
    base = nanmean(window(M, y0, t - BASE_YEARS + 1, t, t), 1, BASE_MIN)
    an = window(M, y0, start, t, t) - base[:, None]
    return nanmean(an[nbs], 1, REG_MIN)


def ols_slope(w, years):
    ok = ~np.isnan(w)
    n = ok.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = np.where(ok, years, 0).sum(1) / n
        ym = np.where(ok, w, 0).sum(1) / n
        dx = np.where(ok, years - xm[:, None], 0.0)
        dy = np.where(ok, w - ym[:, None], 0.0)
        s = (dx * dy).sum(1) / (dx ** 2).sum(1)
    return np.where(n >= 10, s, np.nan), xm


def drift_slope(M, y0, t, nb):
    """Floored mean over self + 15 nearest of each city's OLS-30 slope (v0.3's regional rate)."""
    s, _ = ols_slope(window(M, y0, t - 29, t, t), np.arange(t - 29, t + 1, dtype=float)[None, :])
    return np.maximum(np.nan_to_num(nanmean(np.column_stack([s, s[nb]]), 1)), 0.0)


# ------------------------------------------------------------------ estimation (exact diffuse filter)
def exact_loglik(c, rho):
    """Local level, drift 0, r = 1 scaled, exact diffuse start (first obs sets L = y, P = 1, excluded).
    Returns the concentrated log-likelihood (r profiled out) and the number of terms."""
    n, T = c.shape
    L = np.zeros(n); P = np.zeros(n); started = np.zeros(n, bool)
    s_v2f = s_logf = 0.0; N = 0
    for k in range(T):
        P = np.where(started, P + rho, P)
        y = c[:, k]; ok = ~np.isnan(y)
        upd = ok & started
        F = P + 1.0; v = np.where(upd, y - L, 0.0); K = P / F
        s_v2f += float(np.sum(np.where(upd, v * v / F, 0.0)))
        s_logf += float(np.sum(np.where(upd, np.log(F), 0.0)))
        N += int(upd.sum())
        L = np.where(upd, L + K * v, L); P = np.where(upd, P * (1 - K), P)
        fresh = ok & ~started
        L = np.where(fresh, y, L); P = np.where(fresh, 1.0, P)
        started |= ok
    if N == 0 or s_v2f <= 0:
        return -math.inf, N
    r_hat = s_v2f / N
    return -0.5 * (N * math.log(2 * math.pi * r_hat) + s_logf + N), N


def fit_rho(M, y0, a, nbs, first_year):
    """Pooled bounded MLE of rho = q/r at anchor a from years [a-99, a]. Returns (rho or None, info)."""
    c = regional(M, y0, a, max(first_year, a - MLE_YEARS + 1), nbs)
    keep = (~np.isnan(c)).sum(1) >= MIN_SERIES_YEARS
    info = {"anchor": a, "n_series": int(keep.sum())}
    if keep.sum() < MIN_SERIES:
        return None, {**info, "status": "too_few_series"}
    if not np.nanvar(c[keep]) > 1e-12:
        return None, {**info, "status": "constant"}
    cc = c[keep]
    lo, hi = math.log(RHO_BOUNDS[0]), math.log(RHO_BOUNDS[1])
    res = minimize_scalar(lambda lr: -exact_loglik(cc, math.exp(lr))[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-3})
    if not (res.success and np.isfinite(res.fun)):
        return None, {**info, "status": "no_convergence"}
    at_bound = min(res.x - lo, hi - res.x) < 2e-3
    return math.exp(res.x), {**info, "status": "ok", "rho": math.exp(res.x), "loglik": -float(res.fun),
                             "at_bound": bool(at_bound)}


def anchors_for(t, step=ANCHOR_STEP):
    a = step * (t // step)
    return [a - b * step for b in ANCHOR_BACKOFF]


def steady_gain(rho):
    return (-rho + math.sqrt(rho * rho + 4 * rho)) / 2


# ------------------------------------------------------------------ forecasting (fixed-gain filter)
def fixed_gain_level(c, K, drift):
    """Level at the last column of a window c (n, T) with constant gain K and known drift.

    Initial level at column 0 = drift-adjusted mean of the window's n valid values; then for each column:
    L += drift (after column 0); if observed, L += K (y - L). The innovation gain is K at every update, but an
    observation also enters through the initial mean, so its total weight in L_t is
        K (1 - K)^a + (1 - K)^n / n,   a = number of valid observations after it.
    The maximum is the last observation's K + (1 - K)^n / n, increasing in K and decreasing in n; with K <= K_ss(0.25)
    and n >= MIN_WINDOW_OBS = 10 it is at most 0.391097019 (planB_gain_check.py verifies this numerically).
    Rows with fewer than MIN_WINDOW_OBS valid values or fewer than MIN_RECENT_OBS in the last RECENT_YEARS
    columns return NaN (fallback F2)."""
    n, T = c.shape
    ok = ~np.isnan(c)
    k = np.arange(T, dtype=float)[None, :]
    L = nanmean(np.where(ok, c - drift[:, None] * k, np.nan), 1)
    for j in range(T):
        if j > 0:
            L = L + drift
        L = np.where(ok[:, j], L + K * (np.nan_to_num(c[:, j]) - L), L)
    enough = (ok.sum(1) >= MIN_WINDOW_OBS) & (ok[:, -RECENT_YEARS:].sum(1) >= MIN_RECENT_OBS)
    return np.where(enough, L, np.nan)


def core_W(M, y0, t, rho, phi, metric, nbs, nb):
    """Correction W (cities x 7 leads) for annual_avg_temp, cdd (with drift phi) or hottest_day_temp (phi ignored).
    Returns (W, n_F2). rho None -> fallback F1 (all zeros)."""
    n = M.shape[0]
    if rho is None:
        return np.zeros((n, 7)), n
    p = phi if metric in ("annual_avg_temp", "cdd") else 0.0
    d = drift_slope(M, y0, t, nb) if p > 0 else np.zeros(n)
    win = regional(M, y0, t, t - WINDOW + 1, nbs)
    L = fixed_gain_level(win, steady_gain(rho), p * d)
    city = window(M, y0, t - WINDOW + 1, t, t)
    both = ~np.isnan(city) & ~np.isnan(win)
    cbar = np.where(both.sum(1) >= MIN_WINDOW_OBS, nanmean(np.where(both, win, np.nan), 1), np.nan)
    r = ref30(M, y0, t)
    W = np.zeros((n, 7))
    ok = np.isfinite(L) & np.isfinite(cbar) & np.isfinite(r)
    for h in LEADS:
        w = KAPPA[metric] * (L - cbar + p * d * h)
        W[:, h - 1] = np.where(ok & np.isfinite(w), w, 0.0)
    return W, int((~ok & np.isfinite(r)).sum())


def precip_W(M, y0, t):
    hist = window(M, y0, y0, t, t)
    n_all = (~np.isnan(hist)).sum(1)
    r = ref30(M, y0, t)
    w = np.where((n_all >= PRCP_MIN_YEARS) & np.isfinite(r), PRCP_WEIGHT * (nanmean(hist, 1) - r), 0.0)
    return np.repeat(np.nan_to_num(w)[:, None], 7, axis=1)


class RhoCache:
    """rho by (metric, anchor). Valid across fits because an anchor's inputs are the history cut at the anchor."""

    def __init__(self, step=ANCHOR_STEP):
        self.step, self.values, self.info = step, {}, {}

    def get(self, metric, M, y0, t, nbs, first_year):
        for a in anchors_for(t, self.step):
            key = (metric, a)
            if key not in self.values:
                self.values[key], self.info[key] = fit_rho(M, y0, a, nbs, first_year)
            if self.values[key] is not None:
                return self.values[key]
        return None
