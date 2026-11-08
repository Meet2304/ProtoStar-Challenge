"""Plan B r4: numerical check of the forecast filter's observation weights (no data read).

usage: uv run python docs/plans/creative/analysis/planB_gain_check.py
Weights are measured as the impulse response of planB_core.fixed_gain_level (it is linear in the observations).
Checks: (1) last-observation weight = K + (1-K)^n / n; (2) every weight <= 0.391097019 for K <= K_ss(0.25), n >= 10;
(3) weights sum to 1; (4) Sol's r3 cases, compared with the exact diffuse filter's gains.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import planB_core as C  # noqa: E402

K_MAX = C.steady_gain(C.RHO_BOUNDS[1])
BOUND = K_MAX + (1 - K_MAX) ** C.MIN_WINDOW_OBS / C.MIN_WINDOW_OBS


def weights(mask, K, drift=0.0):
    """Weight of each observed column in the final level (impulse response; NaN where unobserved)."""
    T = len(mask)
    base = np.where(mask, 0.0, np.nan)[None, :]
    zero = C.fixed_gain_level(base, K, np.array([drift]))[0]
    w = np.full(T, np.nan)
    for j in np.flatnonzero(mask):
        x = base.copy(); x[0, j] = 1.0
        w[j] = C.fixed_gain_level(x, K, np.array([drift]))[0] - zero
    return w


rng = np.random.default_rng(20261108)
worst = 0.0
for trial in range(3000):
    K = float(rng.uniform(C.steady_gain(C.RHO_BOUNDS[0]), K_MAX)) if trial % 3 else K_MAX
    mask = rng.random(C.WINDOW) < rng.uniform(0.35, 1.0)
    mask[-C.RECENT_YEARS:] |= np.arange(C.RECENT_YEARS) >= C.RECENT_YEARS - C.MIN_RECENT_OBS   # satisfy the recent rule
    n = int(mask.sum())
    if n < C.MIN_WINDOW_OBS:
        continue
    w = weights(mask, K, drift=float(rng.normal(0, 0.05)))
    last = np.flatnonzero(mask)[-1]
    assert abs(w[last] - (K + (1 - K) ** n / n)) < 1e-12, "last-observation formula"
    assert abs(np.nansum(w) - 1.0) < 1e-12, "weights sum to 1"
    worst = max(worst, float(np.nanmax(w)))
assert worst <= BOUND + 1e-12
print(f"K_ss(0.25) = {K_MAX:.9f}; bound K + (1-K)^10/10 = {BOUND:.9f}; worst weight over 3000 random windows = {worst:.9f}")

full = np.ones(10, bool)
print(f"10 consecutive observations, K = K_ss(0.25): last weight {weights(full, K_MAX)[-1]:.9f} (Sol: 0.391097019)")
sol = np.zeros(30, bool); sol[:9] = True; sol[-1] = True
print("Sol's window (9 early, gap, 1 final): fixed_gain_level returns",
      C.fixed_gain_level(np.where(sol, 0.0, np.nan)[None, :], K_MAX, np.array([0.0]))[0], "(NaN = fallback F2)")
print(f"exact diffuse filter at rho = 0.25: second-observation gain {(1 + 0.25) / (2 + 0.25):.6f}; "
      "Sol's window final gain 0.849410 (r3 review); the fixed-gain filter never uses these gains")
