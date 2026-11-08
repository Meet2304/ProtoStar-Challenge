"""Plan B r4, E8 policy replay: apply the plan's version policy (section 4.3 and the section 5.3 tree) to the
retrospective folds, including rejected versions, conditional versions and the skip rule.

usage: uv run python docs/plans/creative/analysis/planB_e8_replay.py [--out DIR]
Needs the row-error and prediction files written by planB_e8_nested.py for the configurations it visits; it stops
with a message if one is missing. Champion analogue starts as average30 (its errors are the baseline errors);
the carried-forward version starts as v0.5 (config v05). Comparisons use reward.py's arithmetic (2,000 city draws,
seed 20261108) with the promote rule adapted to 5 origins: wins >= 4 of 5 (ceil(2/3 x 5)).
"""
from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = 1983
METRICS = ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"]
SPLITS = ["live", "final"]
KEY = ["city_id", "year", "metric"]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "planB_e8"))
OUT = Path(ap.parse_args().out)
AVG = "average30"


def rows(c):
    files = sorted(OUT.glob(f"{c}-[0-9][0-9][0-9][0-9].csv"))
    if not files:
        sys.exit(f"missing E8 run for configuration {c!r}: run planB_e8_nested.py {c}")
    r = pd.concat([pd.read_csv(p, dtype={"city_id": str}) for p in files])
    assert r.year.max() <= CUTOFF
    return r


def preds(c):
    files = sorted(OUT.glob(f"{c}-[0-9][0-9][0-9][0-9]-pred.csv"))
    if not files:
        sys.exit(f"missing prediction frames for {c!r}: rerun planB_e8_nested.py {c}")
    return pd.concat([pd.read_csv(p, dtype={"city_id": str}).assign(origin=int(p.name.split("-")[1])) for p in files])


CACHE = {}


def mats(c):
    if c not in CACHE:
        r = rows("v05" if c == AVG else c)
        g = r.groupby(["city_id", "origin", "split", "metric"])[["absolute_error", "baseline_absolute_error"]].sum()
        CACHE[c] = g
    return CACHE[c]


base = mats("v05")
origins = sorted(base.index.get_level_values("origin").unique())
cities = sorted(base.index.get_level_values("city_id").unique())
cells = pd.MultiIndex.from_tuples([(o, s, m) for o in origins for s in SPLITS for m in METRICS])
REF = base.baseline_absolute_error.unstack([1, 2, 3]).reindex(index=cities, columns=cells).fillna(0).to_numpy()
rng = np.random.default_rng(20261108)
W = np.vstack([np.ones(len(cities)), rng.multinomial(len(cities), np.full(len(cities), 1 / len(cities)), size=2000).astype(float)])
NEED = max(2, math.ceil(2 / 3 * len(origins)))


def err(c):
    if c == AVG:
        return REF
    g = mats(c)
    assert np.allclose(g.baseline_absolute_error.unstack([1, 2, 3]).reindex(index=cities, columns=cells).fillna(0).to_numpy(), REF)
    return g.absolute_error.unstack([1, 2, 3]).reindex(index=cities, columns=cells).fillna(0).to_numpy()


def skills(c):
    return (1 - (W @ err(c)) / (W @ REF)).reshape(len(W), len(origins), 2, 4)


def compare(c, champ):
    s = skills(c)
    d = 100 * (s - skills(champ)).mean(axis=3)
    fin, live = d.mean(axis=1)[:, 1], d.mean(axis=1)[:, 0]
    by_origin = d[0, :, 1]
    metric = 100 * s[0, :, 1, :].mean(axis=0)
    p = float((fin[1:] > 0).mean())
    checks = {"final": p >= 0.95 and fin[0] > 0, "origins": int((by_origin > 0).sum()) >= NEED,
              "live": float((live[1:] < 0).mean()) < 0.95, "floor": float(metric.min()) >= -2}
    return {"delta": float(fin[0]), "lo": float(np.quantile(fin[1:], 0.05)), "p": p, "wins": int((by_origin > 0).sum()),
            "checks": checks, "promote": all(checks.values()), "by_origin": by_origin}


def bound_and_reward(c, champ):
    if c == champ:
        return 0.0, 0.0
    r = compare(c, champ)
    return r["lo"], r["delta"]


def identical_frames(a, b):
    pa, pb = preds(a).sort_values(["origin"] + KEY), preds(b).sort_values(["origin"] + KEY)
    same_grid = pa[["origin"] + KEY].reset_index(drop=True).equals(pb[["origin"] + KEY].reset_index(drop=True))
    return same_grid and np.array_equal(pa.predicted_value.to_numpy(), pb.predicted_value.to_numpy())


state = {"champion": AVG, "carried": "v05"}
log = []


def step(version, config, parent, skip_check=False):
    if skip_check and identical_frames(config, parent):
        log.append((version, config, parent, "skipped: prediction frames identical to the parent", None))
        return False
    r = compare(config, state["champion"])
    champ_before = state["champion"]
    if r["promote"]:
        state["champion"] = state["carried"] = config
        decision = "KEPT (promote; new champion)"
    else:
        lo_k, d_k = bound_and_reward(state["carried"], state["champion"])
        if r["lo"] > lo_k and r["delta"] > d_k:
            state["carried"] = config
            decision = f"carried forward (lo {r['lo']:+.3f} > {lo_k:+.3f}, delta {r['delta']:+.3f} > {d_k:+.3f})"
        else:
            decision = f"not carried (needs lo > {lo_k:+.3f} and delta > {d_k:+.3f})"
    log.append((version, config, parent, decision, (champ_before, r)))
    return state["carried"] == config


# ---------------------------------------------------------------- the r4 tree (plan section 5.3)
P = "v05"
if not step("v2.0 assembly fix", "v20", P):
    sys.exit("v2.0 not carried forward: plan B stops here (every later version needs the corrected gate)")
P = "v20"
P = "r21" if step("v2.1 hottest-day level", "r21", P) else P
prcp_cfg = {"r21": "r22", "v20": "v21"}[P]
P = prcp_cfg if step("v2.2 precipitation base", prcp_cfg, P) else P
core_cfg = {"r22": "r23p", "r21": "r23", "v21": "v22", "v20": "t21"}[P]
core_only_cfg = {"r22": "r24p", "r21": "r24", "v21": "v23", "v20": "t22"}[P]
if step("v2.3 regional core", core_cfg, P):
    P = core_cfg
elif step("v2.4 core-only (conditional)", core_only_cfg, P):
    P = core_only_cfg
dep_cfg = {"r21": "d03"}.get(P)
if dep_cfg is None:
    sys.exit(f"no E8 configuration declared for the departure learner on parent {P!r}")
P = dep_cfg if step("v2.5 temperature departure learner", dep_cfg, P, skip_check=True) else P

print(f"origins {list(map(int, origins))}, {len(cities)} cities, promote needs P >= 0.95, wins >= {NEED}/{len(origins)}, "
      "live not clearly worse, metric floor -2")
for version, config, parent, decision, info in log:
    if info is None:
        print(f"  {version:38s} [{config} on {parent}] {decision}")
        continue
    champ, r = info
    print(f"  {version:38s} [{config} on {parent}] vs champion {champ}: delta {r['delta']:+.4f} 90% lo {r['lo']:+.3f} "
          f"P {r['p']:.3f} wins {r['wins']}/{len(origins)} checks {r['checks']} -> {decision}")
end = state["carried"]
fin = compare(end, AVG)
vs05 = compare(end, "v05")
print(f"\nENDPOINT (policy-replayed): {end}; champion analogue {state['champion']}")
print(f"  vs average30: {fin['delta']:+.4f} points, 90% lo {fin['lo']:+.3f}, P {fin['p']:.3f}, wins {fin['wins']}/{len(origins)}, "
      f"by origin {' / '.join(f'{v:+.2f}' for v in fin['by_origin'])}")
print(f"  vs v0.5:      {vs05['delta']:+.4f} points, 90% lo {vs05['lo']:+.3f}, P {vs05['p']:.3f}, wins {vs05['wins']}/{len(origins)}, "
      f"by origin {' / '.join(f'{v:+.2f}' for v in vs05['by_origin'])}")
