"""Plan B r5, E9: full policy replay on the retrospective folds (plan sections 4.3 and 5.3), including the
section 4.3.5 diagnostic rescues, conditional versions and the frame-level skip rule.

usage: uv run python docs/plans/creative/analysis/planB_e8_replay.py [--out DIR]
Configurations are named "<parent>+<mod>..." (planB_e8_nested.MODS). A configuration with no stored run is computed
on demand by calling planB_e8_nested.py. Champion analogue starts as average30; carried-forward version as v0.5.
Comparisons use reward.py arithmetic (2,000 city draws, seed 20261108); promote needs P >= 0.95, >= 4/5 origin
wins (ceil(2/3 x 5)), live not clearly worse, every metric >= -2 points.
Rescue order when several diagnostics qualify: those that would promote first, then the higher 90% lower bound
against the champion, then the higher reward, then declaration order. At most one rescue per version.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CUTOFF = 1983
METRICS = ["annual_avg_temp", "total_precip_in", "hottest_day_temp", "cdd"]
SPLITS = ["live", "final"]
KEY = ["city_id", "year", "metric"]
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(Path(tempfile.gettempdir()) / "planB_e8"))
OUT = Path(ap.parse_args().out)
AVG = "average30"
# stored runs whose settings equal a composed name (same derive() changes, same evaluator)
ALIAS = {"v20+hot": "r21", "v20+hot+prcp": "r22", "v20+hot+core": "r23", "v20+hot+coreonly": "r24",
         "v20+hot+dep": "d03", "v20+hot+prcp+core": "r23p", "v20+hot+prcp+coreonly": "r24p",
         "v20+hot+prcpw25": "r21+prcpw25"}


def canon(c):
    return ALIAS.get(c, c)


def files(c, suffix=""):
    return sorted(OUT.glob(f"{canon(c)}-[0-9][0-9][0-9][0-9]{suffix}.csv"))


def ensure(c):
    if c in (AVG,) or files(c):
        return
    print(f"  ... computing {c} with planB_e8_nested.py", flush=True)
    subprocess.run([sys.executable, "-W", "ignore", str(HERE / "planB_e8_nested.py"), canon(c), "--out", str(OUT)],
                   check=True, stdout=subprocess.DEVNULL)


CACHE = {}


def mats(c):
    if c not in CACHE:
        ensure(c)
        r = pd.concat([pd.read_csv(p, dtype={"city_id": str}) for p in files(c)])
        assert r.year.max() <= CUTOFF
        CACHE[c] = r.groupby(["city_id", "origin", "split", "metric"])[["absolute_error", "baseline_absolute_error"]].sum()
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
    checks = {"final": bool(p >= 0.95 and fin[0] > 0), "origins": int((by_origin > 0).sum()) >= NEED,
              "live": float((live[1:] < 0).mean()) < 0.95, "floor": float(metric.min()) >= -2}
    return {"delta": float(fin[0]), "lo": float(np.quantile(fin[1:], 0.05)), "p": p, "wins": int((by_origin > 0).sum()),
            "checks": checks, "promote": all(checks.values()), "by_origin": by_origin}


def bound_and_reward(c, champ):
    return (0.0, 0.0) if c == champ else (compare(c, champ)["lo"], compare(c, champ)["delta"])


def identical_frames(a, b):
    ensure(a); ensure(b)
    pa = pd.concat([pd.read_csv(p, dtype={"city_id": str}) for p in files(a, "-pred")]).sort_values(KEY)
    pb = pd.concat([pd.read_csv(p, dtype={"city_id": str}) for p in files(b, "-pred")]).sort_values(KEY)
    return (len(pa) == len(pb) and pa[KEY].reset_index(drop=True).equals(pb[KEY].reset_index(drop=True))
            and np.array_equal(pa.predicted_value.to_numpy(), pb.predicted_value.to_numpy()))


state = {"champion": AVG, "carried": "v05"}
LOG = []


def judge(c):
    """Rules 1-2 of plan 4.3 against the current champion, without changing state."""
    r = compare(c, state["champion"])
    if r["promote"]:
        return "promote", r
    lo_k, d_k = bound_and_reward(state["carried"], state["champion"])
    return ("carry" if r["lo"] > lo_k and r["delta"] > d_k else "no"), r


def apply(c, outcome):
    if outcome == "promote":
        state["champion"] = state["carried"] = c
    elif outcome == "carry":
        state["carried"] = c


def step(version, config, parent, diags=(), skip_check=False):
    """Decide the version, then judge its diagnostics against the state AFTER that decision (plan 4.3.5).
    Returns the carried-forward configuration, which is the parent of the next version."""
    champ = state["champion"]
    if skip_check and identical_frames(config, parent):
        outcome, r = "skipped (frames identical)", None
    else:
        outcome, r = judge(config)
    LOG.append((version, config, champ, outcome, r))
    if outcome in ("promote", "carry"):
        apply(config, outcome)
    champ = state["champion"]
    qualified = []
    for k, (dname, dconf) in enumerate(diags):
        o, rd = judge(dconf)
        LOG.append((f"  diagnostic {dname}", dconf, champ, f"diagnostic: would {o}", rd))
        if o in ("promote", "carry"):
            qualified.append((o != "promote", -rd["lo"], -rd["delta"], k, dname, dconf))
    if qualified:
        dname, dconf = sorted(qualified)[0][4:]
        outcome2, r2 = judge(dconf)            # re-scored without the flag: same model, same numbers here
        LOG.append((f"{version.split()[0]}r rescue of {dname}", dconf, champ, outcome2, r2))
        apply(dconf, outcome2)
    return state["carried"]


# ---------------------------------------------------------------- the tree (plan section 5.3)
P = step("v2.0 assembly fix", "v20", "v05")
if P != "v20":
    sys.exit("v2.0 not carried forward: plan B stops")
P = step("v2.1 hottest-day level", P + "+hot", P, diags=[("hot-k1", P + "+hotk1")])
P = step("v2.2 precipitation base", P + "+prcp", P, diags=[("prcp-w75", P + "+prcpw25")])
P2 = P
P = step("v2.3 regional core", P2 + "+core", P2, diags=[("core-phi0", P2 + "+core+phi0"), ("core-phi1", P2 + "+core+phi1")])
core_dropped = False
if P == P2:
    P = step("v2.4 core-only (conditional)", P2 + "+coreonly", P2)
    core_dropped = P == P2
P3 = P
P = step("v2.5 departure learner", P3 + "+dep", P3, skip_check=True,
         diags=[("dep-absgate", P3 + "+depabs"), ("gate-only", P3 + "+gaterel")])
P4 = P

print(f"origins {list(map(int, origins))}, {len(cities)} cities; promote: P >= 0.95, wins >= {NEED}/{len(origins)}, "
      "live not clearly worse, metric floor -2")
for name, conf, champ, outcome, r in LOG:
    if r is None:
        print(f"{name:34s} [{canon(conf)}] {outcome}")
        continue
    print(f"{name:34s} [{canon(conf)}] vs {canon(champ)}: delta {r['delta']:+.6f} lo {r['lo']:+.6f} P {r['p']:.3f} "
          f"wins {r['wins']}/{len(origins)} live {r['checks']['live']} floor {r['checks']['floor']} -> {outcome}")
print(f"\ncore dropped: {core_dropped}; P3 = {canon(P3)}; P4 (parent of the v2.6 rehearsal) = {canon(P4)}")
for label, c in (("CHAMPION", state["champion"]), ("CARRIED-FORWARD", state["carried"])):
    a, b = compare(c, AVG), compare(c, "v05")
    print(f"{label} {canon(c)}: vs average30 {a['delta']:+.6f} (90% lo {a['lo']:+.3f}, P {a['p']:.3f}, wins {a['wins']}/5, "
          f"by origin {' / '.join(f'{v:+.2f}' for v in a['by_origin'])}); vs v0.5 {b['delta']:+.6f} "
          f"(lo {b['lo']:+.3f}, wins {b['wins']}/5)")
