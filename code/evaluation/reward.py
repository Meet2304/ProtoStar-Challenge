"""Reward for training agents: one number, a noise test and a champion ladder.

Read-only over sealed runs: it recomputes nothing the official scorer decides, so the evaluator
identity (eval-vN) does not change. Every number comes from each run's sealed ``row_errors.csv``
(development origins only, years <= 2004); no 2019-2025 observation is read or needed.

Reward = the host's leaderboard formula on our historical analogues, in leaderboard points:

    skill_m(origin, split) = 1 - MAE_m / MAE_m(30-year average)       (pooled over cities)
    points(split)          = 100 * mean over origins of mean_m skill_m   (unclipped)

``final`` (leads 4-7, the analogue of 2022-2025, which the final score uses) is the reward.
``live`` (leads 1-3, the analogue of 2019-2021) is reported and guarded.

Noise: cities are resampled with replacement (one draw shared by both runs, all origins, splits and
metrics), so each comparison is paired. With only three reused periods this measures sampling noise
across cities, not climate-regime risk; the "wins on >= 2 of 3 origins" rule covers the latter.

Ladder (Blum & Hardt, 2015): a run replaces the champion only when it is clearly better. Agents
optimise against the champion, so retrying noise does not ratchet the bar upward.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import METRICS  # noqa: E402
from evaluation.core import read_json, verify_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs/evaluation/runs"
SPLITS = ("live", "final")
SAMPLES = 2000
SEED = 20261108
PROMOTE_P = 0.95          # P(candidate better on final) needed to replace the champion
MIN_ORIGIN_WINS = 2       # ...and better on at least this many development origins
METRIC_FLOOR = -0.02      # same floor as screening gate G3; a single metric below it blocks promotion
REWARD_VERSION = "reward-v1"


class RewardError(ValueError):
    pass


def _rows(run: Path, origins) -> pd.DataFrame:
    frames = []
    for origin in origins:
        frame = pd.read_csv(run / f"origin-{origin}/row_errors.csv",
                            usecols=["city_id", "year", "metric", "split", "scored", "origin",
                                     "absolute_error", "baseline_absolute_error"])
        frames.append(frame[frame.scored.astype(bool)])
    rows = pd.concat(frames, ignore_index=True)
    if not np.isfinite(rows[["absolute_error", "baseline_absolute_error"]].to_numpy()).all():
        raise RewardError(f"{run.name}: a scored row has no finite error")
    return rows


class Run:
    """A sealed development run as city x cell sums, ready for weighted (bootstrap) pooling."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = verify_run(self.path)
        if self.meta.get("status") != "complete":
            raise RewardError(f"{self.path.name} is not complete")
        if self.meta.get("phase", "development") != "development":
            raise RewardError("the reward reads development runs only; assessment stays sealed")
        self.origins = list(self.meta.get("origins", self.meta["config"]["origins"]))
        self.name = self.meta.get("model") or self.path.name
        self.rows = _rows(self.path, self.origins)
        self.cells = [(o, s, m) for o in self.origins for s in SPLITS for m in METRICS]
        key = pd.MultiIndex.from_tuples(self.cells, names=["origin", "split", "metric"])
        grouped = self.rows.groupby(["city_id", "origin", "split", "metric"])
        sums = grouped[["absolute_error", "baseline_absolute_error"]].sum()
        self.cities = sorted(self.rows.city_id.unique())
        self.error = sums.absolute_error.unstack(["origin", "split", "metric"]).reindex(
            index=self.cities, columns=key).fillna(0.0).to_numpy()
        self.reference = sums.baseline_absolute_error.unstack(["origin", "split", "metric"]).reindex(
            index=self.cities, columns=key).fillna(0.0).to_numpy()

    def same_rows(self, other: "Run"):
        if self.meta["suite_id"] != other.meta["suite_id"] or self.origins != other.origins:
            raise RewardError("runs must share the suite and origins")
        a = self.rows.set_index(["city_id", "year", "metric"]).baseline_absolute_error.sort_index()
        b = other.rows.set_index(["city_id", "year", "metric"]).baseline_absolute_error.sort_index()
        if not a.index.equals(b.index) or not np.allclose(a.to_numpy(), b.to_numpy(), rtol=1e-12, atol=1e-12):
            raise RewardError("runs disagree on scored rows or the reference forecast")


def _skills(run: Run, weights: np.ndarray) -> np.ndarray:
    """weights: (B, cities) -> skills (B, cells). Pooled MAE ratio; counts cancel in the ratio."""
    reference = weights @ run.reference
    if (reference <= 0).any():
        raise RewardError("zero reference error in a cell")
    return 1.0 - (weights @ run.error) / reference


def _shape(run: Run, skills: np.ndarray) -> np.ndarray:
    return skills.reshape(skills.shape[0], len(run.origins), len(SPLITS), len(METRICS))


def summary(run: Run) -> dict:
    """Point estimate in leaderboard units. Matches each run's sealed scores.json exactly."""
    s = _shape(run, _skills(run, np.ones((1, len(run.cities)))))[0]
    out = {"run_id": run.path.name, "model": run.name, "reward_version": REWARD_VERSION}
    for j, split in enumerate(SPLITS):
        per_origin = s[:, j, :].mean(axis=1)
        out[split] = {
            "points": float(100 * per_origin.mean()),
            "host_score_by_origin": {str(o): float(100 * max(0.0, v)) for o, v in zip(run.origins, per_origin)},
            "points_by_origin": {str(o): float(100 * v) for o, v in zip(run.origins, per_origin)},
            "metric_points": {m: float(100 * s[:, j, k].mean()) for k, m in enumerate(METRICS)},
        }
    out["reward"] = out["final"]["points"]
    return out


def _weights(n: int, samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=samples).astype(float)


def compare(candidate: Run, champion: Run, samples=SAMPLES, seed=SEED) -> dict:
    """Paired city bootstrap of candidate minus champion, plus the ladder verdict."""
    candidate.same_rows(champion)
    if candidate.cities != champion.cities:
        raise RewardError("runs score different cities")
    weights = np.vstack([np.ones(len(candidate.cities)), _weights(len(candidate.cities), samples, seed)])
    delta = _shape(candidate, _skills(candidate, weights)) - _shape(champion, _skills(champion, weights))
    split_delta = 100 * delta.mean(axis=3)                       # (1+B, origins, splits)
    total = split_delta.mean(axis=1)                             # (1+B, splits)
    out = {"champion": champion.path.name, "champion_model": champion.name, "samples": samples, "seed": seed}
    for j, split in enumerate(SPLITS):
        draws = total[1:, j]
        out[split] = {"delta_points": float(total[0, j]),
                      "p_better": float((draws > 0).mean()),
                      "p_worse": float((draws < 0).mean()),   # ties (identical forecasts) count as neither
                      "interval_90_points": [float(v) for v in np.quantile(draws, [0.05, 0.95])],
                      "noise_sd_points": float(draws.std(ddof=1)),
                      "delta_by_origin": {str(o): float(split_delta[0, i, j]) for i, o in enumerate(candidate.origins)}}
    wins = sum(v > 0 for v in out["final"]["delta_by_origin"].values())
    metric_floor = summary(candidate)["final"]["metric_points"]
    worst = min(metric_floor, key=metric_floor.get)
    checks = {
        "final_clearly_better": out["final"]["p_better"] >= PROMOTE_P and out["final"]["delta_points"] > 0,
        "wins_most_origins": wins >= MIN_ORIGIN_WINS,
        "live_not_clearly_worse": out["live"]["p_worse"] < PROMOTE_P,
        "no_metric_below_floor": metric_floor[worst] >= 100 * METRIC_FLOOR,
    }
    if all(checks.values()):
        verdict = "promote"
    elif out["final"]["p_worse"] >= PROMOTE_P and out["final"]["delta_points"] < 0:
        verdict = "worse"
    else:
        verdict = "within_noise"
    out.update(origin_wins=wins, worst_metric=worst, checks=checks, verdict=verdict)
    return out


# ---------------------------------------------------------------- ladder
def _locked_suite():
    path = ROOT / "line.lock.json"
    return read_json(path).get("suite_id") if path.is_file() else None


def candidates(runs=RUNS, official_only=True):
    """Complete development runs on the locked suite, oldest first; official ones when line.py is importable."""
    suite = _locked_suite()
    classify = None
    if official_only:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("_line", ROOT / "scripts/line.py")
            line = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(line)
            lock = line.read_lock()
            classify = lambda meta: line.classify(meta, lock)[0]  # noqa: E731
        except Exception as exc:  # pragma: no cover - line.py always ships with the repo
            raise RewardError(f"cannot classify runs: {exc}") from exc
    found = []
    for path in sorted(Path(runs).glob("*/run.json")):
        meta = read_json(path)
        if meta.get("status") != "complete" or meta.get("phase", "development") != "development":
            continue
        if suite and meta.get("suite_id") != suite:
            continue
        if classify and classify(meta) != "official":
            continue
        found.append((meta["created_utc"], path.parent))
    return [p for _, p in sorted(found)]


def ladder(runs=RUNS, samples=SAMPLES, seed=SEED, official_only=True) -> dict:
    """Replay every eligible run in time order. Starts from the 30-year-average control (reward 0)."""
    paths = candidates(runs, official_only)
    loaded = [Run(p) for p in paths]
    starts = [r for r in loaded if r.name == "control:average30"]
    if not starts:
        raise RewardError("no official control:average30 run on the locked suite; run the controls first")
    champion, history = starts[0], []
    for run in loaded:
        row = summary(run)
        if run is champion:
            row.update(verdict="start", champion=champion.path.name)
        else:
            result = compare(run, champion, samples, seed)
            row.update(verdict=result["verdict"], champion=champion.path.name,
                       delta_points=result["final"]["delta_points"], p_better=result["final"]["p_better"],
                       origin_wins=result["origin_wins"])
            if result["verdict"] == "promote":
                champion = run
        history.append(row)
    return {"reward_version": REWARD_VERSION, "suite_id": _locked_suite(), "champion": champion.path.name,
            "champion_model": champion.name, "champion_reward": summary(champion)["reward"], "history": history}


def current_champion(runs=RUNS) -> Path:
    return Path(runs) / ladder(runs)["champion"]


# ---------------------------------------------------------------- CLI
def _resolve(value, runs):
    path = Path(value)
    if not path.is_dir():
        path = Path(runs) / value
    if not (path / "run.json").is_file():
        raise RewardError(f"no sealed run at {value}")
    return path


def explain(score: dict, result: dict | None) -> str:
    f, l = score["final"], score["live"]
    lines = [f"{score['model']}  ({score['run_id']})",
             f"  REWARD (final-analogue points): {score['reward']:+.3f}    live-analogue points: {l['points']:+.3f}",
             "  by metric (final): " + "  ".join(f"{m} {v:+.2f}" for m, v in f["metric_points"].items()),
             "  by origin (final): " + "  ".join(f"{o} {v:+.2f}" for o, v in f["points_by_origin"].items())]
    if result:
        fr = result["final"]
        lines += [f"  vs champion {result['champion_model']} ({result['champion']}):",
                  f"    final  delta {fr['delta_points']:+.3f} pts, 90% [{fr['interval_90_points'][0]:+.3f}, "
                  f"{fr['interval_90_points'][1]:+.3f}], P(better) {fr['p_better']:.2f}, noise sd {fr['noise_sd_points']:.3f}",
                  f"    live   delta {result['live']['delta_points']:+.3f} pts, P(better) {result['live']['p_better']:.2f}",
                  f"    origins won {result['origin_wins']}/{len(fr['delta_by_origin'])}; checks: "
                  + ", ".join(f"{k}={'yes' if v else 'NO'}" for k, v in result["checks"].items()),
                  f"  VERDICT: {result['verdict'].upper()}"]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", default=str(RUNS))
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="reward of one run, compared with the current champion")
    score.add_argument("run")
    score.add_argument("--against", help="run id or folder (default: the ladder's current champion)")
    score.add_argument("--samples", type=int, default=SAMPLES)
    score.add_argument("--json", action="store_true")
    show = sub.add_parser("ladder", help="replay the champion ladder over all official runs")
    show.add_argument("--samples", type=int, default=SAMPLES)
    show.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "ladder":
        board = ladder(args.runs, args.samples)
        if args.json:
            print(json.dumps(board, indent=2))
            return 0
        print(f"{'run':30s} {'model':38s} {'reward':>8s} {'live':>8s} {'d vs champ':>10s} {'P':>5s}  verdict")
        for row in board["history"]:
            d = f"{row['delta_points']:+.3f}" if "delta_points" in row else ""
            p = f"{row['p_better']:.2f}" if "p_better" in row else ""
            print(f"{row['run_id']:30s} {row['model'][:38]:38s} {row['reward']:+8.3f} {row['live']['points']:+8.3f} "
                  f"{d:>10s} {p:>5s}  {row['verdict']}")
        print(f"\nChampion: {board['champion_model']} ({board['champion']}), reward {board['champion_reward']:+.3f}")
        return 0

    run = Run(_resolve(args.run, args.runs))
    against = _resolve(args.against, args.runs) if args.against else current_champion(args.runs)
    result = None if against.resolve() == run.path.resolve() else compare(run, Run(against), args.samples)
    score_out = summary(run)
    if args.json:
        print(json.dumps({"score": score_out, "comparison": result}, indent=2))
    else:
        print(explain(score_out, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
