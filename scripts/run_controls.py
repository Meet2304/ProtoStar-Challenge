"""Build (or refresh) the control panel: every pre-registered control plus combo and nested.

    uv run python scripts/run_controls.py            # development phase, official track
    uv run python scripts/run_controls.py --profile A   # five simple controls + incumbents only

Entries that are still current are reused; stale ones are rerun. Development phase only.
Prints the panel, then mirrors every panel run to MLflow (best effort).
"""
import argparse

import _harness  # noqa: F401  (sets the import path)
from _harness import DEFAULT_RUNS, DEFAULT_SUITE

from common import METRICS
from evaluation.panel import PROFILES, build_panel
from evaluation.tracking import try_log_run


def print_panel(panel):
    print(f"\nControl panel  {panel['control_set']}  suite {panel['suite_id'][:12]}  "
          f"{panel['phase']}  origins {panel['origins']}")
    head = f"{'entry':14s} {'role':9s} {'headline':>9s}  " + "  ".join(f"{m[:10]:>10s}" for m in METRICS) + "  by origin"
    print(head)
    print("-" * len(head))
    for entry_id, e in sorted(panel["entries"].items(), key=lambda kv: -kv[1]["headline_final_raw_skill"]):
        per = "  ".join(f"{e['final_skill_by_metric'][m]:+10.4f}" for m in METRICS)
        origins = " ".join(f"{v:+.3f}" for v in e["final_skill_by_origin"].values())
        print(f"{entry_id:14s} {e['role']:9s} {e['headline_final_raw_skill']:+9.4f}  {per}  {origins}")
    if panel.get("best_control"):
        print(f"\nBest control (excluding the average30 denominator): {panel['best_control']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--runs", default=DEFAULT_RUNS)
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    path, panel = build_panel(args.suite, args.runs, profile=args.profile, timeout=args.timeout)
    print_panel(panel)
    print(f"Panel: {path}")
    if not args.no_mlflow:
        for entry in panel["entries"].values():
            try_log_run(entry["run_dir"])


if __name__ == "__main__":
    main()
