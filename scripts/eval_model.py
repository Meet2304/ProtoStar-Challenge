"""Evaluate a new model against the control panel in one command (development phase only).

    uv run python scripts/eval_model.py --name my-model-v1 --adapter code/my_model.py:forecast --assets evaluation/no-assets.json

Steps: verify the suite; make sure the control panel is complete and current; run the model on
the development origins (each origin sees only data up to that year and forecasts the next 7);
write its scorecard; compare it with combo (the method behind submission.csv) and with the best
control; print the screening gates; mirror to MLflow; refresh the HTML report.

Official input track only: gates are a model-choice signal, and the cleaned-input track is
barred from model choice. Nothing here touches submission.csv, the assessment origins, or any
data after 2018.
"""
import argparse
import json
from pathlib import Path

import _harness  # noqa: F401
from _harness import DEFAULT_REPORT, DEFAULT_RUNS, DEFAULT_SUITE

from common import METRICS
from evaluation.core import compare_runs, read_json, run_suite
from evaluation.panel import PANEL_ROOT, PROFILES, build_panel, gates, load_current_panel
from evaluation.report import render_report
from evaluation.scorecard import SCORECARD_DIR, write_scorecard
from evaluation.tracking import try_log_run


def fmt(value, spec="+.4f", missing="n/a"):
    return missing if value is None else format(value, spec)


def describe(name, comparison):
    lines = []
    periods = "  ".join(f"{p['origin']}:{p['raw_skill_delta']:+.4f}" for p in comparison["periods"])
    lines.append(f"  vs {name}: mean {comparison['mean_final_raw_skill_delta_second_minus_first']:+.4f}  "
                 f"(by origin {periods}; better on {comparison['positive_periods']}, worse on "
                 f"{comparison['negative_periods']})")
    metric = {m: sum(p["metric_skill_deltas"][m] for p in comparison["periods"]) / len(comparison["periods"])
              for m in METRICS}
    lines.append("     by metric " + "  ".join(f"{m} {v:+.4f}" for m, v in metric.items()))
    loo = comparison.get("leave_one_period_out_delta_points") or []
    if loo:
        lines.append("     leave one origin out (skill points): " + ", ".join(f"{v:+.2f}" for v in loo))
    interval = comparison.get("period_bootstrap_95_percent_sensitivity_interval_points")
    if interval:
        lines.append(f"     whole-period resampling range (points): {interval[0]:+.2f} to {interval[1]:+.2f}  "
                     "(descriptive sensitivity only, not a confidence interval)")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True, help="a short, unique model name, e.g. kohav-gbm-v3")
    parser.add_argument("--adapter", required=True, help="path.py:function or module:function")
    parser.add_argument("--assets", type=Path, help="dependency manifest (required for custom adapters; "
                                                    "evaluation/no-assets.json if the model reads no files)")
    parser.add_argument("--params", type=Path, help="JSON file of keyword arguments for the adapter")
    parser.add_argument("--profile", choices=PROFILES, default="full",
                        help="control panel: A = five simple controls + incumbents; full = A + ridge and RF")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per origin")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--panel-root", type=Path, default=PANEL_ROOT)
    parser.add_argument("--scorecards", type=Path, default=SCORECARD_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-report", action="store_true")
    args = parser.parse_args()

    params = read_json(args.params) if args.params else {}
    if not isinstance(params, dict):
        raise SystemExit("--params must be a JSON object")
    if params.get("input_track", "official") != "official":
        raise SystemExit("eval_model.py runs the official input track only: gates are a model-choice signal and "
                         "the cleaned-input track is diagnostic only (see docs/evaluation-baselines.md).")

    panel = load_current_panel(args.suite, args.runs, args.profile, panel_root=args.panel_root)
    if panel is None:
        print(f"Control panel ({args.profile}) missing or stale; building it (a few minutes, once per code change) ...")
        _, panel = build_panel(args.suite, args.runs, profile=args.profile, panel_root=args.panel_root)
    print(f"Running {args.name} on development origins {panel['origins']} ...")
    run_dir, meta = run_suite(args.suite, args.runs, args.name, adapter=args.adapter, params=params,
                              timeout=args.timeout, phase="development", assets=args.assets)
    card_path, card = write_scorecard(run_dir, args.scorecards)

    combo = panel["entries"]["combo"]
    best_id = panel["best_control"]
    vs_combo = compare_runs(Path(combo["run_dir"]), run_dir)          # candidate minus combo
    vs_best = compare_runs(Path(panel["entries"][best_id]["run_dir"]), run_dir)

    f = card["fidelity"]
    print(f"\n=== {args.name}  ({meta['run_id']}) ===")
    print(f"Headline (mean unclipped final skill, the ranking number): {f['headline_final_raw_skill']:+.4f}")
    print(f"Leaderboard formula per fold, mean: final {f['official_score_final_foldmean']:.2f}  "
          f"live {f['official_score_live_foldmean']:.2f}   (0-100; 0 = no better than the 30-year average)")
    print("Final skill by metric: " + "  ".join(f"{m} {fmt(card['per_metric']['final'][m])}" for m in METRICS))
    print("Final skill by origin: " + "  ".join(f"{o} {fmt(v)}" for o, v in card["stability"]["final_raw_skill_by_origin"].items()))
    print("Skill by lead:         " + "  ".join(f"{k}:{fmt(v, '+.3f')}" for k, v in card["horizon"].items()))
    print("Bias (/ ref MAE):      " + "  ".join(f"{m} {fmt(v['relative_to_reference_mae'], '+.3f')}" for m, v in card["bias"].items()))
    rob = card.get("robustness", {})
    if "status" not in rob:
        win = rob["city_metric_win_rate"]
        print(f"Robustness: beats the 30-year average on {fmt(None if win is None else 100 * win, '.0f')}% of "
              f"city-metrics; p90 error ratio {fmt(rob['p90_abs_error_ratio'], '.3f')}; worst-decile city skill "
              f"{fmt(rob['worst_decile_city_skill'], '+.3f')} ({rob['n_cities_excluded']} cities excluded)")
    print("\nPaired comparisons (positive = this model is better):")
    print(describe("combo", vs_combo))
    print(describe(f"{best_id} (best control)", vs_best))
    print("\nScreening gates (heuristics on 3 reused periods, not statistical tests):")
    results = gates(card, panel, vs_combo)
    for label, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label:58s} {detail}")
    if all(ok for _, ok, _ in results):
        print("All gates pass: discuss with the team before freezing ONE finalist for assessment "
              "(docs/evaluation-baselines.md).")
    print("Note: combo was tuned on years overlapping these origins, so G4 favours combo.")
    print(f"\nRun: {run_dir}\nScorecard: {card_path}")
    out = card_path.with_name(card_path.stem + ".comparisons.json")
    out.write_text(json.dumps({"profile": args.profile, "best_control": best_id, "vs_combo": vs_combo,
                               "vs_best_control": vs_best,
                               "gates": [{"gate": g, "pass": bool(ok), "detail": d} for g, ok, d in results]},
                              indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not args.no_mlflow:
        for entry in panel["entries"].values():
            try_log_run(entry["run_dir"], echo=lambda *_: None)
        if try_log_run(run_dir):
            print("Mirrored to MLflow.")
    if not args.no_report:
        path, _ = render_report(args.runs, args.report, meta["suite_id"])
        print(f"Report: {path}")


if __name__ == "__main__":
    main()
