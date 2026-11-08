"""Team evaluation CLI. Run `python code/evaluate.py --help` from the repo root."""
import argparse
import sys
from pathlib import Path

import pandas as pd

from common import ROOT, METRICS
from evaluation.core import (DEFAULT_CONFIG, prepare, verify_suite, run_suite,
                             read_json, write_json, compare_runs, validate_predictions, rubric_score)
from evaluation.report import render_report
from evaluation.protocol import create_freeze, arguments_from_spec

BASE = ROOT / "outputs/evaluation"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="build immutable pre-2019 seven-year holdouts")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    build.add_argument("--annual", type=Path, help="optional local annual_indices-schema parquet/CSV")
    build.add_argument("--cities", type=Path, help="optional local CSV with city_id")
    verify = commands.add_parser("verify", help="verify suite fingerprints")
    run = commands.add_parser("run", help="train/predict through a trusted team's Python adapter")
    run.add_argument("--adapter", help="module:function or path.py:function; default nested, or frozen settings")
    run.add_argument("--name", required=True)
    run.add_argument("--params", type=Path, help="JSON object passed as adapter keyword arguments")
    run.add_argument("--assets", type=Path, help="dependency manifest; required for custom adapters (explicit empty declaration allowed)")
    run.add_argument("--timeout", type=float, default=600, help="maximum seconds per fold")
    score = commands.add_parser("score", help="score externally trained predictions, with no training step")
    score.add_argument("--predictions", type=Path, required=True, help="directory of origin-YEAR.csv files")
    score.add_argument("--name", required=True)
    score.add_argument("--provenance", type=Path, required=True, help="author, model commit, parameters, dependencies and per-origin training cutoffs")
    freeze = commands.add_parser("freeze", help="pin a committed model before opening assessment scores")
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--params", type=Path)
    freeze.add_argument("--assets", type=Path, help="same dependency manifest used for the Python adapter")
    freeze.add_argument("--provenance", type=Path)
    freeze.add_argument("--note", default="")
    freeze.add_argument("--from-run", type=Path, help="reuse a verified development run's model settings")
    model = freeze.add_mutually_exclusive_group()
    model.add_argument("--adapter", help="default: evaluation.nested:forecast")
    model.add_argument("--baseline", choices=("average", "trend", "last_year"))
    model.add_argument("--predictions", type=Path, help="pin assessment CSV bytes and declared external provenance")
    commands.add_parser("baselines", help="evaluate all three challenge reference methods")
    for command in (build, verify, run, score, freeze, commands.choices["baselines"]):
        command.add_argument("--suite", type=Path, default=BASE / "suite-v3")
    for command in (run, score, commands.choices["baselines"]):
        command.add_argument("--phase", choices=("development", "assessment"), default="development")
        command.add_argument("--freeze", type=Path, help="required for assessment, from the freeze command")
    for command in (run, score, commands.choices["baselines"]):
        command.add_argument("--runs", type=Path, default=BASE / "runs")
    report = commands.add_parser("report", help="write a portable HTML report and CSV run table")
    report.add_argument("--runs", type=Path, default=BASE / "runs")
    report.add_argument("--output", type=Path, default=BASE / "report")
    report.add_argument("--suite-id", help="optional exact suite fingerprint to include")
    diagnose = commands.add_parser("diagnose", help="separate development-only station, CDD and precision sensitivities")
    diagnose.add_argument("--run", type=Path, required=True)
    diagnose.add_argument("--suite", type=Path, default=BASE / "suite-v3")
    compare = commands.add_parser("compare", help="paired raw-skill evidence and whole-period sensitivity")
    compare.add_argument("first", type=Path)
    compare.add_argument("second", type=Path)
    compare.add_argument("--samples", type=int, default=2000)
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--output", type=Path)
    check = commands.add_parser("check-submission", help="offline full 5,880-row forecast validation")
    check.add_argument("path", type=Path, nargs="?", default=ROOT / "submission.csv")
    rubric = commands.add_parser("rubric", help="apply official weights to your human-supplied review ratings")
    rubric.add_argument("--ratings", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.annual is None:
            print("Source: pinned CMU-OSPO training annual_indices; observations end 2018-12-31.")
        manifest = prepare(args.suite, args.config, args.annual, args.cities)
        print(f"Prepared {len(manifest['config']['origins'])} folds / {len(manifest['city_ids'])} cities.")
        print(f"Suite: {args.suite.resolve()}\nFingerprint: {manifest['suite_id']}")
    elif args.command == "verify":
        print(f"Verified suite: {verify_suite(args.suite)['suite_id']}")
    elif args.command in ("run", "score", "baselines"):
        settings = None
        if args.freeze and args.command in ("run", "baselines"):
            settings = arguments_from_spec(read_json(args.freeze)["model_spec"])
            if (args.command == "run") != (settings["adapter"] is not None):
                raise ValueError("use run for a frozen adapter or baselines for a frozen reference method")
        adapter = ((args.adapter or (settings["adapter"] if settings else "evaluation.nested:forecast"))
                   if args.command == "run" else None)
        params = (read_json(args.params) if args.params else (settings["params"] if settings else {})) if args.command == "run" else None
        assets = (args.assets or (settings["assets"] if settings else None)) if args.command == "run" else None
        variants = [("Last year repeated", "last_year"), ("30-year average", "average"),
                    ("Linear trend", "trend")] if args.command == "baselines" else [(args.name, None)]
        if args.command == "baselines" and settings:
            variants = [("Frozen " + settings["baseline"], settings["baseline"])]
        for name, baseline in variants:
            run_dir, meta = run_suite(args.suite, args.runs, name,
                adapter=adapter, params=params,
                prediction_dir=args.predictions if args.command == "score" else None,
                baseline=baseline, timeout=args.timeout if args.command == "run" else 600,
                phase=getattr(args, "phase", "development"), freeze=getattr(args, "freeze", None),
                provenance=read_json(args.provenance) if args.command == "score" else None,
                assets=assets)
            print(f"{name}: {meta['phase']}; mean final raw skill="
                  f"{meta['summary']['final']['mean_skill_unclipped']:.6f}; "
                  f"summary after one clip={meta['summary']['final']['selection_score']:.4f}")
            print(f"Run: {run_dir}")
    elif args.command == "freeze":
        adapter = args.adapter or (None if args.baseline or args.predictions or args.from_run else "evaluation.nested:forecast")
        record = create_freeze(args.suite, args.output, adapter=adapter,
            params=read_json(args.params) if args.params else {}, baseline=args.baseline,
            prediction_dir=args.predictions, provenance=read_json(args.provenance) if args.provenance else None,
            note=args.note, assets=args.assets, from_run=args.from_run)
        print(f"Committed model freeze: {record['freeze_id']}\nRecord: {args.output.resolve()}")
        print("Assessment access is logged. This does not make reused historical years untouched.")
    elif args.command == "report":
        path, table = render_report(args.runs, args.output, args.suite_id)
        print(f"Report: {path}\nComplete runs: {len(table)}")
    elif args.command == "diagnose":
        from evaluation.diagnostic_bundle import create_diagnostics
        print(f"Development sensitivity bundle: {create_diagnostics(args.run, args.suite)}")
    elif args.command == "compare":
        result = compare_runs(args.first, args.second, args.samples, args.seed)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
        import json
        print(json.dumps(result, indent=2))
    elif args.command == "check-submission":
        cities = sorted(pd.read_csv(ROOT / "data/city_info.csv", dtype={"city_id": str}).city_id.unique())
        targets = pd.MultiIndex.from_product([cities, range(2019, 2026), list(METRICS)],
                                              names=["city_id", "year", "metric"]).to_frame(index=False)
        validate_predictions(pd.read_csv(args.path, dtype={"city_id": str}), targets)
        print(f"OK: {len(targets):,} finite forecasts, complete city/year/metric grid; no network needed.")
    elif args.command == "rubric":
        print(f"Internal human self-review: {rubric_score(read_json(args.ratings)):.4f} / 100. "
              "This is not a predicted judges' score.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        sys.exit(1)
