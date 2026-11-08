# Independent evaluator audit, round 3 (Claude Opus 5.5)

Date: 2026-11-08. Reviewer: Claude Opus 5.5 (`claude-opus-5-5`), working alone. I changed only this
report and scratch files under `outputs/review-opus-round-3/`. I did not read any assessment score, and
did not use or query any observed 2019+ weather. Gemini and combined reviews were ignored.

**Reviewed source:** runtime fingerprint `11c52d1af8cb0a5e39cf3840504cfd089c722e11cb9d07035ddff9852e31ce1f`.
It matched `reports/evaluation-reviewed-source-round-3.json` before and after the review, using the
evaluator's LF-normalised `source_pins` (21 files). All 10 acceptance files also matched. Raw-byte
hashes differ for 12 runtime files only because this checkout has CRLF line endings (see I5).

**Live rule check:** Space API `sha` = pinned `44b8ec9f…`. Live `scoring.py`, `config.py`, `utils.py`,
`content/data.md` and `content/about.md` match the manifest hashes. No live-verification limitation.

## 1. Score: 81 / 100 (original report: 70)

| Dimension | Weight | Was | Now | Main reason |
|---|---:|---:|---:|---|
| Rule / scoring fidelity | 30 | 26 | **28** | Pure scorer pinned and live-identical; published numbers tested; wording fixed. The private key cannot be certified. |
| Statistical validation / test-set design | 30 | 15 | **21** | Ranking, comparison, nesting and phases fixed. Labels are reused, periods are few, and the gate has one hole. |
| Implementation robustness | 20 | 14 | **16** | Guards, assets and failure paths are strong. Undeclared reads and imports are not observed. |
| Reproducibility / provenance | 10 | 8 | **8** | Strong run records, but suite identity depends on line endings and the freeze ignores the installed environment. |
| Team usability | 10 | 7 | **8** | Documented paths work. Assessment baselines dead-end, and one documented freeze behaviour is wrong. |
| **Total** | 100 | 70 | **81** | |

| Use | Readiness |
|---|---|
| Exploration | **Ready.** |
| Comparative selection | **Ready with conditions:** development phase only, per-period raw deltas, and human review of each adapter's data reads. |
| Independent final evidence | **No, and not repairable.** Assessment years are reused, so a freeze is procedural discipline only. |
| Leaderboard rank | **No, by design.** Historical skill does not predict official Score, rank or judging. |

## 2. What I verified

| Claim | Verdict | Evidence |
|---|---|---|
| Guard set before adapter import | **Holds** | `worker.py:33-36` runs before the imports at `:51-60`; the import-time loader test passes. |
| No private-label sibling | **Holds** | Temp cwd holds only train, targets and params (`core.py:423-433`). |
| Rank on raw skill with tolerance | **Holds** | `core.py:530`; the report's rank basis is raw skill. |
| No four-year pseudo-CI | **Holds** | Period, metric and lead deltas only; interval needs ≥3 non-overlapping periods (`core.py:583-589`). |
| Nested selection uses earlier windows only | **Holds, also on real data** | Run `…b29b0edef9`: every inner fold has `train_max == origin` and `target_max <=` outer origin, e.g. outer 1997 uses inner 1976/1983/1990. Only 1983/1990/1997 are present. |
| Phase roles declared as reused | **Holds** | Dev 1983/1990/1997, assessment 2004/2011, `reused_historical`; zero real assessment access events. |
| Freeze pins model/source/params/seed/suite and logs access | **Holds** | CLI fixture; the ledger is written before scoring. See I2 and I4. |
| Custom adapters need declared assets | **Holds** | `protocol.py:104-107`. |
| Per-origin snapshot; cutoffs; real timestamps | **Holds** | `assets.py:317`, `:500`; R7 rejected a post-origin row through the CLI. |
| Committed blob / LFS oid+size | **Holds** | `_verify_committed`; `tests/test_assets.py`. |
| Unknown params, booleans, timeouts fail | **Holds** | `worker.py:68-71`, `nested.py:34-48`, `core.py:328`. |
| External CSV provenance and integrity | **Holds** | Original bytes hashed and re-checked (`core.py:404-409`); `verify_run` hashes artifacts. |
| Report metadata links self-contained | **Holds** | R8: five runs, five copied metadata files. |
| M5 combo params | **Holds** | Clean failure at `adapters.py:45-47` and `core.py:330`. |
| Suite and test state | **Passes** | `verify` OK (`d77f1e4f…`); 72 tests OK in 65.9 s; installed numpy, pandas, pyarrow and scikit-learn match `uv.lock`. |

`tests/test_asset_workflow.py` is a real end-to-end check: it copies current source, commits, and
drives prepare, run, freeze and assessment through the CLI, then rejects a changed asset and future
weights. It covers only assets the model *declares*, which is why it misses I1 and I3.

## 3. Findings

### Repairable defects

**I1. Assessment gate can be bypassed through the legacy suite (R6, high).** `--suite` with a schema-1
suite treats every origin as development (`protocol.py:30-33`). In the fixture this scored 2004 and
2011 with no freeze and no ledger entry. The real `suite-v1` holds origins 2004 and 2011, and README:37-39
documents the flag. *Fix:* refuse new runs on schema-1 suites for origins that are assessment in the
current config, unless an explicit flag writes a ledger event.

**I2. Import drift after a freeze goes undetected (R3, high).** An adapter in `labs/m/` was frozen as
a declared source asset. Its sibling `helper.py` is not pinned. After editing the helper, assessment
returned `complete` with different forecasts. `worker.py:52-58` runs the workspace adapter, with its
directory on `sys.path`, so any uncovered import can drift. Code under `code/` and `src/` is safe.
*Fix:* after the callback, list `sys.modules` files outside the environment, write
`imported_source.json`, and fail assessment if one is neither pinned nor declared as source.

**I3. Mislabelled label table accepted (R2, medium).** Declaring the annual table as `static_context`
scored 100 with status `complete`. Author classification is disclosed, but this mistake is cheaply
detectable. *Fix:* reject any declared file whose hash matches the pinned annual file, or that lies
under `data/raw/` or at `data/annual_valid.csv`; require `weather_input` for CSV/parquet with year or
date columns.

**I4. Freeze breaks on any later commit; the docs say otherwise (R5, medium).** `protocol.py:222`
requires HEAD to equal the freeze commit. A docs-only commit failed the assessment, while README:263-264
says unrelated documentation edits do not invalidate a freeze (true only for *uncommitted* edits).
Committing the freeze record would break it the same way. *Fix:* require the freeze commit to be an
ancestor of HEAD with unchanged pins and assets.

**I5. Suite identity depends on line endings (R9, medium).** The manifest stores the raw CRLF scorer
hash, and `verify_suite` failed against an LF copy of the scorer: `scorer changed`. `to_csv` also
writes `\r\n` here. A teammate on an LF checkout gets a different suite ID, so `compare` refuses and
shared runs (README:349) fall into separate groups. *Fix:* use `lineterminator="\n"` and normalised
scorer/city hashes in a new versioned suite.

**I6. Frozen baselines cannot be run (R4, medium).** `baselines --phase assessment` is rejected, and
`freeze --baseline` succeeds with no CLI command to use it. Assessment comparisons need baselines. The
undocumented `evaluation.adapters:reference` route works. *Fix:* add `--phase/--freeze` to `baselines`.

**I7. Repeated assessment is not visible (medium).** Each new freeze allows another logged run, but
`report.py` never reads the ledger (R8). *Fix:* show access count, freeze count and development trial
count per comparison group.

**I8. Freeze does not pin the installed environment (low-medium).** Only `uv.lock` bytes are pinned.
*Fix:* hash the installed distribution list in the freeze and verify it at assessment.

**I9. Scientific diagnostics still absent (low-medium).** Station-change exclusion and `Days Both`
sensitivity are disclosed as future work (README:420), and they are the known threats to the early folds.

**I10. Process (low).** The evaluator is uncommitted (23 dirty paths at the latest run), so real
freezes are currently impossible, and CI has never run (README:417).

### Disclosed limits, not defects

- **L1. Labels are reused.** Every target year has influenced team research. Nesting repairs the
  *procedure*, and the candidate set was still designed with those years in view.
- **L2. Few, non-stationary periods.** There are three development and two assessment periods, and the
  development periods are the least representative (station changes and mid-century baselines).
- **L3. Private key.** The private mask, denominators and post-2018 decimal precision are uncertifiable.
- **L4. Trust boundary (R1).** An adapter reading `DATA_DIR / "annual.csv"` scored 100 with status
  `complete` under an empty manifest. The docs state this boundary (README:205-210; `ASSETS.md:126-128`).
  It is a realistic accident, though, because `DATA_DIR` is the project's own helper. A non-sandbox
  tripwire would catch most cases: a `sys.addaudithook` in the worker that fails or records opens under
  `data/`, `outputs/evaluation/` and suite paths.
- **L5. Author declarations.** Weight cutoffs and external training claims are attested, not verified.

## 4. Why each deduction remains

| Dimension | Deductions |
|---|---|
| Rules −2 | L3 −1 (permanent); I9 `Days Both` and precision sensitivity −1. |
| Statistics −9 | L1 −4 and L2 −2 (permanent); I1 −1; I7 −1; I9 station-change filtering −1. |
| Robustness −4 | L4 tripwire −1.5; I2 −1; I3 −0.5; I4 −0.5; I8 −0.5. |
| Reproducibility −2 | I5 −1; I8 and I10 −1. |
| Usability −2 | I6 and the I4 doc error −1; workflow complexity (asset plus freeze ceremony) −1. |

## 5. Acceptance checks that would raise the score

| # | Check | Points |
|---|---|---:|
| 1 | Default config: a schema-1 run covering 2004/2011 fails unless explicitly overridden and logged. | +1 |
| 2 | Canary: an adapter opening `DATA_DIR/annual_valid.csv` fails or is flagged in `run.json`. | +1.5 |
| 3 | Freeze, edit an undeclared sibling import: assessment fails. | +1 |
| 4 | Declaring the annual table as `static_context` fails. | +0.5 |
| 5 | Freeze, make a docs-only commit: assessment passes; a source commit still fails. | +0.5 |
| 6 | Same input builds the same suite ID on Linux and Windows; CI runs green on both. | +1.5 |
| 7 | `baselines --phase assessment --freeze …` runs and is comparable. | +0.5 |
| 8 | Report shows access, freeze and trial counts. | +1 |
| 9 | `freeze` records and enforces the installed package hash. | +0.5 |
| 10 | Report per-period sensitivity with station-change and `Days Both` exclusions on development folds. | +2 |

Realistic ceiling without new independent labels: **about 90**. Items L1 to L3 cannot be repaired.

## 6. Commands run

```text
python -c <hash check>                         # raw hashes; 12 CRLF-only differences
python outputs/review-opus-round-3/fp.py       # normalised fingerprint, before and after
python outputs/review-opus-round-3/inv.py      # run inventory: phase/status only, no scores
curl -m 15 Space API and 5 raw source files    # metadata and hashes only
python -m unittest discover -s tests -v        # 150 s limit; 72 tests, OK, 65.9 s
python outputs/review-opus-round-3/repros.py   # R1-R9, synthetic fixture
python outputs/review-opus-round-3/nested_bounds.py  # development boundaries only
python code/evaluate.py verify
```

Fixture and output: `outputs/review-opus-round-3/r3-tw189li1/`, `repros.out.txt`. All Python commands used `.venv`.

**AI-use record.** Tool and model: Claude Code, Claude Opus 5.5, one agent. Tasks: source audit, one
full test run, synthetic repros, this report. A person should review I1, I2 and L4 first.
