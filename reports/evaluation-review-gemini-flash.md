# Independent Audit of the ProtoStar Internal Model-Evaluation Platform

**Auditor:** Gemini 3.8 Flash (Skeptical Independent Reviewer)  
**Date:** November 7, 2026  
**Repository:** `.` (repository root; migrated to the eval worktree)  
**Deliverable File:** `reports/evaluation-review-gemini-flash.md`  
**Upstream Target:** `https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence` (pinned at revision `44b8ec9fcb6d4ad6243e14c5f72dc831d79d2ce6`)  
**Dataset Target:** `CMU-OSPO/protostar-climate-train-to-2018` (pinned at revision `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`)

---

## Executive Summary & Scorecard

An independent, skeptical technical audit of Team Placeholders' internal evaluation platform was conducted. The platform's architecture, statistical design, scoring fidelity, implementation code, experiment artifacts, and test suite were systematically inspected and stress-tested with minimal repros.

### Transparent Score Breakdown (Total: 85 / 100)

| Dimension | Max Points | Awarded | Rationale |
|---|---:|---:|---|
| **Challenge-Rule & Scoring Fidelity** | 30 | **29** | Exact AST match to official scoring function; exact reproduction of missing-day and 30-year eligibility thresholds (including the 9 unscoreable city-metrics); strict 2018 date cutoff enforcement with no bypass. Deducted 1 pt solely because private truth and private mask cannot be independently verified from public assets. |
| **Statistical Validation & Test-Set Design** | 30 | **22** | Realistic 7-year expanding horizon matching official live (leads 1–3) and final (leads 4–7) splits. Deducted 8 pts due to: (1) heavy overlap with historical holdouts used to tune the existing `combo_model.py`; (2) decadal non-stationarity across folds (early folds penalize trend models); (3) average-score distortion from fold-level clipping; and (4) small-sample ($N=4$) bootstrap lead-exchangeability distortions. |
| **Implementation Robustness** | 20 | **17** | Solid subprocess worker isolation, deterministic suite hashing, strict prediction grid checks, timeout handling. Deducted 3 pts for proven defects: Windows cross-drive `os.path.relpath` crash in report generation, omission of repo root from `sys.path` in `worker.py` (preventing `src` package imports), and missing unit tests for timeout and CLI failure modes. |
| **Reproducibility & Provenance** | 10 | **9** | Bit-for-bit rebuild verified from pinned local cache; complete run metadata, code snapshots, and SHA256 fingerprints. Deducted 1 pt because `code_snapshot` only copies `code/`, omitting non-`code/` dependencies (e.g., `src/`). |
| **Team Usability** | 10 | **8** | Portable, dependency-free local HTML report with XSS escaping; clean CSV and Python adapter interfaces. Deducted 2 pts for sequential-only execution across folds and lack of automated residual visualization. |
| **Overall Score** | **100** | **85** | **High Confidence (0.95)** |

### Readiness Judgments by Use Case

1. **Exploratory Iteration — READY (High Confidence):** Excellent for rapid sanity checking, checking key formatting, verifying adapter execution, and debugging code before making submissions.
2. **Comparative Model Selection — CONDITIONALLY READY (Moderate Confidence):** Useful for relative ranking between candidates tested under identical conditions, provided the team evaluates models primarily on later folds (e.g., origin 2011) and raw unclipped skill rather than relying on early folds where climate trends differ.
3. **Final Evidence — NOT READY (Skeptical Caveat):** Historical holdouts from 1984–2018 cannot serve as untouched out-of-sample evidence because `combo_model.py` was already tuned on 1987–2018 backtests.
4. **Official Leaderboard-Rank Prediction — NOT READY (Methodological Limitation):** By design, no historical evaluation can predict official rank or score on the private 2019–2025 ground truth. Furthermore, the forecast score informs only 20% of the judges' rubric (under Scientific Value and Technical Creativity).

---

## 1. Challenge-Rule & Scoring Fidelity

### 1.1 Prohibition on 2019–2025 Observations for Model Selection
- **Official Space Policy:** `content/data.md` lines 124–131 explicitly state:
  > *"Can I score my own forecasts against 2019–2025 observations? The observations are public, so you could, but choosing a model by its score on them uses the answers, even if they never enter the model. Scoring model variants on 2019–2025 observations to choose or tune one: ❌ Not allowed. Selecting by the answers uses them. Scoring a frozen, committed forecast afterwards, for analysis: ✅ Allowed. The forecast is fixed; declare it in your README with the commit hash."*
- **Platform Compliance:** The internal evaluation platform adheres strictly to this policy. It caps all internal observation tables and labels at `2018-12-31`. All date assertions (`assert_no_post_cutoff`) raise `DataRuleViolation` with no override or bypass flag.

### 1.2 Live and Final Year Splits
- **Official Space Configuration:** `config.py` lines 47–48 define `LIVE_YEARS = range(2019, 2022)` (2019–2021, leads 1–3) and `FINAL_YEARS = range(2022, 2026)` (2022–2025, leads 4–7).
- **Platform Implementation:** `evaluation/config.json` sets `horizon: 7`, `live_leads: [1, 2, 3]`, and `final_leads: [4, 5, 6, 7]`. In `code/evaluation/core.py` line 142, `answer["split"] = np.where(answer.year <= origin + 3, "live", "final")`. This matches the official 3-year live / 4-year final split.

### 1.3 Missing Days & Eligibility Masking
- **Official Space Rules:** `content/about.md` lines 27 and 94 specify:
  1. A city-year is invalid if >10 days are missing for the variables the metric uses (`Miss Tmax` and `Miss Tmin` for temperature metrics and CDD; `Miss Prcp` for precipitation).
  2. The 10-day rule applies to the 30-year baseline window (1989–2018 for the official test).
  3. A city-metric requires $\ge 10$ valid years in the 30-year baseline window to have a baseline. The organizers noted that exactly 9 city-metrics (63 rows) lack a baseline in 1989–2018.
- **Platform Verification:**
  - `load_annual()` in `code/evaluation/core.py` (lines 101–111) checks `counts.le(10).all(axis=1)` for the required missing columns.
  - `thirty_year_mean()` in `code/common.py` (lines 68–72) enforces `window.notna().sum(axis=1) >= MIN_VALID_YEARS` (where `MIN_VALID_YEARS = 10`).
  - Empirical verification run on `data/annual_valid.csv` for the 1989–2018 window identified **exactly 9 city-metrics** lacking a baseline:
    - `USC00280734`: `annual_avg_temp`, `hottest_day_temp`, `cdd`
    - `USW00003856`: `annual_avg_temp`, `total_precip_in`, `hottest_day_temp`, `cdd`
    - `USW00014935`: `total_precip_in`
    - `USW00024121`: `total_precip_in`
  - This perfectly confirms arithmetic and masking fidelity to the official rules.

### 1.4 Pure Scorer AST Identity & Arithmetic
- `code/evaluation/official_score.py` was inspected against `evaluation/upstream/scoring.py`. AST parsing confirmed the pure scoring function `score()` and `ScoringError` are identical to the upstream implementation.
- Arithmetic is verified:
  $$\text{MAE}_m = \frac{1}{N} \sum_{i=1}^N |\hat{y}_i - y_i|, \quad S_m = 1 - \frac{\text{MAE}_m}{\text{MAE}_{\text{ref}, m}}, \quad \text{Score} = 100 \times \max\left(0, \frac{1}{4} \sum_{m=1}^4 S_m\right)$$
- Clipping occurs after averaging across the four metrics, exactly matching the challenge specification. Full precision is preserved throughout computation.

---

## 2. Statistical Validation & Test-Set Design

### 2.1 Reused Holdouts & Historical Tuning Leakage
- **Methodological Limitation (High):** In `code/combo_model.py` lines 10–12 and 40–42, the current model parameters were tuned across four backtest windows: `1987–1994`, `1995–2002`, `2003–2010`, and `2011–2018`.
- The evaluation platform's default suite (`suite-v1`) tests on:
  - Origin 1983: target `1984–1990`
  - Origin 1990: target `1991–1997`
  - Origin 1997: target `1998–2004`
  - Origin 2004: target `2005–2011`
  - Origin 2011: target `2012–2018`
- **Impact:** The internal testing set spans 1984–2018, which completely overlaps the historical data already used to design and select `combo_model.py`. The team cannot treat `suite-v1` results as fresh, out-of-sample evidence. Any model tuned to outperform `combo-v1` on `suite-v1` is at severe risk of overfitting historical noise.

### 2.2 Decadal Non-Stationarity & Score Disparity
- **Empirical Evidence from Runs:**
  Evaluating `Linear trend` and `combo-v1` across the 5 folds reveals extreme variation driven by historical climate trends:
  - **Origin 1983 (Target 1984–1990):** `combo-v1` final Score = `0.00` (unclipped skill: `-0.0374`). `Linear trend` final Score = `0.00` (unclipped skill: `-0.1413`).
  - **Origin 1990 (Target 1991–1997):** `combo-v1` final Score = `0.00` (unclipped skill: `-0.0286`). `Linear trend` final Score = `0.00` (unclipped skill: `-0.1631`).
  - **Origin 1997 (Target 1998–2004):** `combo-v1` final Score = `2.57` (unclipped skill: `+0.0257`).
  - **Origin 2004 (Target 2005–2011):** `combo-v1` final Score = `2.56` (unclipped skill: `+0.0256`).
  - **Origin 2011 (Target 2012–2018):** `combo-v1` final Score = `10.91` (unclipped skill: `+0.1091`).
- **Diagnosis:** In the 1950s–1980s, aerosol cooling and regional climate trends kept US temperatures relatively flat. Trend extrapolation in 1983 and 1990 had negative skill relative to the 30-year average. Post-1990 warming accelerated, making trend models highly superior by 2011–2018.
- **Impact:** Averaging scores across all 5 folds penalizes trend-based models that reflect modern greenhouse-driven warming. For the actual 2019–2025 forecast task, origin 2011 is far more representative of current climate dynamics than origin 1983.

### 2.3 Aggregate Score Clipping Distortions
- In `code/evaluation/core.py`, the fold score is clipped at 0: `Score = 100 * max(0, mean_skill)`.
- When calculating summary metrics across folds:
  - Mean of fold scores for `combo-v1`: $(0.0 + 0.0 + 2.573 + 2.557 + 10.912) / 5 = \mathbf{3.21}$.
  - Mean of raw unclipped skill: $(-0.0374 - 0.0286 + 0.0257 + 0.0256 + 0.1091) / 5 = \mathbf{+0.0189}$ (which corresponds to an aggregate score of $1.89$).
- **Impact:** Fold-level clipping masks negative performances in early folds, inflating the reported mean score by $+1.32$ points. Model selection must inspect `mean_skill_unclipped` and individual fold scores rather than relying solely on `final_mean_score`.

### 2.4 Paired Bootstrap Resampling Limitations
- In `compare_runs()` (`code/evaluation/core.py` lines 413–450), paired bootstrap is performed by sampling calendar years with replacement within each fold's 4-year final window (leads 4, 5, 6, 7).
- **Flaws & Limitations:**
  1. **Very Small Sample Size ($N=4$):** Resampling 4 items with replacement has only $4^4 = 256$ possible configurations per fold. Percentile bootstrap intervals on $N=4$ have poor asymptotic coverage.
  2. **Lead Time Distortion:** Leads 4, 5, 6, and 7 represent increasing forecast horizons. Sampling years with replacement allows draws like $[4, 4, 4, 5]$ (omitting long horizons 6 and 7). Because errors vary by lead time, this distorts the horizon error trajectory.
  3. **Serial Dependence:** Multi-year climate oscillations (ENSO, PDO) create serial dependence across years that independent resampling ignores.

---

## 3. Implementation Robustness & Provenance

### 3.1 Proven Defects Identified

#### Finding 1 (Medium Severity): Windows Cross-Drive `os.path.relpath` Crash in Report Generation
- **Evidence:** `code/evaluation/report.py` line 20:
  ```python
  row["metadata_url"] = quote(os.path.relpath(run / "run.json", output).replace(os.sep, "/"), safe="/:.")
  ```
- **Repro & Impact:** On Windows, if `--runs` and `--output` reside on different drives (e.g., repository on `D:\` and report output directed to `C:\temp\report`), `os.path.relpath` raises `ValueError: path is on mount 'D:', start on mount 'C:'`. This completely crashes report generation.
- **Fix:** In `report.py`, wrap in a try/except or check if drive mounts match:
  ```python
  try:
      rel = os.path.relpath(run / "run.json", output)
  except ValueError:
      rel = str((run / "run.json").resolve())
  row["metadata_url"] = quote(rel.replace(os.sep, "/"), safe="/:.")
  ```

#### Finding 2 (Low Severity): Subprocess Worker Path Omission
- **Evidence:** `code/evaluation/worker.py` line 16:
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  ```
- `parents[1]` resolves to `<repo>/code`. The repository root (`ROOT`) is never added to `sys.path`.
- **Impact:** If a model adapter attempts to import from `src.protostar_challenge` (a top-level package defined in the project), the subprocess worker fails with `ModuleNotFoundError: No module named 'src'`.
- **Fix:** In `worker.py`, also insert `parents[2]` (the repository root):
  ```python
  sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
  ```

#### Finding 3 (Low Severity): Incomplete Code Snapshotting Outside `code/`
- **Evidence:** `code/evaluation/core.py` lines 311–314:
  ```python
  for source in (ROOT / "code").rglob("*.py"):
      dest = run / "code_snapshot" / source.relative_to(ROOT / "code")
  ```
- **Impact:** Only Python files under `code/` are snapshotted into the run directory. Any utility code, feature pipelines, or configuration files in `src/` or elsewhere are not captured in the provenance snapshot.
- **Fix:** Expand snapshot collection to include `(ROOT / "src").rglob("*.py")`.

### 3.2 Robustness Strengths Verified
- **Deterministic Bit-for-Bit Rebuild:** Running `prepare()` in a scratch directory against cached parquet data produced the identical SHA256 suite ID (`6080a20846b98ae099bb23034271bc8ab198e510e560e68f943326ec595c4608`).
- **Strict Prediction Grid Validation:** `validate_predictions()` rigorously verifies that all 5,880 rows exist, are finite, have no duplicates, and match the target index exactly. Omitted unscored rows are caught and rejected.
- **Process Isolation & Thread Limits:** Subprocess invocation with `PYTHONHASHSEED`, `OMP_NUM_THREADS="1"`, `OPENBLAS_NUM_THREADS="1"`, and `MKL_NUM_THREADS="1"` ensures consistent CPU execution.
- **Failure Resilience:** Adapters returning NaNs, crashing, or timing out are safely captured: `run.json` records `status: "failed"` with full tracebacks, and failed runs are automatically excluded from the leaderboard and HTML report.

---

## 4. Test Suite Analysis & Missing Tests

### 4.1 Current Test Coverage
The 19 tests in `tests/test_evaluation.py` pass and cover:
- Rubric normalization and AST parity with upstream scoring.
- Metric weighting equality and aggregate score clipping.
- 10-day missing day thresholds and 10-year baseline validity thresholds.
- Date cutoff enforcement (pre-2019).
- Full prediction grid requirements and duplicate rejection.
- Suite tampering detection and file hashing.
- End-to-end baseline and combo adapter execution.
- Basic XSS escaping in HTML reporting.

### 4.2 Important Missing Tests
1. **Worker Timeout Enforcement:** No test currently checks that a hung adapter is terminated when `timeout` is exceeded and logged as failed. (Verified in scratch repro `outputs/review-gemini/test_timeout.py`).
2. **Invalid Adapter Return Types / NaNs:** No unit test verifies that an adapter returning NaNs or non-DataFrames triggers a clean run failure. (Verified in scratch repro `outputs/review-gemini/test_nan_adapter.py`).
3. **Cross-Drive File Path Handling:** No test exercises `render_report()` across different drive letters on Windows.
4. **Zero Reference MAE Guard:** No test asserts that `build_fold()` raises `ValueError` if a synthetic baseline produces zero MAE.
5. **CLI Main Execution & Non-Zero Exit Codes:** Tests only invoke functions directly; CLI execution via `main()` is untested in the unit test suite.

---

## 5. Prioritized Action Plan & Recommendations

### Phase 1: Fixes Required Before Team Reliance (Immediate)
1. **Fix Cross-Drive Path Crash:** Patch `code/evaluation/report.py` line 20 to handle drive mount mismatches gracefully.
2. **Add Repo Root to Worker `sys.path`:** Add `parents[2]` to `code/evaluation/worker.py` so adapters can import project packages (such as `src/protostar_challenge`).
3. **Add Missing Robustness Tests:** Add tests to `tests/test_evaluation.py` covering timeout termination, invalid adapter outputs, and CLI error exits.

### Phase 2: Methodological Improvements for Model Selection (Later)
1. **Focus Selection on Recent Folds:** Rely primarily on origin 2011 (and 2004) for model selection decisions rather than equal-weighting across all 5 folds, as recent decades better reflect current climate trends.
2. **Track Raw Unclipped Skill:** Use `mean_skill_unclipped` rather than clipped `mean_score` when comparing close model variants to avoid clipping distortions.
3. **Refine Bootstrap Resampling:** For paired model comparison, consider block-bootstrap across folds or residual perturbation to preserve the 4-year forward horizon structure.
4. **Expand Provenance Snapshotting:** Include `src/` and environment lockfiles in the run code snapshots.

---

## 6. Audit Metadata & Verification Log

### Pinned Upstream Sources Checked
- **Space:** `CMU-OSPO/protostar-challenge-climate-risk-intelligence`
  - URL: `https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence`
  - Space SHA Verified: `44b8ec9fcb6d4ad6243e14c5f72dc831d79d2ce6`
  - Pinned Upstream Manifest: `evaluation/upstream/manifest.json` (all 6 files SHA256 verified)
- **Dataset:** `CMU-OSPO/protostar-climate-train-to-2018`
  - Revision Verified: `bbbdebdab2a18e00f97ef87bd3f35a71e222180b`
  - Cache Verified: `data/raw/annual_indices/train-00000-of-00001.parquet` (SHA256: `0b5f5fa9...`)

### Commands Executed During Audit
```bash
# 1. Existing test suite verification
uv run python -m unittest discover -s tests -v

# 2. Remote Space SHA confirmation
uv run python -c "import urllib.request, json; print(json.loads(urllib.request.urlopen('https://huggingface.co/api/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence').read()).get('sha'))"

# 3. Baseline eligibility threshold check (reproducing 9 missing city-metrics)
uv run python outputs/review-gemini/check_folds.py

# 4. Per-fold score and unclipped skill inspection across all existing runs
uv run python outputs/review-gemini/inspect_runs.py

# 5. Paired bootstrap sensitivity check
uv run python code/evaluate.py compare outputs/evaluation/runs/20261107T235348Z-d0874d94e1 outputs/evaluation/runs/20261107T235353Z-0c44618cdb --samples 200

# 6. Offline submission format validation
uv run python code/evaluate.py check-submission

# 7. Deterministic suite rebuild check
uv run python -c "from evaluation.core import prepare, read_json; manifest=prepare('outputs/review-gemini/suite-scratch'); print('Match:', manifest['suite_id'] == read_json('outputs/evaluation/suite-v1/manifest.json')['suite_id'])"

# 8. Timeout handling test
uv run python outputs/review-gemini/test_timeout.py

# 9. NaN return handling test
uv run python outputs/review-gemini/test_nan_adapter.py
```
