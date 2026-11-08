# One-shot assessment: integrator only

Prepared for evaluator **eval-v1**, protocol `parent-audited-v6`, and locked
`suite-v3` ID
`4029caa1fb85cd376baf677e5d89549655684044aac68b6b3efb368eb6e46872`.
No finalist is selected by this document. No assessment freeze, prediction or
score was created during preparation. The metadata-only cohort inventory found
zero assessment-phase accesses when the seven official controls were sealed.

| Forecast origin | Live analogue | Final analogue |
|---|---|---|
| 2004 | 2005–2007 | 2008–2011 |
| 2011 | 2012–2014 | 2015–2018 |

Only the integrator executes assessment, **once for one chosen model**, in one
run covering both origins. Do not assess a shortlist, rerun controls or combo,
make a second suite to reset exposure, or retry automatically after an error.
The scorer already supplies the thirty-year-average reference denominator.
This is a team execution rule; the harness records exposure but is not an
authentication service or an OS sandbox.

## Before opening results

1. Use development origins 1983, 1990 and 1997 to choose the model. Record the
   chosen development run ID, parameters, asset declaration, training cutoffs,
   decision rule and team sign-off in a committed selection record. Decide how
   to interpret the assessment before opening it. Do not select a replacement
   model from the assessment results.
2. Merge the chosen model and its approved inputs through the integrator.
   Commit the exact model source and compact final input snapshots required by
   the asset manifest. Git-ignored assets cannot be frozen. Preserve release
   hashes and gate eligibility when promoting data; do not change input values.
   Port submission code into `code/` as required by the assembly-line plan.
3. If promotion changes source pins, adapter paths or assets, run the final
   implementation on **development only**, through `line.py score`, to create
   a matching sealed development run. This binds the implementation; it is not
   permission to tune on assessment. Its classification must be `official`, its
   runtime input summaries must require no review, and its files must verify.
4. Coordinate a quiet execution window. The worktree must be clean, the
   evaluator must still be the released `eval-v1`, the suite must match the
   lock, and installed packages must remain fixed after freezing. Evaluator
   changes are limited to bug fixes and at most one release per half-day.
5. Inventory assessment exposure across the original city-year cohort using
   metadata only. Also confirm there has been no manual/cache/test viewing of
   these results. Ledger counts are lower bounds, not proof of zero access.

Run the following **only as the integrator**, from its prepared worktree. Set
the chosen run ID first; the placeholder deliberately fails verification.

```powershell
$env:PROTOSTAR_CHOSEN_DEV_RUN = 'CHOSEN_DEVELOPMENT_RUN_ID'
uv run python scripts/line.py status
@'
import importlib.util, os, sys
from pathlib import Path
sys.path.insert(0, "code")
from evaluation.core import verify_run, read_json
from evaluation.exposure import assessment_inventory
from evaluation.protocol import source_pins, installed_environment
spec = importlib.util.spec_from_file_location("assembly_line", "scripts/line.py")
line = importlib.util.module_from_spec(spec); spec.loader.exec_module(line)
lock = line.read_lock()
assert not line.git("status", "--porcelain"), "Commit the final implementation"
assert line.evaluator_name(source_pins(), lock) == "eval-v1"
suite = read_json(Path("outputs/evaluation/suite-v3/manifest.json"))
assert suite["suite_id"] == lock["suite_id"]
assert suite["config"]["assessment_origins"] == [2004, 2011]
meta = verify_run(Path("outputs/evaluation/runs") / os.environ["PROTOSTAR_CHOSEN_DEV_RUN"])
assert meta["phase"] == "development" and meta["origins"] == [1983, 1990, 1997]
assert line.classify(meta, lock) == ("official", [])
assert meta["source_pins"] == source_pins(), "Rerun the final implementation on development"
assert meta["installed_environment"] == installed_environment(), "Use the verified development environment"
assert not any(v.get("requires_review") for v in meta.get("runtime_input_summary", {}).values())
inventory = assessment_inventory(Path("outputs/evaluation/runs"))
cohorts = [c for c in inventory["cohorts"] if c["suite_id"] == lock["suite_id"]]
assert len(cohorts) == 1
assert cohorts[0]["known_assessment_phase_accesses_across_suites"] == 0
assert inventory["accesses_with_unknown_cohort"] == 0
print("Metadata preflight passed; no assessment predictions or scores were opened.")
'@ | uv run python -
if ($LASTEXITCODE -ne 0) { throw 'Stop: assessment preflight failed' }
```

Investigate any warning or existing attempt with the team. Do not erase a
ledger, edit a sealed run, re-freeze a changed model, or reset the suite.

## Freeze, then execute once

These commands are prepared for the integrator; the eval station has **not**
run them. The freeze reuses the chosen development run's exact settings and
binds source, assets and installed packages. Keep the same environment through
execution. A freeze itself does not score assessment.

```powershell
uv run python code/evaluate.py freeze --suite outputs/evaluation/suite-v3 --from-run "outputs/evaluation/runs/$env:PROTOSTAR_CHOSEN_DEV_RUN" --output outputs/evaluation/freezes/one-shot-finalist.json --note "Integrator-approved one-shot assessment; one chosen model, both reserved origins"
if ($LASTEXITCODE -ne 0) { throw 'Stop: freeze failed' }
```

After final sign-off, reserve the **shared** attempt before launching. The
exclusive receipt prevents two cooperating integrators from launching the
documented procedure twice. Never delete it to obtain another attempt. It is a
coordination guard, not a complete defense against deliberate bypass.

```powershell
& {
    $ErrorActionPreference = 'Stop'
    New-Item -ItemType Directory -Path outputs/evaluation/assessment-once -Force | Out-Null
    $attemptPath = Join-Path (Resolve-Path outputs/evaluation/assessment-once) 'chosen-model.json'
    $attempt = [ordered]@{ chosen_development_run = $env:PROTOSTAR_CHOSEN_DEV_RUN; evaluator = 'eval-v1'; reserved_origins = @(2004,2011); started_utc = [DateTime]::UtcNow.ToString('o') } | ConvertTo-Json
    $attemptFile = [System.IO.File]::Open($attemptPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
    try { $attemptBytes = [System.Text.Encoding]::UTF8.GetBytes($attempt); $attemptFile.Write($attemptBytes, 0, $attemptBytes.Length) } finally { $attemptFile.Dispose() }
    uv run python scripts/line.py score --name chosen-model-assessment --suite outputs/evaluation/suite-v3 --phase assessment --freeze outputs/evaluation/freezes/one-shot-finalist.json
    if ($LASTEXITCODE -ne 0) { throw 'Stop: the attempt is consumed; ask the integrator to investigate, do not retry' }
}
```

One invocation covers both origins. Preserve the receipt, freeze, sealed run,
audits and access records even if it fails or scores poorly. An MLflow failure
does not justify rerunning the model: authenticate and use `line.py sync`.
Publish the analysis handoff through `line.py release`, never by handing off
working data folders.

## Interpret the result

Read raw final skill for each origin and metric, with the official clipped
scores alongside it. Positive raw skill beats the thirty-year reference;
negative raw skill is worse. Neither three development periods nor two reused
assessment periods justify a precise leaderboard rank or a formal confidence
claim. These historical labels influenced earlier team research; freezing does
not make them newly independent. The organizers' private answer key remains
unavailable. Never obtain or use observed 2019+ weather to select the model.
