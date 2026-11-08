"""Dependency-free, portable local report. No server or network calls."""
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

from evaluation.core import leaderboard, read_json
from evaluation.protocol import PROTOCOL_VERSION


def collect_exposure_counts(runs, suite_id=None):
    """Summarize known trials and assessment access, without reading any scores.

    Counts are scoped to suite and protocol, not model-name/search filters. The
    ledger and run copies describe the same access, so a run ID counts once.
    Missing or unreadable ledgers leave totals unknown; the known counts remain
    explicit lower bounds. This inventory cannot detect deleted/unshared runs
    or somebody opening labels outside the evaluator.
    """
    groups, run_index, suite_paths, warnings = {}, {}, {}, []

    def group_for(suite, protocol):
        key = (suite, protocol)
        if key not in groups:
            groups[key] = {"suite_id": suite, "protocol_version": protocol,
                           "development_runs": {}, "attempts": set(),
                           "accesses": set(), "freezes": {}, "ledger_paths": set(),
                           "ledger_event_records": 0, "warnings": []}
        return groups[key]

    def freeze_for(meta, run):
        found = set()
        direct = meta.get("freeze_id")
        if isinstance(direct, str) and direct:
            found.add(direct)
        embedded = meta.get("freeze")
        if isinstance(embedded, dict) and embedded.get("freeze_id"):
            found.add(str(embedded["freeze_id"]))
        path = run / "freeze.json"
        if path.is_file():
            try:
                record = read_json(path)
                if isinstance(record, dict) and record.get("freeze_id"):
                    found.add(str(record["freeze_id"]))
            except (OSError, ValueError) as exc:
                warnings.append(f"Unreadable freeze metadata for {run.name}: {type(exc).__name__}")
        return found

    for path in sorted(Path(runs).glob("*/run.json")):
        try:
            meta = read_json(path)
        except (OSError, ValueError) as exc:
            warnings.append(f"Unreadable run metadata at {path.parent.name}: {type(exc).__name__}")
            continue
        suite = meta.get("suite_id") if isinstance(meta, dict) else None
        if isinstance(meta, dict) and meta.get("phase", "development") == "development":
            snapshot = path.parent / "suite_manifest.json"
            if snapshot.is_file():
                try:
                    from evaluation.phase_policy import protected_development_years
                    manifest = read_json(snapshot)
                    config = manifest.get("config", {})
                    if "development_origins" not in config:
                        manifest = {**manifest, "config": {**config, "development_origins": config.get("origins", [])}}
                    overlap = protected_development_years(manifest)
                    if overlap:
                        warnings.append(f"Cross-suite exposure: development run {meta.get('run_id', path.parent.name)} "
                                        f"in suite {suite} includes reserved assessment target years {overlap}. "
                                        "A new suite or freeze cannot undo that exposure.")
                except (OSError, ValueError, TypeError):
                    warnings.append(f"Cannot inventory city-year overlap for {path.parent.name}.")
        if not isinstance(suite, str) or not suite or (suite_id and suite != suite_id):
            continue
        protocol = str(meta.get("protocol_version", "legacy-v1-unprotected"))
        group = group_for(suite, protocol)
        run_id = str(meta.get("run_id", path.parent.name))
        run_index.setdefault((suite, run_id), set()).add(protocol)
        suite_path = meta.get("suite_path")
        if isinstance(suite_path, str) and suite_path:
            suite_paths.setdefault(suite, set()).add(Path(suite_path))
        phase = meta.get("phase", "development")
        if phase == "development":
            status = str(meta.get("status", "unknown"))
            earlier = group["development_runs"].get(run_id)
            if earlier is not None and earlier != status:
                group["warnings"].append(f"Conflicting status copies for development run {run_id}.")
                status = "conflicting"
            group["development_runs"][run_id] = status
        elif phase == "assessment":
            group["attempts"].add(run_id)
            freezes = freeze_for(meta, path.parent)
            group["freezes"].setdefault(run_id, set()).update(freezes)
            if meta.get("assessment_access_record"):
                group["accesses"].add(run_id)

    ledger_state = {}
    for suite in {key[0] for key in groups}:
        paths = suite_paths.get(suite, set())
        available, failures = [], []
        for path in sorted(paths, key=str):
            folder = path / "assessment_access"
            if not folder.is_dir():
                failures.append(f"Assessment ledger unavailable: {folder}")
                continue
            try:
                files = sorted(p for p in folder.iterdir() if p.suffix == ".json" and p.is_file())
                available.append(folder)
            except OSError:
                failures.append(f"Assessment ledger unreadable: {folder}")
                continue
            for event_path in files:
                try:
                    event = read_json(event_path)
                    if not isinstance(event, dict) or event.get("suite_id") != suite or not event.get("run_id"):
                        raise ValueError("invalid suite/run identity")
                except (OSError, ValueError):
                    failures.append(f"Invalid assessment ledger entry: {event_path.name}")
                    continue
                run_id = str(event["run_id"])
                protocol = event.get("protocol_version")
                if not protocol:
                    candidates = run_index.get((suite, run_id), set())
                    protocol = next(iter(candidates)) if len(candidates) == 1 else "unattributed-protocol"
                group = group_for(suite, str(protocol))
                group["ledger_paths"].add(str(folder))
                group["ledger_event_records"] += 1
                group["attempts"].add(run_id)
                group["accesses"].add(run_id)
                freeze_id = event.get("freeze_id")
                if isinstance(freeze_id, str) and freeze_id:
                    group["freezes"].setdefault(run_id, set()).add(freeze_id)
                if protocol == "unattributed-protocol":
                    group["warnings"].append("Some ledger accesses lack protocol attribution; retained separately.")
        if not paths:
            failures.append("Suite location is absent from known run metadata; ledger availability is unknown.")
        ledger_state[suite] = (available, failures)

    summaries = []
    for (suite, protocol), group in sorted(groups.items()):
        available, failures = ledger_state.get(suite, ([], ["Suite ledger availability is unknown."]))
        # A ledger without protocol attribution cannot establish exact totals
        # for any particular protocol, even though its observed events survive.
        unattributed = (suite, "unattributed-protocol") in groups
        complete = bool(available) and not failures and not unattributed and not warnings
        status = "available" if complete else "incomplete" if available else "unavailable"
        known_freezes = set().union(*(group["freezes"].get(run_id, set()) for run_id in group["accesses"]))
        freeze_counts = Counter()
        unresolved = 0
        for run_id in group["accesses"]:
            identities = group["freezes"].get(run_id, set())
            if len(identities) == 1:
                freeze_counts[next(iter(identities))] += 1
            else:
                unresolved += 1
        repeats = sum(max(0, count - 1) for count in freeze_counts.values())
        development_status_counts = Counter(group["development_runs"].values())
        summary = {"suite_id": suite, "protocol_version": protocol,
                   "known_development_trials": len(group["development_runs"]),
                   "development_status_counts": dict(development_status_counts),
                   "known_assessment_attempts": len(group["attempts"]),
                   "known_assessment_accesses": len(group["accesses"]),
                   "known_unique_freezes": len(known_freezes),
                   "known_repeated_assessment_accesses": repeats,
                   "accesses_without_unique_freeze": unresolved,
                   "assessment_accesses": len(group["accesses"]) if complete else None,
                   "unique_freezes": len(known_freezes) if complete and not unresolved else None,
                   "ledger_status": status, "ledger_event_records": group["ledger_event_records"],
                   "warnings": sorted(set(group["warnings"] + failures))}
        if unresolved:
            summary["warnings"].append("Some known access run IDs have missing or conflicting freeze identities.")
        summaries.append(summary)
    return {"groups": summaries, "warnings": warnings}


def render_report(runs, output, suite_id=None):
    runs, output = Path(runs).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = leaderboard(runs, suite_id)
    table.to_csv(output / "leaderboard.csv", index=False)
    exposures = collect_exposure_counts(runs, suite_id)
    (output / "exposure_counts.json").write_text(json.dumps(exposures, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    records = []
    for row in table.to_dict(orient="records"):
        run = runs / row["run_id"]
        scores = read_json(run / "scores.json")
        row["scores"] = scores
        # Copy linked metadata into the report package. Relative paths across
        # Windows drives are undefined, and links into another machine's run
        # directory would not survive sharing the report folder.
        name = row["run_id"]
        if not name or Path(name).name != name or "\\" in name:
            raise ValueError("run ID must be a single directory name")
        metadata_dir = output / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        (metadata_dir / f"{name}.json").write_bytes((run / "run.json").read_bytes())
        row["metadata_url"] = quote(f"metadata/{name}.json", safe="/:.")
        records.append(row)
    payload = json.dumps(records, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    exposure_payload = json.dumps(exposures, allow_nan=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    document = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ProtoStar · Internal evaluation</title>
<style>
:root{--ink:#17313b;--muted:#54686c;--paper:#f5f3eb;--line:#c9d1ca;--accent:#b44225;--mint:#dce8dd}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 'Segoe UI',sans-serif}
main{max-width:1320px;margin:auto;padding:44px 5vw 80px}.eyebrow{font:12px/1.4 Consolas,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
header{border-top:5px solid var(--ink);padding-top:24px;display:grid;grid-template-columns:2fr 1fr;gap:30px;align-items:end}
h1{font:clamp(40px,6vw,76px)/1.03 Georgia,serif;font-weight:400;letter-spacing:-.045em;margin:18px 0 20px}h2{font:28px Georgia,serif;font-weight:400;margin:0 0 20px}
p{max-width:76ch}header aside{padding:20px;border-left:2px solid var(--accent);font-size:14px;color:var(--muted)}
.notice{margin:28px 0;background:var(--mint);padding:18px 22px;border-left:4px solid var(--ink)}
.controls{display:flex;gap:20px;align-items:end;flex-wrap:wrap;margin:30px 0}label{display:grid;gap:6px;font-size:13px;font-weight:600}
select,input{font:15px 'Segoe UI',sans-serif;padding:12px;border:1px solid var(--line);background:#fffefa;color:var(--ink);max-width:100%}select{max-width:480px}
a{color:var(--accent);text-underline-offset:3px}a:focus-visible,input:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:3px}
.section{border-top:1px solid var(--ink);padding-top:24px;margin-top:34px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px;text-align:left}
th{font:11px/1.4 Consolas,monospace;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);padding:12px 12px;border-bottom:2px solid var(--ink)}
td{padding:16px 12px;border-bottom:1px solid var(--line);vertical-align:top}.number{font-family:Consolas,monospace;text-align:right;white-space:nowrap}.score{font-size:21px;color:var(--accent)}
.bar{width:160px;max-width:30vw;height:7px;background:#e0e1d7;margin:7px 0 3px}.bar span{display:block;height:100%;background:var(--accent)}
.small{font-size:12px;color:var(--muted)}.model{font-size:17px;font-weight:600}summary{cursor:pointer;color:var(--ink);padding:6px 0}details{margin:18px 0;padding:16px 20px;background:#fffefa;border:1px solid var(--line)}
.empty{padding:30px;border:1px dashed var(--line)}footer{font-size:13px;color:var(--muted);margin-top:35px}.pill{font:11px Consolas,monospace;border:1px solid var(--line);padding:3px 6px}
@media(max-width:720px){main{padding:24px 18px}header{grid-template-columns:1fr}header aside{margin-top:0}.controls{display:grid}h1{font-size:48px}}
@media(prefers-reduced-motion:no-preference){main{animation:appear .45s ease-out}@keyframes appear{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}}
</style>
<main><header><div><div class="eyebrow">Team Placeholders / Research instrument 01</div><h1>Evidence before<br>submission.</h1><p>Historical seven-year forecasts. Official scoring arithmetic. A shared record of what the team has actually tested.</p></div>
<aside><strong>INTERNAL EVALUATION</strong><br>Historical scores compare models on the selected suite. They do not estimate an official rank. All weather labels end by 2018.</aside></header>
<div class="notice">Live analogue = leads 1-3. Final analogue = leads 4-7. Rank uses mean unclipped final skill. The summary clips once after averaging; official fold Scores are preserved below. Current historical years are reused evidence.</div>
<div class="controls"><label>Suite / phase / protocol<select id="suite"></select></label><label>Find a model<input id="search" type="search" placeholder="Model name or run ID"></label><a href="leaderboard.csv" download>Download run table ↗</a></div>
<section class="section"><h2>Trial and assessment exposure</h2><div id="exposure"></div><p class="small">Counts include known complete, failed and running trials across all models in this suite/protocol, unaffected by the search box. Accesses merge ledger events and run metadata by run ID; retries with another run ID count again. Repeated accesses mean extra access run IDs using an already-used freeze. Unknown ledger totals are disclosed as unknown, with known lower bounds. Deleted/unshared trials and manual label reads cannot be detected. <a href="exposure_counts.json" download>Download exposure inventory</a>.</p></section>
<section class="section"><h2>Experiment register <span class="pill" id="count"></span></h2><div class="scroll"><table><thead><tr><th>Internal rank</th><th>Model / run</th><th>Raw final skill / rank basis</th><th>Summary /100</th><th>Official fold Score range</th><th>Raw live skill</th><th>Folds</th></tr></thead><tbody id="rows"></tbody></table></div><p class="small">Rank applies only within the same suite, phase and protocol. Negative skill lowers the ranking. Ties use a 1e-10 skill tolerance. Legacy runs used weaker origin guards and appear separately. Repeated experiments remain separate records.</p></section>
<section class="section"><h2>Fold and metric evidence</h2><div id="details"></div></section>
<footer>MAE is pooled over eligible city-year rows within each metric. Metric skill = 1 - MAE / reference MAE; official fold Score = 100 * max(0, mean of four skills). Display rounding never feeds evaluation.<br>Inspect diagnostics.csv and row_errors.csv for city/year/lead evidence. Separately sealed station-change/Days Both bundles use the diagnostics workflow and do not change rankings. Every current run is fingerprint-verified.<br>Forecast Score informs value and creativity, whose combined rubric weight is 20%; it is not a separate points bonus. Assessment access requires a committed freeze and is logged. A freeze does not make reused years untouched.</footer></main>
<script type="application/json" id="data">__PAYLOAD__</script>
<script type="application/json" id="exposure-data">__EXPOSURE_PAYLOAD__</script>
<script>
const runs=JSON.parse(document.getElementById('data').textContent), select=document.getElementById('suite'), search=document.getElementById('search');
const exposures=JSON.parse(document.getElementById('exposure-data').textContent);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,n=2)=>Number(v).toFixed(n);
[...new Set(runs.map(r=>r.comparison_group))].forEach(id=>{const r=runs.find(r=>r.comparison_group===id),o=document.createElement('option');o.value=id;o.textContent=r.suite_id.slice(0,12)+' / '+r.phase+' / '+r.protocol_version;select.append(o)});
const currentProtocol=__CURRENT_PROTOCOL__;
const preferred=runs.filter(r=>r.protocol_version===currentProtocol && r.phase==='development').sort((a,b)=>b.run_id.localeCompare(a.run_id))[0];if(preferred)select.value=preferred.comparison_group;
function render(){const visible=runs.filter(r=>r.comparison_group===select.value && (r.model+' '+r.run_id).toLowerCase().includes(search.value.toLowerCase()));
const selected=runs.find(r=>r.comparison_group===select.value);
const scope=selected ? exposures.groups.filter(e=>e.suite_id===selected.suite_id && (e.protocol_version===selected.protocol_version || e.protocol_version==='unattributed-protocol')) : exposures.groups;
document.getElementById('exposure').innerHTML=scope.map(e=>`<div><strong>${esc(e.suite_id.slice(0,12))} / ${esc(e.protocol_version)}</strong><p>Known development trials: <strong>${e.known_development_trials}</strong> (${Object.entries(e.development_status_counts).map(([k,v])=>esc(k)+': '+v).join(', ') || 'none in available run records'}). Known assessment attempts: <strong>${e.known_assessment_attempts}</strong>.</p><p>Assessment access run IDs: <strong>${e.assessment_accesses===null?'unknown total; at least '+e.known_assessment_accesses:e.assessment_accesses}</strong>. Unique accessed freezes: <strong>${e.unique_freezes===null?'unknown total; at least '+e.known_unique_freezes:e.unique_freezes}</strong>. Known repeated access run IDs: <strong>${e.known_repeated_assessment_accesses}</strong>. Ledger: ${esc(e.ledger_status)}.</p>${e.warnings.map(w=>'<p class="small">'+esc(w)+'</p>').join('')}</div>`).join('') + exposures.warnings.map(w=>'<p class="small">'+esc(w)+'</p>').join('');
document.getElementById('count').textContent=visible.length+' runs';
document.getElementById('rows').innerHTML=visible.map(r=>`<tr><td class="number">${r.internal_rank}</td><td><div class="model">${esc(r.model)}</div><a class="small" href="${esc(r.metadata_url)}">${esc(r.run_id)}</a><br><span class="small">${esc(r.phase)} / ${esc(r.history_status)}</span></td><td><span class="score">${fmt(100*r.final_mean_skill_unclipped)}%</span></td><td class="number">${fmt(r.final_selection_score)}</td><td class="number">${fmt(r.final_min_score)} to ${fmt(r.final_max_score)}</td><td class="number">${fmt(100*r.live_mean_skill_unclipped)}%</td><td class="number">${r.final_n_folds}</td></tr>`).join('');
document.getElementById('details').innerHTML=visible.length ? visible.map(r=>`<details><summary><strong>${esc(r.model)}</strong> <span class="small">${esc(r.run_id)}</span></summary><div class="scroll"><table><thead><tr><th>Origin</th><th>Split</th><th>Official fold Score</th><th>Raw fold skill</th><th>Metric</th><th>MAE</th><th>Skill</th><th>Total scored rows</th></tr></thead><tbody>${r.scores.flatMap(s=>Object.keys(s.mae).map(m=>`<tr><td>${s.origin}</td><td>${esc(s.split)}</td><td>${fmt(s.score)}</td><td>${fmt(s.mean_skill_unclipped,4)}</td><td>${esc(m)}</td><td class="number">${fmt(s.mae[m],4)}</td><td class="number">${fmt(s.skill[m],4)}</td><td class="number">${s.n_rows}</td></tr>`)).join('')}</tbody></table></div></details>`).join('') : '<p class="empty">No complete matching experiments. Run an evaluation, then regenerate this report.</p>';
}select.addEventListener('change',render);search.addEventListener('input',render);render();
</script></html>'''
    path = output / "index.html"
    document = document.replace("__EXPOSURE_PAYLOAD__", exposure_payload).replace("__PAYLOAD__", payload)
    document = document.replace("__CURRENT_PROTOCOL__", json.dumps(PROTOCOL_VERSION))
    path.write_text(document, encoding="utf-8")
    return path, table
