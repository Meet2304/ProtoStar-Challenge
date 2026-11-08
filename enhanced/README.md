# Tagged enhanced datasets

The current enhanced input dataset is **`enhance-v2`**, pinned by
[`latest.json`](latest.json), including its release-manifest SHA256. New model
evaluations should declare this version when using the latest enhanced inputs.
Freeze the tag and hashes when a run starts; a future change to the pointer must
not alter a running or sealed evaluation.

Create the real tagged directory in your own worktree:

```powershell
uv run python scripts/materialize_enhanced.py
```

This uses the assembly-line release downloader, verifies the immutable release,
and creates **`enhanced/enhance-v2/`**. It also ensures that the canonical
`data/processed/releases/enhance-v2/` copy exists for evaluator asset declarations.
It is a real local copy, not a junction to another station. Generated contents
are ignored by Git; the pointer, downloader and documentation are tracked.

The tagged dataset contains:

- `static/city_static.csv`: fixed facts for all 210 cities, unchanged from
  `enhance-v1` (elevation, approximate ocean-coast distance, land cover).
- `ocean/origin-YYYY/monthly.csv` and `annual.csv`: three SST indices and
  per-origin anomalies for every origin from 1979 through 2018.
- `release.json`, `schema.json`, `provenance.json`, `quality.json` and ocean
  source snapshots. Retained ocean observations end in December 2018.

**Eligibility:** `enhance-v2` remains exploratory, not gate-eligible, pending
review of the SST reconstruction's historical information vintage. The directory
name or "latest" pointer does not override that flag. The eligible structural-only
release is `enhance-v1`. See [source documentation](../data/README.md) and
[the handoff guide](../docs/enhancement-data.md).

The evaluator currently recognizes canonical release paths for asset lineage.
Declare model assets under `data/processed/releases/enhance-v2/` and use the
origin-specific ocean loader there. The friendly tagged directory has identical
content; a move of asset bindings to `enhanced/` needs explicit evaluator support.
The enhancement changes forecast inputs; evaluation targets, score definitions,
and protected assessment rules continue to apply. Score models with `line.py score`.
