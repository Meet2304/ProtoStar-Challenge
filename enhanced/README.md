# Tagged enhanced datasets

The current enhanced input dataset is **`enhance-v3`**, pinned by
[`latest.json`](latest.json), including its release-manifest SHA256. New model
evaluations should declare this version when using the latest enhanced inputs.
Freeze the tag and hashes when a run starts; a future change to the pointer must
not alter a running or sealed evaluation.

Create the real tagged directory in your own worktree:

```powershell
uv run python scripts/materialize_enhanced.py
```

This uses the assembly-line release downloader, verifies the immutable release,
and creates **`enhanced/enhance-v3/`**. It also ensures that the canonical
`data/processed/releases/enhance-v3/` copy exists for evaluator asset declarations.
It is a real local copy, not a junction to another station. Generated contents
are ignored by Git; the pointer, downloader and documentation are tracked.

The tagged dataset contains:

- `static/city_static.csv`: fixed facts for all 210 cities, unchanged from
  `enhance-v1` (elevation, approximate ocean-coast distance, land cover).
- `ocean/origin-YYYY/monthly.csv` and `annual.csv`: three SST indices and
  per-origin anomalies for every origin from 1979 through 2018.
- `forcing/origin-O/annual.csv`: annual global effective radiative forcing,
  meaning changes to Earth's energy balance, from 1750 through each of
  1983, 1990, 1997, 2004, 2011 and 2018. Components include greenhouse gases,
  aerosols, volcanic activity, solar activity and total forcing.
- `forcing/design-to-1982/annual.csv`: the requested full design resource,
  plus a separate `ar5-reference.csv` for sensitivity analysis.
- Source snapshots, licenses, origin-scoped asset templates and hash manifests.
  Retained ocean and forcing inputs end in 2018. Previous release metadata is
  retained under `upstream/`.

**Eligibility:** `enhance-v3` is exploratory, not gate-eligible. Its primary
forcing source is a retrospective 2021 IPCC AR6 reconstruction. Neither that
source nor the inherited SST reconstruction is certified origin-vintage-clean.
The directory
name or "latest" pointer does not override that flag. The eligible structural-only
release is `enhance-v1`. See [source documentation](../data/README.md) and
[the handoff guide](../docs/enhancement-data.md), and
[the forcing handoff](../docs/forcing-handoff.md).

The evaluator currently recognizes canonical release paths for asset lineage.
Declare model assets under `data/processed/releases/enhance-v3/` and use the
origin-specific loaders there. The friendly tagged directory has identical
content; a move of asset bindings to `enhanced/` needs explicit evaluator support.
The enhancement changes forecast inputs; evaluation targets, score definitions,
and protected assessment rules continue to apply. Score models with `line.py score`.
