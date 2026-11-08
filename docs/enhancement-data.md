# Enhancement station handoff

Station: `enhance`, branch: `Meet/enhance`. Generated datasets are handed off only
with `scripts/line.py release`; source and processing documentation live in
[`data/README.md`](../data/README.md).

The latest enhanced input dataset is pinned in `enhanced/latest.json`. Run
`uv run python scripts/materialize_enhanced.py` to create the verified real copy
under `enhanced/enhance-v2/` in your own worktree. The canonical release copy is
also retained for evaluator asset binding and eligibility checks. See
[`enhanced/README.md`](../enhanced/README.md). New model evaluations should pin
this latest enhanced input version while preserving its exploratory status.

## `enhance-v1`: fixed station geography

```powershell
uv run python scripts/line.py pull enhance-v1
```

Join `data/processed/releases/enhance-v1/static/city_static.csv` to a model's
city identifiers on `city_id`, using a many-to-one validation. There are exactly
210 unique city IDs. The columns contain observing coordinates, USGS ground
elevation, approximate distance to the ocean coast, and ESA WorldCover 2021
station class and class fractions within 1 km. Read the released `schema.json`
for units, class codes and limitations, and `quality.json` for coverage checks.

This is permitted structural information from any year, including a 2021 land
cover snapshot. It should not be described as historically observed land cover
at each training origin. The coordinate/elevation catalog differences are exposed
instead of silently assigning a station elevation to a different location.

The publisher saves a checksummed source catalog subset, a version-pinned
coastline, point-elevation query responses, and native-value raster chips. None
is a weather observation. A failed validation prevents publication.

For forecast trials, the features and training stations must declare the release
as an asset and use `line.py score`; this station has not trained or scored a model.
Do not open protected assessment labels to investigate these inputs.

## `enhance-v2`: ocean-temperature candidates and the static table

```powershell
uv run python scripts/line.py pull enhance-v2
```

This cumulative handoff contains the same fixed city facts plus three ERSSTv5
regional temperature indices: Niño 3.4, a North Pacific box, and a North Atlantic
box. Each has an absolute SST and a per-origin anomaly. The latter two are not
the official PDO or AMO indices. No observations later than December 2018 are
retained. Definitions, units and bounds are in the data card and released schema.

Use the origin-specific loader rather than an all-years table:

```python
from pathlib import Path
from enhance_ocean import load_ocean_for_origin

history = load_ocean_for_origin(
    Path("data/processed/releases/enhance-v2"), origin=1983, frequency="monthly"
)
```

Supported origins are every year from 1979 to 2018. The loader verifies the file
hash and both its row cutoff and normalization cutoff. A trainer predicting a
future year cannot look up that year's observed index: it must forecast the
future ocean state, marginalize its uncertainty, or use only the origin-time
history. These files are historical inputs, not future scenario predictions.

`enhance-v2` is **exploratory and not gate-eligible**. A modern reconstructed SST
record has not been certified to exclude later information in its reconstruction
at early historical origins. Do not describe origin-specific row slicing as a
complete solution to source-vintage leakage. `enhance-v1` remains the eligible
static-only option. The integrator owns the decision on official gated use.

The release points to `enhance-v1` as upstream. Pull that release too when full
structural source snapshots are needed. Offline rebuilds must write a new output
directory and leave the immutable releases untouched.

## Operational findings

- The MLflow server responds with HTTP 401 without station credentials. Local
  immutable releases remain valid and pullable; mirroring is pending authenticated
  `line.py sync`. No credentials are saved in code or data.
- The USGS point endpoint sometimes times out; batched 3DEP requests provide a
  documented alternative. Some source acquisition dates are malformed, so their
  raw strings are retained and normalized dates are explicitly unknown.
- NCSS can select the next monthly coordinate when an end date falls near it.
  Exact month-start bounds plus returned-date validation prevent retained
  post-cutoff observations. The rejected probe is documented in the data card.
