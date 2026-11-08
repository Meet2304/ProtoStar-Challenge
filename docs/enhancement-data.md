# Enhancement station handoff

Station: `enhance`, branch: `Meet/enhance`. Generated datasets are handed off only
with `scripts/line.py release`; source and processing documentation live in
[`data/README.md`](../data/README.md).

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
