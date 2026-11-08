# Declared model assets

The integrated implementation is `code/evaluation/assets.py`.

For a custom adapter, pass the same manifest to run and freeze:

```bash
uv run python code/evaluate.py run --name my-model --adapter code/my_model.py:forecast --assets evaluation/model-assets.json
uv run python code/evaluate.py freeze --adapter code/my_model.py:forecast --assets evaluation/model-assets.json --output outputs/evaluation/freezes/my-model.json
```

An explicit empty declaration is supplied in `evaluation/no-assets.json` for
callbacks using only provided train/targets and their code/libraries. Built-in
reference/nested adapters declare that no external assets are used internally.
Model code should read declared dependencies relative to
`Path(os.environ["PROTOSTAR_ASSET_ROOT"])`, the worker's verified snapshot.
The worker verifies the snapshot before adapter import and after prediction.
Changes to original inputs during the run also fail validation.

## API

```python
binding = fingerprint_assets(root, "evaluation/model-assets.json", origins,
                             commit=full_commit_sha_or_none)
verified = verify_asset_binding(root, binding, commit=full_commit_sha_or_none)
worker_binding = project_asset_binding(binding, [origin])
snapshot = snapshot_assets(root, binding, run_dir, origin=origin,
                           commit=full_commit_sha_or_none)
```

`AssetError` is a `ValueError`. `root` and `run_dir` accept paths; manifest paths
are portable project-relative POSIX **strings**. All explicit origins are unique
integers from 1 through 2018. The full manifest is always pinned byte-for-byte;
only assets applicable to requested origins are inspected or copied. An explicit
scope may include origins not requested by the current phase/worker. Inactive
assets may be absent. Metadata is still parsed strictly for all declarations.

`binding` contains:

- `manifest`: original relative path, SHA256 and byte size;
- `declarations`: normalized full declarations, including inactive scopes;
- `files`: active files, content hashes, sizes, roles, cutoffs and scopes;
- `per_origin`: each origin's pins and fingerprint;
- `origins`, `schema_version`, and `binding_id` over the full record.

`commit` checks the manifest and active assets against a **local** full Git SHA.
Ordinary blobs first compare exact bytes. Manifest JSON and declared textual
configuration/static/source extensions (plus weather CSV) may compare with
CRLF-to-LF normalization when both contents are UTF-8 without NUL. Runtime
bindings/snapshots still preserve RAW SHA256/size; changing line endings after
freeze invalidates that raw binding. Learned weights, binary extensions and
LFS receive no text normalization. Materialized LFS files must match the
committed standard v1 pointer's `oid sha256` and size. No filters,
remote fetches, network calls or LFS commands are used. Missing/unmaterialized
LFS content fails. Binary assets should use Git attributes preserving bytes.

`snapshot_assets` returns `snapshot_root`, `manifest_path`, `binding_path` and
the projected `binding`. Files live under
`run_dir/declared_assets/files/<original-project-relative-path>`. The full
manifest is copied at its **original relative path** within `snapshot_root` so
workers can call `verify_asset_binding(snapshot_root, projected_binding)` before
importing the adapter. The binding JSON lives outside that file tree. Existing
snapshots, overlaps and file/manifest collisions fail. Copy/hash operations are
streaming; copying is staged, verified, then renamed atomically.

## Manifest

```json
{
  "schema_version": 1,
  "files": [
    {
      "path": "models/weights-1983.bin",
      "kind": "model_weights",
      "information_end_year": 1980,
      "origins": [1983]
    },
    {
      "path": "models/settings.json",
      "kind": "configuration"
    },
    {
      "path": "models/historical-covariates.csv",
      "kind": "weather_input",
      "information_end_year": 1980,
      "time_column": "Year",
      "time_format": "year",
      "origins": [1983]
    }
  ]
}
```

An explicit empty `files` list is valid. Supported kinds: `model_weights`,
`configuration`, `static_context`, `weather_input`, `source`. Learned/weather
inputs require an author-declared cutoff no later than each applicable origin
and 2018. Config/static/source entries do not take cutoff metadata. Config/static
classification is an author's assertion of weather independence, not an
automatic scientific check.

Weather CSV/parquet declarations additionally require a `time_column` and one
of:

- `year`: exact four-digit `YYYY` integer values; null, fractional or scientific
  notation values fail;
- `iso_date`: exact valid `YYYY-MM-DD` strings or parquet dates/midnight typed
  datetimes;
- `iso_datetime`: exact valid
  `YYYY-MM-DDTHH:MM:SS[.ffffff][Z|+HH:MM|-HH:MM]` strings, or typed parquet
  datetimes. Calendar years use stated timestamp dates, without timezone
  conversion.

Actual weather years are checked with `common.assert_no_post_cutoff` and against
the origin and declared information-end year. No rows are silently subset.
Missing/duplicate headers, malformed times, unsupported formats and empty inputs
fail. CSV/parquet temporal-column inspection is batched; other columns are not
loaded. `common` must be importable in the worker, as it already is for the
evaluation adapters. No import of `evaluation.core` is made.

All unknown fields/kinds, duplicate JSON keys, duplicate/case-colliding/prefix-
colliding declarations, traversal, absolute paths, backslashes, Windows-reserved
names, symlinks, nonordinary files and credential-looking names/extensions fail.
Credential-looking paths are rejected lexically before opening them. Tests
reference such names only in manifest strings and create no secret files.

This protects against accidental drift and helps reproducibility. It cannot
certify real training history or stop deliberately undeclared reads; it is not
an OS sandbox or cryptographic execution attestation.

## Verification

Run the full regression suite from the project root:

```bash
uv run python -m unittest discover -s tests -v
```

The tests use synthetic data/local Git fixtures. LFS checks never fetch remote
objects, and the examples do not use observed weather after 2018.
