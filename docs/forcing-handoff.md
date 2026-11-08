# Forcing handoff to deep-learning training

Release: **`enhance-v3`**, cumulative with `enhance-v2`, exploratory only.
No observed or scenario forcing beyond 2018 is supplied.

```powershell
uv run python scripts/line.py pull enhance-v3
```

Primary files are `data/processed/releases/enhance-v3/forcing/origin-O/annual.csv`
for O in 1983, 1990, 1997, 2004, 2011 and 2018. Each contains years 1750..O.
The requested full design file is
`data/processed/releases/enhance-v3/forcing/design-to-1982/annual.csv`.
An alternate AR5 design file is beside it as `ar5-reference.csv`.

Primary columns are `year`, `wmghg_erf_wm2`, `co2_erf_wm2`,
`tropospheric_aerosol_erf_wm2`, `volcanic_erf_wm2`, `solar_erf_wm2`, and
`total_erf_wm2`, with individual components and subtotals also retained.
All forcing columns use W/m2. The totals overlap the components, so don't add
every input column together. Published background offsets are preserved.

The homogeneous primary series is IPCC AR6, dated 2021-08-12, capped at 2018.
Pre-2019 AR5 historical forcing ends in 2011; the separate AR5 source revision
is dated 2018-01-08. There is no invented 2012–2018 extension and no splice of
the two forcing definitions. See [data/README.md](../data/README.md) for the exact
source commits, citations, byte-bounded acquisition, units, licenses and limits.

Asset templates live under the release's `assets/` directory. Incorporate the
origin-scoped declarations into the model's own manifest and follow the current
harness freeze/committed-asset contract. Each is `weather_input` with its
`information_end_year`, `origins`, `time_column` and `time_format` declared.
Templates do not attest model readiness or override Git asset verification.
The 1982 design resource must be cut again before it is used inside a forecast
screen with an origin earlier than 1982. Use only information through that
screen's actual origin to fit transformations or predict future forcing.

```python
from pathlib import Path
from enhance_forcing import load_forcing_for_origin

forcing = load_forcing_for_origin(
    Path("data/processed/releases/enhance-v3"), 1983
)
```

Eligibility decision: **gate_eligible=false**. A retrospective forcing estimate
is not automatically origin-vintage-clean merely because forcing is an external
physical driver. This particularly matters for aerosol/cloud and volcanic
reconstructions. Do not promote the dataset or its trial to official. No model
or assessment has been run by the enhancement station.

Existing city and ocean data remain available in this cumulative release. The
previous releases remain immutable; they are not replaced in sealed runs.
