# Final submission record (2026-11-10)

The team's final submission is the Hugging Face dataset
[`DragonDare/protostar-placeholders-f4`](https://huggingface.co/datasets/DragonDare/protostar-placeholders-f4)
at commit `cbf9232` (submitted 2026-11-11 02:52 UTC = 2026-11-10 21:52 ET).

It is assembled from:

| Part | From |
|---|---|
| Model, forecasts and evidence (Placeholder_F4) | [`Meet2304/protostar-placeholders`](https://huggingface.co/datasets/Meet2304/protostar-placeholders) at commit `099a52d`, unchanged. `submission.csv` SHA-256 `8344e656c942e95fd522d9e7f10f2d4c5a4847b82650f4e6de1d5c9d4633bc37` |
| Outside data and policy scripts | this branch (`kohav`): `code/prepare_ocean_indices.py`, `code/prepare_population.py`, `code/policy_cdd.py`, `data/ocean_indices.csv`, `data/ocean_indices_sources.csv`, `data/city_population.csv`, `reports/enso_rolling_test.csv` |
| Data-cleaning docs and pipeline | `main`: `documentation/`, `scripts/`, `src/`, `tests/`, `notebooks/Meet/`, two plan docs, `docs/evaluation-baselines.md`, `evaluation/independent/` (without `snapshots/`) |
| New in the final dataset (copied here) | `README.md` (merged write-up), `data/README.md` (data card with the outside-data rows), `code/download_raw.py` (standalone raw-data download for the F4 rebuild), `reports/policy_cdd_by_area.csv` and `reports/policy_cdd_by_city.csv` (computed from F4's `submission.csv`); `data/processed/f2/` (F4's processed daily data from the rebuild, not copied here: 28 MB, rebuilt by `code/f2/f2_daily.py` and `code/f2/f2_cddgrid.py`) |

**Rebuild check (2026-11-10, Kohav's laptop, from the dataset folder):** `python code/download_raw.py`,
`python code/f2/f2_daily.py`, `python code/f2/f2_cddgrid.py`, `python code/make_submission.py` gave the same SHA-256 as above.

The files in this folder are copies for the record: their relative links point to the dataset's layout, not this branch's.
