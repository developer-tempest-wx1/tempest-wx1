# WX-1 Validation Kit (KNMI / ECMWF)

Self-contained single-command validation script for **`tempest-wx1`** (`>= 0.1.7`).

## What it checks

1. **Test Suite & `v0.1.7` Regressions:** Runs `python/tests/` (when inside the repository) plus direct checks for:
   - same-PID leftover temporary files never blocking subsequent `write_array` calls (`v0.1.7` atomic temp-file fix),
   - `from_netcdf` emitting an explicit `UserWarning` when importing from a multi-variable NetCDF file without `variable_name=...`.
2. **Crash-Consistency (`kill -9` / `SIGKILL`):** Spawns subprocess writers overwriting a committed dataset and sends `SIGKILL` mid-write (`--crash-trials 30` per format across WX-1, Zarr v3, and NetCDF-4), checking for silent corruption and follow-up write blocking.
3. **GRIB1 / GRIB2 Message-Exact Verification:** For each `--grib` file passed, compares `wx1.adapters.grib_exact` (`1d` delta and `2d` Lorenzo) size vs. source GRIB, `float64` bit-exactness against ecCodes, and SHA-256 byte-identical message reconstruction.
4. **NetCDF-4 Roundtrip Verification:** Verifies lossless `from_netcdf` -> `to_netcdf` roundtrip on supplied (or synthetic) NetCDF-4 files.

All usernames, hostnames, and full directory paths are stripped from the generated report (`--output`).

## Quickstart

Install inside WSL2 or Linux (`x86_64`, Python 3.10–3.13):

```bash
pip install "tempest-wx1[all]>=0.1.7" pytest
git clone https://github.com/developer-tempest-wx1/tempest-wx1.git
cd tempest-wx1/validation
```

Run with built-in synthetic fixtures:

```bash
python run_validation.py
```

Run against your own operational GRIB1/GRIB2 and NetCDF-4 files (optionally targeting NFS/Lustre with `--scratch-dir`):

```bash
python run_validation.py \
  --grib /path/to/ifs_or_era5.grib2 \
  --grib /path/to/harmonie_or_icon.grib2 \
  --netcdf /path/to/dataset.nc \
  --crash-trials 30 \
  --scratch-dir /path/on/target_filesystem \
  --output wx1_validation_report.md
```

Real-data run results from our own ERA5, ECMWF IFS 0.25°, KNMI HARMONIE-AROME Cy46, and DWD ICON-EU files are recorded in [VALIDATION_REPORT_REAL_DATA.md](VALIDATION_REPORT_REAL_DATA.md).
