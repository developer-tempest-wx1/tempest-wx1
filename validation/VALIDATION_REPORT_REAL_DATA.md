# WX-1 Validation Report

- **Timestamp:** 2026-11-19 17:51:35 UTC
- **tempest-wx1 version:** `0.1.7`
- **Platform:** `Linux x86_64` | **Python:** `3.12.16`

## 1. Test Suite & v0.1.7 Regression Checks

- **Repository `pytest` suite:** `PASS` (60 passed, 3 skipped in 11.64s)
- **Stale temp-file same-PID write tolerance (`v0.1.7`):** `PASS`
- **`from_netcdf` multi-variable warning (`v0.1.7`):** `PASS`

## 2. Crash-Consistency Test (`kill -9` / `SIGKILL` mid-write overwrite)

Each trial restores a committed dataset `v1`, launches a child writer overwriting it with `v2`, and sends `SIGKILL` mid-write.
**Silent corruption** means a reader opened the file without error and returned neither intact `v1` nor intact `v2`.

| Format | Trials | Intact `v1` or `v2` | Caught Error / Unopened | Silent Corruption | Follow-up Write Blocked |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `wx1` | 30 | 30 (`v1`=11, `v2`=19) | 0 | **0** | 0 |
| `zarr` | 30 | 8 (`v1`=2, `v2`=6) | 0 | **22** | 0 |
| `netcdf4` | 30 | 8 (`v1`=0, `v2`=8) | 0 | **22** | 0 |

## 3. GRIB1 / GRIB2 Message-Exact Roundtrip Verification (`wx1.adapters.grib_exact`)

| File (basename) | Msgs | Editions | Packing | Grid | Source Size | WX-1 `1d` (vs Src) | WX-1 `2d` Lorenzo (vs Src) | `2d` Lattice / RawFallback | SHA-256 Byte-Identical (`2d`) | `float64` Bit-Exact (`2d`) | Max Abs Err |
| :--- | ---: | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |
| `era5_pl_74msg.grib` | 74 | [1] | `grid_simple` | `721x1440` | 136.65 MiB | 46.08 MiB (-66.3%) | **35.97 MiB (-73.7%)** | 74 / 0 | 74 / 74 | **74 / 74** | `0` |
| `ecmwf_0p25_2026111618_7a39663d_pl.grib2` | 600 | [2] | `grid_ccsds` | `721x1440` | 267.61 MiB | 258.17 MiB (-3.5%) | **233.99 MiB (-12.6%)** | 600 / 0 | 600 / 600 | **600 / 600** | `0` |
| `HA46_N20_202611181700_00000_GB` | 46 | [2] | `grid_simple` | `390x390` | 9.90 MiB | 7.17 MiB (-27.6%) | **6.97 MiB (-29.6%)** | 44 / 2 | 46 / 46 | **46 / 46** | `0` |
| `icon_eu_t2m_sample.grib2` | 8 | [2] | `grid_ccsds` | `657x1377` | 7.93 MiB | 8.33 MiB (+5.1%) | **7.93 MiB (-0.0%)** | 8 / 0 | 8 / 8 | **8 / 8** | `0` |

## 4. NetCDF-4 Roundtrip Verification

| File (basename) | Variable | Shape | Dtype | NetCDF-4 Size | WX-1 Size (vs NC4) | Bit-Exact Roundtrip |
| :--- | :--- | :--- | :--- | ---: | ---: | :---: |
| `enfo_2t.nc` | `2t` | `[50, 261, 661]` | `float32` | 3.15 MiB | 3.11 MiB (-1.3%) | `True` |
| `t.nc` | `t` | `[14, 261, 661]` | `float64` | 0.61 MiB | 0.64 MiB (+3.6%) | `True` |
