# WX-1 — Current Project Status

**Status Date:** 2026-11-18  
**Current Releases:**
- `tempest-wx1` `v0.1.6` ([GitHub](https://github.com/developer-tempest-wx1/tempest-wx1), [PyPI](https://pypi.org/project/tempest-wx1/)) — Rust core (`wx1-format`, `wx1-registry`, `wx1-ffi`, `wx1-cli`) + Python bindings (`wx1`, `engine="wx1"`)
- `wx1-lorenzo` `v0.1.1` ([GitHub](https://github.com/developer-tempest-wx1/wx1-lorenzo), [PyPI](https://pypi.org/project/wx1-lorenzo/)) — Standalone native Zarr v3 `ArrayArrayCodec` (`wx1.lorenzo2d`)

For installation instructions, API examples, and benchmark tables, see [README.md](README.md) and [docs/WX1_TECHNICAL_EVALUATION_KNMI_ECMWF.md](docs/WX1_TECHNICAL_EVALUATION_KNMI_ECMWF.md).

## Verification Summary

- **Rust Workspace (`cargo fmt`, `cargo clippy -- -D warnings`, `cargo test --workspace`):** 160+ Rust unit, integration, golden-vector, fuzzing/robustness, doc, and C-ABI (`gcc`) tests passing with `#![forbid(unsafe_code)]`.
- **Python Suite (`pytest python/tests/`):** All core container, golden-vector, xarray/dask, adapter error-semantics, and roundtrip tests passing on Linux (`x86_64`, Python 3.10–3.13; Windows supported via WSL2).
- **Large-Scale GRIB Lattice Verification:**
  - Bit-exact `float64` (`max_abs_err = 0.0` vs. ecCodes) across all 28,416 fields of the 52 GiB CDS ERA5 pressure-level download (`wx1_grib_2d`: 19.52 GB vs. 23.03 GB GRIB2-AEC, −15.2%).
  - Byte-exact SHA-256 ecCodes message roundtrip verified on 988/988 ecCodes-generated GRIB1 `grid_simple` and GRIB2 `grid_ccsds` messages (see Section 6 of [docs/WX1_TECHNICAL_EVALUATION_KNMI_ECMWF.md](docs/WX1_TECHNICAL_EVALUATION_KNMI_ECMWF.md) for limitations on NOAA GFS complex packing).
- **Standalone Zarr v3 Codec (`wx1-lorenzo`):** 8/8 codec unit tests passing; end-to-end GRIB2-to-Zarr example achieves bit-exact `float64` reconstruction and −12.6% size vs. GRIB2-AEC on 600 IFS 0.25° pressure-level fields.
