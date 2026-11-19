# WX-1: Open Scientific & Meteorological Container Format

[![PyPI](https://img.shields.io/badge/pypi-v0.1.7-blue.svg)](https://pypi.org/project/tempest-wx1/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://pypi.org/project/tempest-wx1/)
[![Rust](https://img.shields.io/badge/rust-1.85%2B-lightgrey.svg)](https://www.rust-lang.org)
[![Safety](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-lang/rfcs)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**WX-1** (`tempest-wx1` v0.1.7) is a scientific storage container format engineered for numerical weather prediction (NWP), climate reanalysis (ERA5), and AI-driven weather modeling.

It is written in safe Rust (`#![forbid(unsafe_code)]`), featuring a stable C-ABI (`include/wx1.h`), Python PyO3 bindings, an `xarray` backend engine (`engine="wx1"`), and bidirectional adapters for WMO GRIB Edition 1 & 2, NetCDF-4 (HDF5), and Zarr v2/v3.

---

## Two Storage Modes: Exact GRIB Integer Packing vs. Native Float32

WX-1 separates storage into two explicit modes depending on the origin of the data:

1. **GRIB Integer Packing (`wx1_grib_delta` 1D & `wx1_grib_2d` 2D Lorenzo)**:
   - For data originating from WMO GRIB1/GRIB2, WX-1 stores the exact quantized GRIB integers $X_i \in \{0, \dots, 2^{\text{bpv}}-1\}$ alongside the per-slice linear scaling parameters $(R, E, D)$ where $Y = (R + X \cdot 2^E) \cdot 10^{-D}$.
   - **100% Bit-Exact on both `float64` and `float32`**: Reconstructed arrays match `eccodes.codes_get_values(gid)` (IEEE-754 64-bit `double` as well as 32-bit `float`) with $\Delta_{\max} = 0.0$ across all $29.5\times 10^9$ grid points tested, including GRIB bitmaps (`bitmapPresent=1`, `NaN` land masks) and NOAA GFS `grid_complex_spatial_differencing`.
   - **File Size vs. GRIB2-AEC (`grid_ccsds`)**:
     - With 1D horizontal delta (`wx1_grib_delta`), file size is at **parity with GRIB2-AEC** (`-3.5%` to `+5.4%` on multi-variable datasets; `-2.5%` on the full 52 GB ERA5 pressure-level dataset), while being **26–48% smaller** than lossless `float32` NetCDF-4 (`zlib4`) and Zarr v3.
     - With the chunk-local **2D Lorenzo spatial predictor (`wx1_grib_2d`)**, WX-1 is **12.6% smaller than GRIB2-AEC** on ECMWF IFS 4D pressure levels and **15.2% smaller than GRIB2-AEC** (`19.52 GB` vs. `23.03 GB`) across all 28,416 fields of the 52 GB ERA5 pressure-level dataset.
2. **Native Float32 Container (`wx1_zstd_shuffle` & `wx1_bitround`)**:
   - For native IEEE-754 `float32` model output (e.g., NeuralGCM, JAX/PyTorch rollouts, or unquantized simulations), WX-1 stores `float32` chunks with byte-shuffle + Zstd.
   - **File Size**: Lossless `float32` size is at **parity (`-0.5%` to `+2.5%`)** with NetCDF-4 `zlib4` and Zarr v3 `blosc_zstd3`. Optional mantissa bit-rounding (`bitround`) or bounded error quantization (`profile="output"`, `lossy_error_bound=...`) reduces size by 15–38%.

---

## Measured Benchmark Evidence

### 1. Full 52 GiB ERA5 Pressure-Level Dataset (48 Hours × 37 Levels × 16 Variables = 28,416 Fields)

Measured across the **entire** unarchived Copernicus CDS ERA5 pressure-level download (`era5-pl.grib`, `55,320,496,128` bytes, $721 \times 1440$ global $0.25^\circ$ grid, $29,502,627,840$ grid points; no extrapolation):

| Format | Measured Size (MB) | Measured Size (GB) | vs. CDS GRIB1 (`grid_simple`) | vs. GRIB2-AEC (`grid_ccsds`) | `float64` Bit-Exact (`eccodes`) |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Raw `float64` (`eccodes` native decode) | 236,021.02 | 236.02 | +326.6% | +924.9% | reference |
| Raw `float32` | 118,010.51 | 118.01 | +113.3% | +412.5% | ✓ (32-bit) |
| Original CDS GRIB1 (`grid_simple`, 8/12/16-bit) | 55,320.50 | 55.32 | +0.0% | +140.2% | ✓ (28,416 / 28,416) |
| Transcoded GRIB2-AEC (`grid_ccsds`, native bpv) | 23,027.87 | 23.03 | -58.4% | +0.0% | ✓ (28,416 / 28,416) |
| **WX-1 `wx1_grib_delta` (1D-delta)** | **22,443.07** | **22.44** | **-59.4%** | **-2.5%** | **✓ (28,416 / 28,416, $\Delta=0$)** |
| **WX-1 `wx1_grib_2d` (2D Lorenzo predictor)** | **19,519.06** | **19.52** | **-64.7%** | **-15.2%** | **✓ (28,416 / 28,416, $\Delta=0$)** |

### 2. Multi-Variable Containers Across Operational NWP & Reanalysis Datasets

| Dataset | Source Packing | `bitsPerValue` | Source GRIB (MB) | GRIB2-AEC (MB) | **WX-1 `wx1_grib_delta` (MB)** | **WX-1 `wx1_zstd_shuffle` (MB)** | NetCDF-4 `zlib4` (MB) | Zarr v3 Sharded (MB) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **ERA5 4D PL (48h, 16 vars, 37 levs)** | GRIB1 `grid_simple` | 8, 12, 16 | 55,320.50 | 23,027.87 | **22,443.07** *(-2.5% vs AEC; **19,519.06** 2D)* | ~33,980 *(+1.2% vs NC4)* | ~33,570 | ~35,980 |
| **ERA5 3D SFC (24h, 7 vars)** | GRIB1 `grid_simple` | 16 | 335.09 | 170.71 | **170.20** *(-0.3% vs AEC, -28.1% vs NC4)* | 241.54 *(+2.0% vs NC4)* | 236.77 | 251.46 |
| **ECMWF AIFS 0.25° 3D SFC (8 vars, T=12)** | GRIB2 `grid_ccsds` | 8–12 | 19.43 | 19.42 | **19.27** *(-0.8% vs AEC, -46.8% vs NC4)* | 35.77 *(-1.3% vs NC4)* | 36.23 | 37.00 |
| **ECMWF IFS 0.25° 3D SFC (9 vars, T=12)** | GRIB2 `grid_ccsds` | 7–12 | 52.78 | 52.77 | **54.91** *(+4.0% vs AEC, -36.9% vs NC4)* | 87.32 *(+0.3% vs NC4)* | 87.06 | 89.97 |
| **ECMWF IFS 0.25° 4D PL (5 vars, 10 levs, T=12)** | GRIB2 `grid_ccsds` | 7–12 | 280.61 | 280.60 | **270.75** *(-3.5% vs AEC; **245.36** 2D = -12.6%)* | 531.31 *(+2.5% vs NC4)* | 518.42 | 553.09 |
| **ECMWF IFS 0.1° 3D SFC (9 vars, T=12)** | GRIB2 `grid_ccsds` | 7–12 | 185.99 | 185.98 | **196.09** *(+5.4% vs AEC, -45.2% vs NC4)* | 356.74 *(-0.3% vs NC4)* | 357.80 | 369.05 |
| **NOAA GFS 0.25° 4D PL (6 vars, 13 levs, T=12)** | GRIB2 `grid_complex` | 9–16 | 464.47 | — | **452.92** *(-2.5% vs GFS-orig, -59.8% vs NC4)* | 1,149.49 *(+2.1% vs NC4)* | 1,126.22 | 1,204.20 |
| **DWD ICON-EU 0.0625° 4D PL (6 vars, 8 levs, T=24)** | GRIB2 `grid_ccsds` | 16 | 1,174.80 | 1,172.56 | **1,195.21** *(+1.7% vs AEC, -25.5% vs NC4)* | 1,622.10 *(+1.1% vs NC4)* | 1,605.05 | 1,725.08 |

### 3. 4D Sub-Volume Random Access (Median $\pm$ IQR ms, $n=5$, Chunks `[1, 1, 721, 1440]`)

On 4D pressure-level cubes (`ECMWF IFS 0.25°` `t` `[12, 10, 721, 1440]` and `ERA5` `t` `[4, 37, 721, 1440]`):
- **Single-Level Map (`t500_map`, cold)**: `zarr3_blosc_zstd3` `6.4 ±0.2 ms`, `wx1_grib_int` `7.0 ±0.2 ms`, `zarr3_sharded` `7.5 ±0.3 ms`, `wx1_zstd_shuffle` `8.6 ±2.7 ms`, `wx1_grib_delta` `11.0 ±1.6 ms`, `grib2_orig_indexed` `11.2 ±0.2 ms`, `netcdf4_zlib4` `16.5 ±0.3 ms`.
- **Vertical Sounding (`sounding_col`, 10–37 levels at 1 station, cold)**: WX-1 (`wx1_grib_delta` `14.0 ±0.5 ms` on 10 levels / `48.7 ±1.5 ms` on 37 levels) and Zarr v3 (`12.5–15.6 ms` on 10 levels / `48.6–66.0 ms` on 37 levels) perform at parity, and both are **6–9× faster** than NetCDF-4 (`97.5 ms` / `429.3 ms`), HDF5 (`85.7 ms` / `446.5 ms`), and indexed GRIB2 (`111.1 ms` / `526.8 ms`).
- **Column Time-Series (`col_timeseries`, $T \times L$ slices at 1 station, cold)**:
  - On IFS (`12 × 10 = 120` slices): `wx1_grib_delta` `98.9 ±1.3 ms`, `wx1_grib_int` `117.0 ±6.5 ms`, `zarr3_blosc_zstd3` `139.7 ±2.7 ms`, `zarr3_sharded` `142.8 ±10.2 ms`, vs. `netcdf4_zlib4` `987.4 ±18.1 ms` and `grib2_orig_indexed` `1,225.4 ±18.0 ms`.
  - On ERA5 (`4 × 37 = 148` slices): `zarr3_blosc_zstd3` `171.4 ±3.4 ms`, `wx1_grib_int` `174.9 ±14.0 ms`, `wx1_grib_delta` `199.2 ±1.3 ms`, `zarr3_sharded` `226.4 ±5.3 ms`, vs. `netcdf4_zlib4` `1,756.3 ±40.5 ms` and `grib2_ccsds_native` `1,966.3 ±29.0 ms`.

---

## Quickstart

### Python Installation

```bash
pip install tempest-wx1
```

> **Supported Platforms:** Prebuilt binary wheels and CI checks target **Linux (`x86_64`)** (Python 3.10–3.13) only. **Windows is supported via WSL2 (Windows Subsystem for Linux 2)** — install and run `tempest-wx1` inside your WSL2 Linux distribution rather than native Windows. Native Windows and macOS are not supported targets.

### 1. Multi-Variable Dataset (`write_dataset` & `xarray`)

```python
import numpy as np
import xarray as xr
import wx1

t2m = np.random.uniform(250, 310, size=(24, 181, 360)).astype(np.float32)
u10 = np.random.uniform(-20, 20, size=(24, 181, 360)).astype(np.float32)

wx1.write_dataset(
    "surface.wx1",
    variables={"t2m": t2m, "u10": u10},
    dimensions={"time": 24, "latitude": 181, "longitude": 360},
    variable_dimensions={
        "t2m": ["time", "latitude", "longitude"],
        "u10": ["time", "latitude", "longitude"],
    },
    variable_attributes={
        "t2m": {"units": "K", "long_name": "2m temperature"},
        "u10": {"units": "m s-1", "long_name": "10m U wind"},
    },
    codec="zstd_shuffle",
    workers=4,
)

# Open lazily via xarray backend
ds = xr.open_dataset("surface.wx1", engine="wx1")
print(ds)
```

### 2. 4D Atmospheric Tensor & Sub-Volume Slicing

```python
import numpy as np
import wx1

temperature = np.random.randn(12, 37, 721, 1440).astype(np.float32)
wx1.write_array(
    "forecast_4d.wx1",
    temperature,
    dataset_id="era5_t",
    variable_path="/t",
    dimensions=["time", "level", "latitude", "longitude"],
    chunk_shape=[1, 1, 721, 1440],
    codec="zstd_shuffle",
    profile="restart",
    attributes={"units": "K", "long_name": "temperature"},
)

with wx1.open_file("forecast_4d.wx1") as f:
    # Read single 2D pressure level slice
    t500 = f.read_array_slice("/t", start=[0, 21, 0, 0], count=[1, 1, 721, 1440])
    # Read vertical sounding column across all 37 levels at 1 grid point
    sounding = f.read_array_slice("/t", start=[0, 0, 151, 20], count=[1, 37, 1, 1])
```

### 3. Multi-Format Conversion (GRIB1/GRIB2, NetCDF-4, Zarr v3)

```python
import wx1
import xarray as xr

wx1.from_grib("input.grib2", "converted.wx1", variable_name="t2m")
wx1.to_netcdf("converted.wx1", "output.nc")
wx1.to_zarr("converted.wx1", "output.zarr", zarr_format=3)

ds_zarr = xr.open_zarr("output.zarr")
```

---

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).
Copyright 2026 Jordy van Gils.
