# WX-1 & 2D Lorenzo Meteorological Predictor: Technical Evaluation Report
## Storage Efficiency, Decode Throughput, and Random Access for Operational NWP & ML Weather Archives

**Author:** Jordy van Gils  
**Computational Analysis & Verification:** Assisted by Google DeepMind Antigravity AI  
**Target Institutions:** Scientific Computing, Archiving & Machine Learning Teams at **KNMI** and **ECMWF**  
**Contact:** developer.tempestwx1@gmail.com  
**Software:** `tempest-wx1` 0.1.7 (Rust core + Python bindings, on PyPI, https://github.com/developer-tempest-wx1/tempest-wx1) and `wx1-lorenzo` 0.1.1 (Zarr v3 codec, on PyPI, https://github.com/developer-tempest-wx1/wx1-lorenzo)  
**Date:** November 2026  

---

## 1. Executive Summary & Objective Findings

This report evaluates the **WX-1 container** and the **2-D Lorenzo lattice predictor** for storing meteorological grids that are already quantised on a GRIB integer lattice. Baselines: GRIB1 `grid_simple`, GRIB2 `grid_ccsds` (AEC), `grid_complex_spatial_differencing` (SD1/SD2), `grid_jpeg` (JPEG2000), NetCDF-4/HDF5 and Zarr v3. All results are empirical, on the datasets and machine listed in Section 2; they are not claims about every grid or every field.

### Primary findings

1. **Bit-exact in double precision on ERA5 reanalysis data.** All **28,416 fields** of one ERA5 pressure-level download (CDS, regular lat/lon, 48 steps × 37 levels × 16 variables = 29.5 × 10⁹ points, 55.32 GB GRIB1) reconstruct with `max_abs_err = 0.0` against the ecCodes `float64` decode (and `float32`).
2. **Size (regular lat/lon, smooth upper-air fields).** Second-smallest of the lossless packings tested, after JPEG2000:
   * ERA5 (592-field subset): 407.4 MB vs 480.5 MB GRIB2-AEC (**−15.2%**) and 422.1 MB GRIB2 complex+SD2 (**−3.5%**); full 28,416-field file: 19.52 GB vs 23.03 GB (−15.2%).
   * IFS 0.25° pressure levels, 600 fields (5 variables × 120): 245.4 MB vs 280.6 MB (**−12.6%**); on the 120-field subset of Table 3.2B: −13.2% vs AEC and −12.2% vs SD2.
   * JPEG2000 is smaller still (−23% on ERA5, −28% on the IFS subset, vs AEC).
   * Not uniformly better: on the 7-bit (`r`) and 9-bit (`t`) IFS variables WX-1 2-D is **larger** than AEC (+0.8%, +4.6%), and on 9 of the 10 IFS surface variables it is larger (+3.9% to +22.5%; only `msl` is smaller, −9.4%; see `bench/results/baselines_20261117_221417.md`).
   * Tuning AEC (9 block-size/RSI combinations) saved only 0.8%; this suggests that parameter tuning alone does not close the gap, but we have not tested every AEC option.
3. **Decode speed (single thread, ecCodes call vs. WX-1 pipeline, Section 4).** WX-1 2-D decodes a 721×1440 field to `float32` in 6.9–11.3 ms, faster than libaec through ecCodes (9.4–14.0 ms) in all six test fields, and 3–11× faster than JPEG2000 (32–158 ms). **ecCodes' complex+SD2 decoder is faster than WX-1 2-D in 5 of the 6 fields (about 7–28% higher throughput).** The honest trade-off against SD2 is therefore 3–12% smaller files for 7–28% slower decoding. Encode timings are not like-for-like (Section 4.2).
4. **Message-exact archiving (ecCodes-generated messages only).** 988/988 messages (ERA5 GRIB1 PL and surface incl. bitmaps, IFS GRIB2 AEC PL and surface) were rebuilt byte-identically (SHA-256, `cmp`, `grib_compare` rc = 0). GFS complex-packing and DWD messages were **not** shown to roundtrip byte-exactly (Section 6).
5. **Random access cost is O(K), K = chunks touched.** With one field per chunk, 2-D mode must decode whole fields: measured 37-level ERA5 sounding 282.8 ms, 10-level IFS sounding 68.4 ms, 74-field ERA5 series 575.9 ms, 120-field IFS series 830.5 ms (all on variable `t`). The 1-D delta mode is much faster for these patterns (37.8 / 12.7 / 82 / 98 ms) but compresses less (−2.5% vs AEC on ERA5). Plain Zarr v3 without predictor is as fast as 1-D delta. See Table 5.1.

---
## 2. Benchmark Environment & Methodology

### 2.1 Hardware and software (as recorded on the benchmark machine)

| Item | Value |
|---|---|
| CPU | 12th Gen Intel Core i5-12500H (6 P-cores + 8 E-cores in hardware; WSL2 exposes 12 threads) |
| Memory | 24 GB (23 GiB visible in WSL2) |
| OS | Linux 6.18.58.1-microsoft-standard-WSL2 (Ubuntu), x86_64; data on Windows NTFS drives (`/mnt/d`) and ext4 copies |
| Python / NumPy | 3.12.16 / 2.5.4 |
| ecCodes | 2.50.0 (Python package `eccodes` 2.50.0 loading the `eccodeslib` 2.50.0.33 wheel's `libeccodes.so`) |
| libaec | the copy bundled in that wheel (`libaec.so.0.1.6`), called **through ecCodes**; not benchmarked standalone |
| JPEG2000 | `libopenjp2.so.7` bundled in the same wheel (version not recorded) |
| GDAL | 3.13.3 with the `libgdal-grib` plugin (independent GRIB decoder) |
| Zarr / numcodecs | 3.4.2 / 0.17.1 |
| Rust / WX-1 build | rustc 1.99.1 stable; `maturin build --release` → rustc flags `-C opt-level=3 -C strip=debuginfo` (no LTO, no `target-cpu`); zstd is the statically linked `zstd-sys` bundled by the Rust crate |
| C compiler | GCC 15.2.0 is installed, but libaec/ecCodes are prebuilt wheels, so their compiler flags are unknown |

### 2.2 Methodology and caveats

* Timings: median of *n* repeats, warm page cache, single-threaded, **not CPU-pinned**. The throughput run published here used *n* = 5 (`speed_20261117_221747.json`); the 4-D access run in Section 5 used *n* = 5.
* **ecCodes decode** = `codes_new_from_message` + `codes_get_values` on a message already in memory (header parse + unpack to float64). There is **no comparison against libaec called directly without ecCodes**; the libaec numbers include ecCodes overhead.
* **WX-1 decode** = three separately timed stages summed: read + zstd (opens the file each call), inverse predictor (NumPy), affine lattice → `float32`/`float64`. The predictor is a NumPy reference implementation; a compiled kernel would change the numbers.
* **Encode** is *not* like-for-like: WX-1 timings include lattice quantisation, predictor, zstd **and writing the container file** (buffered durability); the ecCodes timings are `codes_set_values` on an in-memory handle only (no message export, no disk write). Table 4.2 is therefore biased in favour of ecCodes.
* **GPU:** `to_device="torch"` decodes on the CPU and copies the result to CUDA memory. It is not GPU-side decompression and was not shown to be faster than reading + `.cuda()` by hand.
* **Data provenance:** ERA5 = CDS reanalysis download (regular 0.25° lat/lon, GRIB1 `grid_simple`); IFS = ECMWF Open Data 0.25° (GRIB2 AEC); GFS = NOAA 0.25° (GRIB2 complex + spatial differencing). These are public/reanalysis datasets, not the operational MARS archive.

---

## 3. Storage Efficiency & Baseline Comparisons

### 3.1 Full ERA5 pressure-level download (28,416 fields)
One complete ERA5 pressure-level download (`era5-pl.grib`, 48 hourly steps, 55.32 GB GRIB1 from the CDS) was converted and verified:

| Format / Configuration | File Size | % vs. Original GRIB1 | % vs. GRIB2-AEC | Float64 Max Abs Error |
|---|---:|---:|---:|:---:|
| **CDS ERA5 GRIB1 (`grid_simple`)** | 55.32 GB | Baseline | +140.2% | Reference |
| **GRIB2-AEC (`grid_ccsds`, ecCodes)** | 23.03 GB | -58.4% | Baseline | Verified |
| **WX-1 1D Delta (`wx1_grib_delta`)** | 22.44 GB | -59.4% | -2.5% | $\Delta = 0.0\text{e}+00$ |
| **WX-1 2D Lorenzo (`wx1_grib_2d`)** | **19.52 GB** | **-64.7%** | **-15.2%** | **$\Delta = 0.0\text{e}+00$** |

*Bit-Exactness Check:* Every one of the 28,416 fields matched ecCodes double precision identically across all 29,514,624,000 grid points.

---

### 3.2 Strong-Baseline Benchmark: Tuned AEC, Complex SD2, and JPEG2000
*(Measured on 1,390 fields across 36 variable groups; ecCodes 2.50.0; JSON artifact: `baselines_20261117_221417.json`).*

#### Table 3.2A: ERA5 reanalysis pressure levels (592 fields, 1 hour $\times$ 16 variables $\times$ 37 levels)

| Variable | Native GRIB1 | AEC Default | AEC Tuned (Best of 9) | Complex SD2 | JPEG2000 | WX-1 2D Lorenzo |
|---|---:|---:|---:|---:|---:|---:|
| `z` (Geopotential) | 76.83 MB | 32.01 MB | 32.00 MB (-0.0%) | 21.19 MB (-33.8%) | 16.44 MB (-48.7%) | **18.87 MB (-41.1%)** |
| `t` (Temperature) | 76.83 MB | 39.31 MB | 39.30 MB (-0.0%) | 31.42 MB (-20.1%) | 26.87 MB (-31.6%) | **30.46 MB (-22.5%)** |
| `u` (U-wind) | 76.83 MB | 42.58 MB | 42.57 MB (-0.0%) | 34.04 MB (-20.0%) | 29.19 MB (-31.5%) | **33.11 MB (-22.2%)** |
| `v` (V-wind) | 76.83 MB | 43.90 MB | 43.89 MB (-0.0%) | 33.98 MB (-22.6%) | 29.82 MB (-32.1%) | **34.02 MB (-22.5%)** |
| `q` (Specific humidity) | 76.83 MB | 40.69 MB | 40.63 MB (-0.2%) | 37.45 MB (-8.0%) | 32.99 MB (-18.9%) | **35.71 MB (-12.2%)** |
| `r` (Relative humidity) | 76.83 MB | 42.90 MB | 42.84 MB (-0.2%) | 38.07 MB (-11.3%) | 33.92 MB (-20.9%) | **37.50 MB (-12.6%)** |
| `o3` (Ozone) | 76.83 MB | 42.13 MB | 42.05 MB (-0.2%) | 38.18 MB (-9.4%) | 33.76 MB (-19.9%) | **36.54 MB (-13.3%)** |
| `pv` (Potential vorticity) | 76.83 MB | 33.51 MB | 33.42 MB (-0.2%) | 32.70 MB (-2.4%) | 28.12 MB (-16.1%) | **30.51 MB (-9.0%)** |
| `d` (Divergence) | 76.83 MB | 47.40 MB | 47.39 MB (-0.0%) | 44.15 MB (-6.9%) | 39.88 MB (-15.9%) | **44.17 MB (-6.8%)** |
| `cc` (Cloud cover) | 38.42 MB | 6.40 MB | 5.60 MB (-12.5%) | 6.82 MB (+6.6%) | 5.56 MB (-13.2%) | **5.93 MB (-7.3%)** |
| `ciwc` (Cloud ice) | 57.63 MB | 5.77 MB | 5.16 MB (-10.6%) | 6.03 MB (+4.4%) | 5.06 MB (-12.3%) | **5.31 MB (-8.1%)** |
| `clwc` (Cloud liquid) | 57.63 MB | 5.25 MB | 4.55 MB (-13.4%) | 5.69 MB (+8.3%) | 4.66 MB (-11.4%) | **4.82 MB (-8.3%)** |
| `crwc` (Rain water) | 76.83 MB | 3.83 MB | 3.18 MB (-17.0%) | 4.26 MB (+11.4%) | 3.37 MB (-11.8%) | **3.49 MB (-8.6%)** |
| `cswc` (Snow water) | 76.83 MB | 6.79 MB | 6.00 MB (-11.6%) | 6.88 MB (+1.3%) | 5.79 MB (-14.7%) | **6.08 MB (-10.5%)** |
| `w` (Vertical velocity) | 76.83 MB | 43.01 MB | 43.01 MB (-0.0%) | 40.01 MB (-7.0%) | 35.28 MB (-18.0%) | **39.43 MB (-8.3%)** |
| `vo` (Vorticity) | 76.83 MB | 45.02 MB | 45.01 MB (-0.0%) | 41.22 MB (-8.4%) | 37.22 MB (-17.3%) | **41.44 MB (-7.9%)** |
| **TOTAL (592 fields)** | **1152.5 MB** | **480.5 MB** | **476.6 MB (-0.8%)** | **422.1 MB (-12.2%)** | **367.9 MB (-23.4%)** | **407.4 MB (-15.2%)** |

#### Table 3.2B: ECMWF Open Data IFS 0.25° Pressure Levels (120 fields)

| Variable | Native GRIB2-AEC | AEC Tuned | Complex SD2 | JPEG2000 | WX-1 2D Lorenzo |
|---|---:|---:|---:|---:|---:|
| `gh` (Geopotential height) | 13.61 MB | 13.60 MB (-0.1%) | 11.21 MB (-17.6%) | 7.62 MB (-44.0%) | **9.49 MB (-30.3%)** |
| `u` (U-wind) | 11.92 MB | 11.91 MB (-0.0%) | 11.61 MB (-2.5%) | 9.15 MB (-23.2%) | **10.40 MB (-12.7%)** |
| `v` (V-wind) | 12.01 MB | 12.01 MB (-0.0%) | 11.21 MB (-6.7%) | 8.81 MB (-26.6%) | **10.10 MB (-15.9%)** |
| `t` (Temperature, 9-bit) | 8.41 MB | 8.40 MB (-0.1%) | 10.30 MB (+22.4%) | 6.88 MB (-18.2%) | 8.80 MB (+4.6%) |
| `r` (Relative humidity, 7-bit)| 7.96 MB | 7.92 MB (-0.5%) | 8.96 MB (+12.7%) | 6.58 MB (-17.3%) | 8.02 MB (+0.8%) |
| **TOTAL (120 fields)** | **53.90 MB** | **53.85 MB (-0.1%)** | **53.30 MB (-1.1%)** | **39.04 MB (-27.6%)** | **46.81 MB (-13.2%)** |

---

## 4. Single-Core Decode and Encode Throughput

*(Measured on single core of Intel i5-12500H, warm cache, single thread, $n=5$ repeats; JSON artifact: `speed_20261117_221747.json`; see Section 2.2 for what each timing includes).*

### Table 4.1: Decode Throughput to In-Memory `float32` (4.15 MB per $721 \times 1440$ field)

| Test Field | GRIB2-AEC (libaec) | GRIB2 Complex SD2 | GRIB2 JPEG2000 | WX-1 2D Lorenzo $\to$ f32 | WX-1 1D Delta $\to$ f32 | WX-1 Int Only $\to$ f32 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **ERA5 `t` 500 hPa (16-bit)** | 14.04 ms (296 MB/s) | 9.21 ms (451 MB/s) | 112.5 ms (37 MB/s) | **11.14 ms (373 MB/s)** | 7.90 ms (526 MB/s) | 6.23 ms (666 MB/s) |
| **ERA5 `z` 500 hPa (16-bit)** | 13.97 ms (297 MB/s) | 9.01 ms (461 MB/s) | 82.32 ms (50 MB/s) | **9.73 ms (427 MB/s)** | 8.84 ms (470 MB/s) | 5.70 ms (728 MB/s) |
| **ERA5 `q` 850 hPa (16-bit)** | 11.89 ms (349 MB/s) | 8.88 ms (468 MB/s) | 157.8 ms (26 MB/s) | **11.30 ms (367 MB/s)** | 10.39 ms (400 MB/s) | 6.00 ms (693 MB/s) |
| **IFS `t` 500 hPa (9-bit)** | 9.40 ms (442 MB/s) | 7.37 ms (563 MB/s) | 32.24 ms (129 MB/s) | **6.94 ms (598 MB/s)** | 6.91 ms (601 MB/s) | 4.60 ms (903 MB/s) |
| **IFS `u` 500 hPa (11-bit)** | 11.29 ms (368 MB/s) | 8.07 ms (515 MB/s) | 65.01 ms (64 MB/s) | **8.61 ms (482 MB/s)** | 6.87 ms (605 MB/s) | 4.09 ms (1015 MB/s) |
| **IFS `gh` 500 hPa (12-bit)** | 10.08 ms (412 MB/s) | 6.67 ms (622 MB/s) | 33.79 ms (123 MB/s) | **7.63 ms (544 MB/s)** | 6.10 ms (680 MB/s) | 4.07 ms (1020 MB/s) |

### Table 4.2: Encode time (**not like-for-like**, see note)

| Test Field | GRIB2-AEC (libaec) | GRIB2 Complex SD2 | GRIB2 JPEG2000 | WX-1 2D Lorenzo | WX-1 1D Delta |
|---|:---:|:---:|:---:|:---:|:---:|
| **ERA5 `t` 500 hPa** | 11.90 ms (349 MB/s) | 9.56 ms (434 MB/s) | 134.3 ms (31 MB/s) | **13.75 ms (302 MB/s)** | 14.57 ms (285 MB/s) |
| **ERA5 `z` 500 hPa** | 10.60 ms (392 MB/s) | 9.00 ms (462 MB/s) | 94.34 ms (44 MB/s) | **16.34 ms (254 MB/s)** | 14.41 ms (288 MB/s) |
| **ERA5 `q` 850 hPa** | 10.25 ms (405 MB/s) | 159.1 ms (26 MB/s)* | 175.3 ms (24 MB/s) | **18.63 ms (223 MB/s)** | 15.83 ms (262 MB/s) |
| **IFS `t` 500 hPa** | 9.70 ms (428 MB/s) | 10.33 ms (402 MB/s) | 38.39 ms (108 MB/s) | **10.94 ms (379 MB/s)** | 12.45 ms (334 MB/s) |
| **IFS `u` 500 hPa** | 9.91 ms (419 MB/s) | 14.09 ms (295 MB/s) | 74.68 ms (56 MB/s) | **14.90 ms (279 MB/s)** | 12.79 ms (325 MB/s) |
| **IFS `gh` 500 hPa** | 8.86 ms (469 MB/s) | 7.90 ms (526 MB/s) | 37.84 ms (110 MB/s) | **11.80 ms (352 MB/s)** | 11.86 ms (350 MB/s) |

*WX-1 columns include quantisation + predictor + zstd + **container file write**; the three ecCodes columns are `codes_set_values` on an in-memory handle (no export, no disk write). Read this table as "WX-1 including I/O takes about 1.1–1.8× the time of ecCodes AEC re-encoding without I/O", not as a ranking. The ecCodes SD2 outlier (159 ms on ERA5 `q`) is a single measurement.*

### 4.3 Trade-off versus GRIB2 complex packing with spatial differencing (SD2)

| | WX-1 2-D vs. ecCodes SD2 |
|---|---|
| Size, ERA5 592 fields | −3.5% (407.4 vs 422.1 MB) |
| Size, IFS 120-field subset | −12.2% (46.81 vs 53.30 MB) |
| Decode throughput, 6 test fields | SD2 higher in 5 of 6 (about +7% to +28%); WX-1 higher on IFS `t` (9-bit) by 6% |

WX-1 2-D is therefore a **size-for-speed trade against SD2** (and a speed-for-size trade against JPEG2000), not a dominating option. Note that SD2 messages also did not reproduce byte-exactly through our rebuild path (Section 6).

---

## 5. Random Access & 4-D Slicing

Access time scales with **O(K)**, K = number of chunks that must be decoded. All values below are **measured medians (ms, warm cache)**; the **fastest value in each row is bold**.

### Table 5.1: Measured access time (ms)

| pattern | WX-1 1-D delta | Zarr v3 blosc-zstd | Zarr v3 sharded | NetCDF-4 zlib4 | HDF5 gzip4 | GRIB2 AEC, indexed | GRIB2 AEC, re-encoded | WX-1 2-D (separate run) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ERA5 2-D map (1 field) | 9.9 | 6.4 | 6.8 | 16.3 | 15.0 | **6.2** | 21.7 | 7.9 |
| ERA5 sounding (37 levels) | **37.8** | 41.1 | 43.7 | 423.2 | 446.8 | 205.0 | 469.6 | 282.8 |
| ERA5 series (74 fields) | **82.4** | 87.9 | 96.8 | 841.9 | 892.6 | 379.8 | 932.4 | 575.9 |
| IFS 2-D map (1 field) | 8.7 | **5.4** | 6.1 | 11.9 | 9.0 | 10.3 | 12.2 | 7.1 |
| IFS sounding (10 levels) | 12.7 | 12.3 | **11.1** | 82.2 | 85.2 | 103.1 | 97.6 | 68.4 |
| IFS series (120 fields) | **98.4** | 127.6 | 108.7 | 958.5 | 996.4 | 1205.7 | 1218.7 | 830.5 |


Notes:
* Columns 1–7 come from `formats_era5_pl_all16_4steps_20261117_200733.json` (ERA5 `t`, 2 steps × 37 levels) and `formats_ecmwf_0p25_2026111618_7a39663d_pl_20261117_202659.json` (IFS `t`, 12 steps × 10 levels); *n* = 5. Zarr/NetCDF/HDF5/WX-1 1-D were stored as the benchmark's native layouts (Zarr without predictor, one 721×1440 slab per chunk).
* The last column is a **separate run** of `bench/bench_access_2d.py` (`access_2d_20261118_110432.json`, *n* = 5) with one field per chunk, using the **identical variable `t`** on both ERA5 (`2 × 37` fields) and IFS (`12 × 10` fields).
* "sounding" = all levels of one step at one grid point; "series" = all steps and levels at one point. Reading one point still decodes whole fields.
* Same 2-D runs for context: ERA5 series at one level (2 fields) 15.2 ms; IFS series at one level (12 fields) 81.7 ms. For comparison, **ecCodes decoding messages already in memory** (no file scan/index): ERA5 sounding 162.6 ms, ERA5 series 336.5 ms, IFS sounding 95.6 ms, IFS series 1,122.2 ms (ERA5 source is GRIB1 simple packing, which unpacks faster than AEC).

### What this shows
* **2-D mode loses on deep soundings of cheap-to-decode data:** the 37-level ERA5 sounding (282.8 ms) is slower than indexed GRIB2 (205 ms) and than in-memory ecCodes decode of the GRIB1 source (162.6 ms), and ~7× slower than plain Zarr v3 (41 ms). It wins against NetCDF-4/HDF5 (≈ 420–450 ms) and against GRIB2 AEC messages (IFS: 68.4 vs 98–103 ms).
* **Zarr v3 without a predictor is as fast as WX-1 1-D mode** (and faster on 2-D maps: 6.4 vs 9.9 ms ERA5, 5.4 vs 8.7 ms IFS). WX-1 1-D is fastest in 3 of 6 rows only.
* Smaller tiles (e.g. 64×64 chunks) would reduce the work per point read, at a compression cost; **this was not measured**.

---
## 6. Scope & Limitations

1. **Regular lat/lon grids only.** The predictor assumes a rectangular (y, x) lattice. It was tested on 0.25° regular lat/lon (ERA5, IFS Open Data, GFS). It does **not** apply to reduced Gaussian grids (e.g. O1280 octahedral, where row length varies; a per-ring 1-D delta would be needed), nor to spectral coefficients. Statements about size savings must not be extrapolated to the MARS archive, which is largely reduced Gaussian and spectral.
2. **Only data already on a GRIB lattice.** Bit-exactness and the size figures apply to fields whose values are exactly `(R + X·2^E)·10^−D` with X ≤ 16 bit. Bits per value > 16 and fields that fail this check fall back to verbatim storage.
3. **Byte-exact message roundtrip is proven only for ecCodes-written GRIB1 `grid_simple` and GRIB2 `grid_ccsds`** (988/988 messages). For NOAA GFS (`grid_complex_spatial_differencing`) the *values* were verified bit-exact (570/570 messages, float64), but our rebuild of the *message* re-packs the data through ecCodes and does not reproduce the original bitstream (a test rebuild differed in decoded values by up to 0.24), and a 20-message GFS file also hit the 32 MiB metadata limit when messages are kept verbatim. Message-level archiving of GFS, DWD, bitmapped GRIB2 and other producers is therefore **not supported yet**. GDAL was run on 12 sampled messages per dataset: GDAL's decode of the original and of the rebuilt file was identical (12/12), but its values did not match the ecCodes float64 decode for most samples (0/12 ERA5, 7–9/12 IFS; differences such as 273.15 look like GDAL's K→°C conversion, the larger ERA5 differences were not investigated). The GDAL check therefore only shows that original and rebuilt files are equivalent to GDAL; it is **not** an independent value verification, and we do not claim one.
4. **Unquantised `float32` model output** (NeuralGCM, JAX, PyTorch): lossless WX-1 is at parity with NetCDF-4 `zlib4` and Zarr v3 blosc-zstd in our measurements (within about ±1%), so no benefit is claimed.
5. **Not tested:** other CPU architectures, other decoders than ecCodes/GDAL for the full set, GRIB2 multi-field-per-message files, DWD ICON data (the benchmark ran on ICON-EU files for format sizes only).

---
## 7. Adoption Routes (in order of realism)

```mermaid
flowchart LR
    A["Route 1: wx1-lorenzo Zarr v3 codec"] --> B["Route 2: WX-1 container for ML pipelines"] --> C["Route 3: standardisation (WMO)"]
```

**Route 1 (realistic, small): a standalone Zarr v3 codec.** `wx1-lorenzo` is a native zarr-python ≥ 3.1 `ArrayArrayCodec` named `wx1.lorenzo2d` (it is **not** a `numcodecs.abc.Codec`). Importing the package registers it with `zarr.registry.register_codec`; the installed package also advertises it via the `zarr.codecs` entry point. Arrays are plain Zarr v3 directories; a reader needs the package. Measured end-to-end (Section 8.1): bit-exact, −12.6% vs the GRIB2-AEC messages on 600 IFS fields. Status: published on PyPI (`pip install wx1-lorenzo`), 8 unit tests pass in a fresh virtual environment, Apache-2.0 licence file included; **not yet registered in zarr-extensions** (PR prepared).

**Route 2: the WX-1 container** for ML training pipelines that want chunked random access with atomic appends. Gains depend on layout (Section 5); GPU benefit not shown.

**Route 3 (long-term, multi-year): a GRIB2 packing template.** This would need a C reference implementation in ecCodes and a WMO approval process (the former Commission for Basic Systems' responsibilities have been with INFCOM since 2019). We make no schedule claim.

---
## 8. Examples (each executed in a fresh virtual environment)

Environment: `python -m venv` with `pip install zarr==3.4.2 numpy numcodecs eccodes wx1-lorenzo==0.1.1 tempest-wx1==0.1.7`.

### 8.1 Zarr v3 + Lorenzo2D, end to end: write → read back → bit-exact check → size vs GRIB2-AEC

```python
# examples/grib_to_zarr_e2e.py in the wx1-lorenzo repository (python grib_to_zarr_e2e.py FILE.grib2 gh out.zarr)
import shutil, sys, os
import eccodes as ec, numpy as np, zarr
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.codecs.numcodecs import Shuffle
import wx1_lorenzo as wl            # importing registers the codec "wx1.lorenzo2d"

path, short, out = sys.argv[1:4]
fields, sizes, params, refs = [], 0, [], []
with open(path, "rb") as f:
    while (gid := ec.codes_grib_new_from_file(f)) is not None:
        if ec.codes_get(gid, "shortName") == short:
            R, E, D = wl.lattice_params(gid)
            v = ec.codes_get_values(gid).reshape(int(ec.codes_get(gid, "Nj")), int(ec.codes_get(gid, "Ni")))
            X = wl.to_lattice(v, R, E, D)
            assert np.array_equal(wl.from_lattice(X, R, E, D), v), "field is not on its GRIB lattice"
            fields.append(X); refs.append(v); params.append([R, E, D])
            sizes += len(ec.codes_get_message(gid))
        ec.codes_release(gid)
X = np.stack(fields)
shutil.rmtree(out, ignore_errors=True)
z = zarr.create_array(out, shape=X.shape, chunks=(1, *X.shape[1:]), dtype="uint16", zarr_format=3,
                      filters=[wl.Lorenzo2DCodec()], serializer=BytesCodec(),
                      compressors=[Shuffle(elementsize=2), ZstdCodec(level=3)],
                      attributes={"grib_lattice_R_E_D": params})
z[:] = X
z2 = zarr.open_array(out, mode="r")                      # fresh handle
ok = all(np.array_equal(wl.from_lattice(z2[i], *p), refs[i]) for i, p in enumerate(z2.attrs["grib_lattice_R_E_D"]))
nbytes = sum(os.path.getsize(os.path.join(r, n)) for r, _, ns in os.walk(out) for n in ns)
print(f"{short}: {len(fields)} fields  GRIB(AEC)={sizes/1e6:.2f} MB  Zarr+Lorenzo2D={nbytes/1e6:.2f} MB "
      f"({100*(nbytes/sizes-1):+.1f}%)  float64 bit-exact vs ecCodes: {ok}")
```

Output on `ecmwf_0p25_2026111618_7a39663d_pl.grib2` (IFS Open Data, 0.25°, 10 levels × 12 steps per variable):

| variable | fields | GRIB2-AEC (MB) | Zarr + Lorenzo2D (MB) | Δ size | float64 bit-exact vs ecCodes |
|---|---:|---:|---:|---:|:---:|
| `u` | 120 | 69.15 | 60.73 | −12.2% | yes |
| `v` | 120 | 75.81 | 63.47 | −16.3% | yes |
| `gh` | 120 | 54.40 | 37.90 | −30.3% | yes |
| `t` (9-bit) | 120 | 33.43 | 34.93 | **+4.5%** | yes |
| `r` (7-bit) | 120 | 47.82 | 48.28 | **+1.0%** | yes |
| **total** | 600 | **280.61** | **245.31** | **−12.6%** | yes |

The codec chain in `zarr.json` is `wx1.lorenzo2d → bytes → numcodecs.shuffle → zstd`. Sizes count all files in the Zarr directory (chunks + metadata).

### 8.2 Message-exact archive with WX-1 (ecCodes-written GRIB only)

```python
import hashlib, eccodes as E
from wx1.adapters import grib_exact

raw_messages = []
with open("ecmwf_0p25_pl.grib2", "rb") as f:
    while (gid := E.codes_grib_new_from_file(f)) is not None:
        raw_messages.append(E.codes_get_message(gid)); E.codes_release(gid)
        if len(raw_messages) == 120: break          # first 120 messages (tested size)

stats = grib_exact.write_grib_messages("archive.wx1", raw_messages, mode="2d", workers=4)
print(stats)
rebuilt = grib_exact.read_grib_messages("archive.wx1")
assert all(hashlib.sha256(a).digest() == hashlib.sha256(b).digest() for a, b in zip(raw_messages, rebuilt))
```

### 8.3 Reading one field or a vertical profile from the archive

The file stores predictor residuals in variable `/fields` (shape `[messages, ny, nx]`, one field per chunk). Slicing it directly returns residuals, not values; use `read_field`:

```python
from wx1.adapters import grib_exact
# message order = original file order; here 10 levels of one step
profile = [grib_exact.read_field("archive.wx1", i)[150, 300] for i in range(10)]   # float64, bit-identical to ecCodes
```

Each call decodes one full field (~7 ms); a 10-level profile took 68.4 ms in Section 5.

---
## 9. Conclusions

1. On regular lat/lon pressure-level fields, a 2-D Lorenzo predictor on the GRIB lattice produced files 12.6–15.2% smaller than GRIB2-AEC, bit-exact in float64, in the datasets tested. It is not smaller than AEC for the 7- and 9-bit IFS variables and several surface fields, and JPEG2000 is smaller still.
2. Against GRIB2 complex packing with SD2 the trade is 3–12% smaller files for roughly 7–28% slower decoding; against JPEG2000 it is larger files for 3–11× faster decoding.
3. Random access depends on the layout: 2-D mode needs whole-field decodes (37-level sounding ≈ 283 ms); a 1-D predictor or plain Zarr is faster for point/profile access and compresses less.
4. The codec route (Route 1) is the one that can be adopted without a new container. Before external use it still needs: zarr-extensions registration, independent runs on other hardware and by other people, tests on reduced Gaussian grids (where a different predictor is needed), and message-level support for non-ecCodes producers.

*Analysis, scripts and drafting were done with AI assistance (Google DeepMind Antigravity); the author is responsible for the content. All numbers refer to the JSON/markdown files in `bench/results/`.*
