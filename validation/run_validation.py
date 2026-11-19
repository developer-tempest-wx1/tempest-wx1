#!/usr/bin/env python3
"""WX-1 End-to-End Validation Kit for KNMI and ECMWF Evaluators.

Runs a reproducible validation suite on the local machine and filesystem:
  1. Verifies the installed `tempest-wx1` package (and pytest suite when run in-repo).
  2. Verifies the v0.1.7 stale-temp-file same-PID regression and `from_netcdf` warning.
  3. Runs a SIGKILL (`kill -9`) crash-consistency test on WX-1 vs. NetCDF-4 and Zarr v3.
  4. Validates user-supplied GRIB1/GRIB2 files via `wx1.adapters.grib_exact` (1-D delta
     and 2-D Lorenzo): measures size vs. original GRIB, verifies `float64`/`float32`
     bit-exactness against ecCodes (`max_abs_err == 0.0`), and reports SHA-256
     byte-identical message counts.
  5. Validates user-supplied (or synthetic) NetCDF-4 files for lossless roundtrip.

The generated Markdown report intentionally omits hostnames, usernames, and
directory paths (logging basenames only) so it can be shared externally.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any


def _ensure_wx1_importable() -> None:
    try:
        import wx1  # noqa: F401
        return
    except ImportError:
        repo_py = Path(__file__).resolve().parents[1] / "python"
        if (repo_py / "wx1").is_dir():
            sys.path.insert(0, str(repo_py))
        import wx1  # noqa: F401


_ensure_wx1_importable()

import numpy as np  # noqa: E402
import wx1  # noqa: E402
from wx1.adapters import grib_exact  # noqa: E402


def _fmt_mib(nbytes: int) -> str:
    return f"{nbytes / (1024.0 * 1024.0):.2f} MiB"


def _fmt_pct(new_bytes: int, ref_bytes: int) -> str:
    if ref_bytes <= 0:
        return "n/a"
    return f"{((new_bytes - ref_bytes) / ref_bytes) * 100.0:+.1f}%"


def run_pytest_suite() -> dict[str, Any]:
    """Run pytest python/tests/ when executed inside the tempest-wx1 repository."""
    repo_root = Path(__file__).resolve().parents[1]
    tests_dir = repo_root / "python" / "tests"
    if not tests_dir.is_dir():
        return {"ran": False, "reason": "Not inside source repository"}

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "python")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-q"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    summary_line = ""
    for line in reversed((proc.stdout + "\n" + proc.stderr).splitlines()):
        if "passed" in line or "failed" in line or "error" in line.lower():
            summary_line = line.strip()
            break
    return {
        "ran": True,
        "returncode": proc.returncode,
        "summary": summary_line or f"exit code {proc.returncode}",
    }


def verify_v017_regressions(work_dir: Path) -> dict[str, Any]:
    """Verify same-PID leftover temp-file tolerance and multi-var NetCDF warning."""
    reg_dir = work_dir / "v017_regressions"
    reg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leftover temp files with identical PID must never block writes
    target = reg_dir / "container.wx1"
    pid = os.getpid()
    for kind in ("wx1-tmp", "wx1-commit-tmp", "wx1-generation-tmp", "wx1-generation-stream-tmp"):
        (reg_dir / f"container.wx1.{kind}-{pid}").write_bytes(b"crashed-partial")

    arr = np.arange(256, dtype=np.float32).reshape(16, 16)
    wx1.write_array(
        target,
        arr,
        dataset_id="stale_pid_check",
        variable_path="/field",
        dimensions=["y", "x"],
        chunk_shape=[8, 8],
        durability="durable",
    )
    with wx1.open_file(target) as f:
        pid_collision_ok = bool(np.array_equal(f.read_array("/field"), arr))

    # 2. from_netcdf warning check when netCDF4 is available
    netcdf_warn_ok: str
    try:
        import netCDF4

        nc_path = reg_dir / "multi_var.nc"
        with netCDF4.Dataset(nc_path, "w", format="NETCDF4") as ds:
            ds.createDimension("lat", 4)
            ds.createDimension("lon", 4)
            for vname in ("t2m", "u10", "v10"):
                ds.createVariable(vname, "f4", ("lat", "lon"))[:] = np.ones((4, 4), dtype=np.float32)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wx1.from_netcdf(nc_path, reg_dir / "nc_auto.wx1").close()
        warn_msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
        netcdf_warn_ok = (
            "PASS"
            if any("did NOT import 2 other data variable" in m and "u10" in m for m in warn_msgs)
            else "FAIL"
        )
    except ImportError:
        netcdf_warn_ok = "SKIPPED (netCDF4 not installed)"

    return {
        "stale_temp_same_pid_ok": pid_collision_ok,
        "from_netcdf_warning": netcdf_warn_ok,
    }


_CRASH_WORKER_SCRIPT = r"""
import os, sys, time
import numpy as np

fmt, path, n_steps, ny, nx, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
seed = 200 if mode == "v2" else 100
data = (np.arange(n_steps * ny * nx, dtype=np.float32).reshape(n_steps, ny, nx) + np.float32(seed))

if fmt == "wx1":
    import wx1
    print("READY", flush=True)
    wx1.write_array(
        path, data, dataset_id="crash_test", variable_path="/field",
        dimensions=["t", "y", "x"], chunk_shape=[1, ny, nx],
        codec="zstd_shuffle", workers=2, durability="durable",
    )
elif fmt == "zarr":
    import zarr
    print("READY", flush=True)
    arr = zarr.open_array(path, mode="w", shape=data.shape, chunks=(1, ny, nx), dtype="float32", fill_value=np.float32(-999.0))
    for i in range(n_steps):
        arr[i] = data[i]
elif fmt == "netcdf4":
    import netCDF4
    print("READY", flush=True)
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("t", n_steps)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        v = ds.createVariable("field", "f4", ("t", "y", "x"), chunksizes=(1, ny, nx), zlib=True, complevel=1, fill_value=np.float32(-999.0))
        for i in range(n_steps):
            v[i] = data[i]
            ds.sync()
"""


def _read_crash_target(fmt: str, path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    if fmt == "wx1":
        with wx1.open_file(path) as f:
            f.validate()
            return np.asarray(f.read_array("/field"), dtype=np.float32)
    if fmt == "zarr":
        import zarr
        arr = zarr.open_array(str(path), mode="r")
        return np.asarray(arr[:], dtype=np.float32)
    if fmt == "netcdf4":
        import netCDF4
        with netCDF4.Dataset(path, "r") as ds:
            var = ds.variables["field"]
            try:
                var.set_auto_maskandscale(False)
            except Exception:
                pass
            return np.asarray(var[:], dtype=np.float32)
    raise ValueError(fmt)


def _format_available(fmt: str) -> bool:
    if fmt == "wx1":
        return True
    try:
        __import__("netCDF4" if fmt == "netcdf4" else "zarr")
        return True
    except ImportError:
        return False


def run_crash_consistency(work_dir: Path, trials_per_format: int) -> list[dict[str, Any]]:
    """Run SIGKILL crash-consistency trials across WX-1, NetCDF-4, and Zarr v3."""
    crash_dir = work_dir / "crash_consistency"
    crash_dir.mkdir(parents=True, exist_ok=True)
    worker = crash_dir / "worker.py"
    worker.write_text(_CRASH_WORKER_SCRIPT, encoding="utf-8")

    n_steps, ny, nx = 40, 128, 256
    v1 = (np.arange(n_steps * ny * nx, dtype=np.float32).reshape(n_steps, ny, nx) + np.float32(100))
    v2 = (np.arange(n_steps * ny * nx, dtype=np.float32).reshape(n_steps, ny, nx) + np.float32(200))
    rng = random.Random(2026)
    results = []

    def _run_clean_and_time_write(fmt_name: str, tgt: Path, ver: str) -> float:
        p = subprocess.Popen(
            [sys.executable, str(worker), fmt_name, str(tgt), str(n_steps), str(ny), str(nx), ver],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert p.stdout is not None
        p.stdout.readline()
        t_start = time.perf_counter()
        rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"clean write worker failed for {fmt_name} (rc={rc})")
        return time.perf_counter() - t_start

    for fmt, ext in [("wx1", ".wx1"), ("zarr", ".zarr"), ("netcdf4", ".nc")]:
        if not _format_available(fmt):
            results.append({"format": fmt, "skipped": True, "trials": 0})
            continue

        target = crash_dir / f"target{ext}"
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        write_dur = max(0.010, _run_clean_and_time_write(fmt, target, "v1"))

        intact_v1 = intact_v2 = caught_error = silent_corruption = followup_blocked = 0
        for _ in range(trials_per_format):
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
            _run_clean_and_time_write(fmt, target, "v1")
            proc = subprocess.Popen(
                [sys.executable, str(worker), fmt, str(target), str(n_steps), str(ny), str(nx), "v2"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            assert proc.stdout is not None
            proc.stdout.readline()  # synchronize right before write starts
            time.sleep(rng.uniform(0.0, write_dur * 0.85))
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()

            try:
                got = _read_crash_target(fmt, target)
            except Exception:
                caught_error += 1
            else:
                if got is None:
                    caught_error += 1
                elif np.array_equal(got, v1):
                    intact_v1 += 1
                elif np.array_equal(got, v2):
                    intact_v2 += 1
                else:
                    silent_corruption += 1

            # Verify a follow-up write can still succeed cleanly after the crash.
            if fmt == "wx1":
                followup = subprocess.run(
                    [sys.executable, str(worker), fmt, str(target), "4", "64", "64", "v2"],
                    capture_output=True,
                )
                if followup.returncode != 0:
                    followup_blocked += 1

        results.append(
            {
                "format": fmt,
                "skipped": False,
                "trials": trials_per_format,
                "intact_v1": intact_v1,
                "intact_v2": intact_v2,
                "caught_error": caught_error,
                "silent_corruption": silent_corruption,
                "followup_blocked": followup_blocked,
            }
        )
    return results


def _split_grib_messages(path: Path) -> list[bytes]:
    import eccodes

    msgs: list[bytes] = []
    with open(path, "rb") as f:
        while (gid := eccodes.codes_grib_new_from_file(f)) is not None:
            try:
                msgs.append(eccodes.codes_get_message(gid))
            finally:
                eccodes.codes_release(gid)
    return msgs


def _make_synthetic_grib2(out_path: Path) -> Path:
    import eccodes

    rng = np.random.default_rng(7)
    with open(out_path, "wb") as fout:
        for lvl in (500, 700, 850, 925):
            gid = eccodes.codes_grib_new_from_samples("regular_ll_pl_grib2")
            try:
                eccodes.codes_set(gid, "shortName", "t")
                eccodes.codes_set(gid, "typeOfLevel", "isobaricInhPa")
                eccodes.codes_set(gid, "level", lvl)
                eccodes.codes_set(gid, "bitsPerValue", 16)
                ni, nj = int(eccodes.codes_get(gid, "Ni")), int(eccodes.codes_get(gid, "Nj"))
                lat = np.linspace(-1.0, 1.0, nj)[:, None]
                lon = np.linspace(-1.0, 1.0, ni)[None, :]
                smooth = 270.0 + 10.0 * np.cos(lat) * np.sin(lon) + 0.05 * rng.standard_normal((nj, ni))
                eccodes.codes_set_values(gid, smooth.ravel())
                eccodes.codes_write(gid, fout)
            finally:
                eccodes.codes_release(gid)
    return out_path


def validate_grib_file(grib_path: Path, work_dir: Path) -> dict[str, Any]:
    """Validate one GRIB1/GRIB2 file on both 1-D delta and 2-D Lorenzo modes."""
    import eccodes

    t0 = time.perf_counter()
    raw_msgs = _split_grib_messages(grib_path)
    src_bytes = sum(len(m) for m in raw_msgs)
    editions = set()
    packings = set()
    grids = set()
    for m in raw_msgs[:8]:
        gid = eccodes.codes_new_from_message(m)
        try:
            editions.add(int(eccodes.codes_get(gid, "editionNumber")))
            packings.add(str(eccodes.codes_get(gid, "packingType")))
            if eccodes.codes_is_defined(gid, "Nj") and eccodes.codes_is_defined(gid, "Ni"):
                grids.add(f"{eccodes.codes_get(gid, 'Nj')}x{eccodes.codes_get(gid, 'Ni')}")
        finally:
            eccodes.codes_release(gid)

    out = {
        "name": grib_path.name,
        "messages": len(raw_msgs),
        "editions": sorted(editions),
        "packings": sorted(packings),
        "grids": sorted(grids),
        "source_bytes": src_bytes,
    }

    for mode in ("1d", "2d"):
        wx1_path = work_dir / f"{grib_path.stem}.{mode}.wx1"
        stats = grib_exact.write_grib_messages(wx1_path, raw_msgs, mode=mode, workers=4)
        rebuilt_msgs = grib_exact.read_grib_messages(wx1_path)

        byte_identical = 0
        f64_identical = 0
        f32_identical = 0
        max_abs_err = 0.0

        for idx, (orig, back) in enumerate(zip(raw_msgs, rebuilt_msgs)):
            if hashlib.sha256(orig).digest() == hashlib.sha256(back).digest():
                byte_identical += 1
            ga = eccodes.codes_new_from_message(orig)
            gb = eccodes.codes_new_from_message(back)
            try:
                for g in (ga, gb):
                    if eccodes.codes_is_defined(g, "bitmapPresent") and eccodes.codes_get(g, "bitmapPresent"):
                        eccodes.codes_set_double(g, "missingValue", float("nan"))
                va = eccodes.codes_get_values(ga)
                vb = eccodes.codes_get_values(gb)
                f64_identical += int(np.array_equal(va, vb, equal_nan=True))
                f32_identical += int(np.array_equal(va.astype(np.float32), vb.astype(np.float32), equal_nan=True))
                finite = ~np.isnan(va) & ~np.isnan(vb)
                if np.any(finite):
                    max_abs_err = max(max_abs_err, float(np.max(np.abs(va[finite] - vb[finite]))))
            finally:
                eccodes.codes_release(ga)
                eccodes.codes_release(gb)

            # Spot-check random-access single-field read_field() on first and last lattice field
            if idx in (0, len(raw_msgs) - 1):
                try:
                    rf = grib_exact.read_field(wx1_path, idx, dtype="float64")
                    ga = eccodes.codes_new_from_message(orig)
                    try:
                        if eccodes.codes_is_defined(ga, "bitmapPresent") and eccodes.codes_get(ga, "bitmapPresent"):
                            eccodes.codes_set_double(ga, "missingValue", float("nan"))
                        va = eccodes.codes_get_values(ga).reshape(rf.shape)
                        assert np.array_equal(rf, va, equal_nan=True)
                    finally:
                        eccodes.codes_release(ga)
                except ValueError:
                    pass  # verbatim raw fallback field

        out[mode] = {
            "file_bytes": stats["file_bytes"],
            "lattice": stats["lattice"],
            "raw_fallback": stats["raw_fallback"],
            "byte_identical": byte_identical,
            "f64_identical": f64_identical,
            "f32_identical": f32_identical,
            "max_abs_err": max_abs_err,
        }
    out["elapsed_s"] = round(time.perf_counter() - t0, 2)
    return out


def validate_netcdf_file(nc_path: Path, work_dir: Path) -> dict[str, Any]:
    """Validate lossless import + roundtrip export on a NetCDF-4 file."""
    import netCDF4

    with netCDF4.Dataset(nc_path, "r") as ds:
        multi_vars = [k for k, v in ds.variables.items() if v.ndim >= 2]
        if not multi_vars:
            multi_vars = list(ds.variables.keys())[:1]
        target_var = multi_vars[0]
        v = ds.variables[target_var]
        try:
            v.set_auto_maskandscale(False)
        except Exception:
            pass
        raw = v[:]
        if hasattr(raw, "filled"):
            fv = getattr(v, "_FillValue", np.nan)
            orig_arr = np.asarray(raw.filled(fv))
        else:
            orig_arr = np.asarray(raw)

    wx1_out = work_dir / f"{nc_path.stem}.wx1"
    nc_rt = work_dir / f"{nc_path.stem}.roundtrip.nc"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wx1.from_netcdf(nc_path, wx1_out, variable_name=target_var).close()
    wx1.to_netcdf(wx1_out, nc_rt)

    with netCDF4.Dataset(nc_rt, "r") as ds_rt:
        vrt = ds_rt.variables[target_var]
        try:
            vrt.set_auto_maskandscale(False)
        except Exception:
            pass
        rt_raw = vrt[:]
        if hasattr(rt_raw, "filled"):
            fv = getattr(vrt, "_FillValue", np.nan)
            rt_arr = np.asarray(rt_raw.filled(fv))
        else:
            rt_arr = np.asarray(rt_raw)

    exact = bool(np.array_equal(orig_arr, rt_arr, equal_nan=True))
    return {
        "name": nc_path.name,
        "variable": target_var,
        "all_2d_plus_vars": len(multi_vars),
        "shape": list(orig_arr.shape),
        "dtype": str(orig_arr.dtype),
        "nc_bytes": nc_path.stat().st_size,
        "wx1_bytes": wx1_out.stat().st_size,
        "exact_roundtrip": exact,
        "unexpected_warnings": len(caught),
    }


def render_report(
    pytest_res: dict[str, Any],
    reg_res: dict[str, Any],
    crash_res: list[dict[str, Any]],
    grib_res: list[dict[str, Any]],
    nc_res: list[dict[str, Any]],
    grib_skipped_reason: str | None,
    nc_skipped_reason: str | None,
) -> str:
    """Render an anonymized Markdown validation report (no usernames or host paths)."""
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# WX-1 Validation Report",
        "",
        f"- **Timestamp:** {ts}",
        f"- **tempest-wx1 version:** `{wx1.__version__}`",
        f"- **Platform:** `{platform.system()} {platform.machine()}` | **Python:** `{platform.python_version()}`",
        "",
        "## 1. Test Suite & v0.1.7 Regression Checks",
        "",
    ]
    if pytest_res["ran"]:
        status = "PASS" if pytest_res["returncode"] == 0 else f"FAIL (rc={pytest_res['returncode']})"
        lines.append(f"- **Repository `pytest` suite:** `{status}` ({pytest_res['summary']})")
    else:
        lines.append(f"- **Repository `pytest` suite:** skipped ({pytest_res['reason']})")
    lines.append(f"- **Stale temp-file same-PID write tolerance (`v0.1.7`):** `{'PASS' if reg_res['stale_temp_same_pid_ok'] else 'FAIL'}`")
    lines.append(f"- **`from_netcdf` multi-variable warning (`v0.1.7`):** `{reg_res['from_netcdf_warning']}`")
    lines.append("")

    lines.extend([
        "## 2. Crash-Consistency Test (`kill -9` / `SIGKILL` mid-write overwrite)",
        "",
        "Each trial restores a committed dataset `v1`, launches a child writer overwriting it with `v2`, and sends `SIGKILL` mid-write.",
        "**Silent corruption** means a reader opened the file without error and returned neither intact `v1` nor intact `v2`.",
        "",
        "| Format | Trials | Intact `v1` or `v2` | Caught Error / Unopened | Silent Corruption | Follow-up Write Blocked |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for r in crash_res:
        if r.get("skipped"):
            lines.append(f"| `{r['format']}` | 0 | — | — | skipped (dependency absent) | — |")
        else:
            intact = r["intact_v1"] + r["intact_v2"]
            lines.append(
                f"| `{r['format']}` | {r['trials']} | {intact} (`v1`={r['intact_v1']}, `v2`={r['intact_v2']}) | "
                f"{r['caught_error']} | **{r['silent_corruption']}** | {r['followup_blocked']} |"
            )
    lines.append("")

    lines.append("## 3. GRIB1 / GRIB2 Message-Exact Roundtrip Verification (`wx1.adapters.grib_exact`)")
    lines.append("")
    if grib_skipped_reason:
        lines.append(f"- Skipped: {grib_skipped_reason}")
    else:
        lines.extend([
            "| File (basename) | Msgs | Editions | Packing | Grid | Source Size | WX-1 `1d` (vs Src) | WX-1 `2d` Lorenzo (vs Src) | `2d` Lattice / RawFallback | SHA-256 Byte-Identical (`2d`) | `float64` Bit-Exact (`2d`) | Max Abs Err |",
            "| :--- | ---: | :--- | :--- | :--- | ---: | ---: | ---: | :---: | ---: | ---: | ---: |",
        ])
        for g in grib_res:
            m1, m2 = g["1d"], g["2d"]
            lines.append(
                f"| `{g['name']}` | {g['messages']} | {g['editions']} | `{','.join(g['packings'])}` | `{','.join(g['grids'])}` | "
                f"{_fmt_mib(g['source_bytes'])} | {_fmt_mib(m1['file_bytes'])} ({_fmt_pct(m1['file_bytes'], g['source_bytes'])}) | "
                f"**{_fmt_mib(m2['file_bytes'])} ({_fmt_pct(m2['file_bytes'], g['source_bytes'])})** | "
                f"{m2['lattice']} / {m2['raw_fallback']} | {m2['byte_identical']} / {g['messages']} | "
                f"**{m2['f64_identical']} / {g['messages']}** | `{m2['max_abs_err']:.3g}` |"
            )
    lines.append("")

    lines.append("## 4. NetCDF-4 Roundtrip Verification")
    lines.append("")
    if nc_skipped_reason:
        lines.append(f"- Skipped: {nc_skipped_reason}")
    else:
        lines.extend([
            "| File (basename) | Variable | Shape | Dtype | NetCDF-4 Size | WX-1 Size (vs NC4) | Bit-Exact Roundtrip |",
            "| :--- | :--- | :--- | :--- | ---: | ---: | :---: |",
        ])
        for n in nc_res:
            lines.append(
                f"| `{n['name']}` | `{n['variable']}` | `{n['shape']}` | `{n['dtype']}` | "
                f"{_fmt_mib(n['nc_bytes'])} | {_fmt_mib(n['wx1_bytes'])} ({_fmt_pct(n['wx1_bytes'], n['nc_bytes'])}) | "
                f"`{n['exact_roundtrip']}` |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="WX-1 end-to-end validation runner for KNMI/ECMWF.")
    parser.add_argument("--grib", action="append", default=[], help="GRIB1/GRIB2 file to validate (repeatable).")
    parser.add_argument("--netcdf", action="append", default=[], help="NetCDF file to validate (repeatable).")
    parser.add_argument("--crash-trials", type=int, default=30, help="SIGKILL trials per format (default: 30).")
    parser.add_argument("--scratch-dir", default=None, help="Directory for crash/temp runs (test your target filesystem).")
    parser.add_argument("--skip-pytest", action="store_true", help="Skip running repository pytest suite.")
    parser.add_argument("--output", default="wx1_validation_report.md", help="Markdown report path.")
    args = parser.parse_args()

    work_parent = Path(args.scratch_dir) if args.scratch_dir else None
    if work_parent:
        work_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="wx1_val_", dir=str(work_parent) if work_parent else None) as tmp:
        work_dir = Path(tmp)

        pytest_res = {"ran": False, "reason": "--skip-pytest"} if args.skip_pytest else run_pytest_suite()
        reg_res = verify_v017_regressions(work_dir)
        crash_res = run_crash_consistency(work_dir, args.crash_trials)

        grib_res: list[dict[str, Any]] = []
        grib_skipped_reason: str | None = None
        try:
            import eccodes  # noqa: F401
        except ImportError:
            grib_skipped_reason = "eccodes not installed"
        else:
            grib_paths = [Path(p) for p in args.grib]
            if not grib_paths:
                grib_paths = [_make_synthetic_grib2(work_dir / "synthetic_pl_sample.grib2")]
            for gp in grib_paths:
                grib_res.append(validate_grib_file(gp, work_dir))

        nc_res: list[dict[str, Any]] = []
        nc_skipped_reason: str | None = None
        try:
            import netCDF4
        except ImportError:
            nc_skipped_reason = "netCDF4 not installed"
        else:
            nc_paths = [Path(p) for p in args.netcdf]
            if not nc_paths:
                syn_nc = work_dir / "synthetic_cf_sample.nc"
                with netCDF4.Dataset(syn_nc, "w", format="NETCDF4") as ds:
                    ds.createDimension("time", 4)
                    ds.createDimension("lat", 32)
                    ds.createDimension("lon", 64)
                    v = ds.createVariable("t2m", "f4", ("time", "lat", "lon"), zlib=True, complevel=4, fill_value=-9999.0)
                    v.units = "K"
                    arr = (275.0 + 10.0 * np.sin(np.linspace(0, 6.28, 4 * 32 * 64, dtype=np.float32))).reshape(4, 32, 64)
                    arr[0, 0, 0] = -9999.0
                    v[:] = arr
                nc_paths = [syn_nc]
            for np_path in nc_paths:
                nc_res.append(validate_netcdf_file(np_path, work_dir))

        report = render_report(pytest_res, reg_res, crash_res, grib_res, nc_res, grib_skipped_reason, nc_skipped_reason)
        out_path = Path(args.output)
        out_path.write_text(report, encoding="utf-8")
        print(report)

    ok_pytest = (not pytest_res["ran"]) or pytest_res["returncode"] == 0
    ok_reg = reg_res["stale_temp_same_pid_ok"] and not reg_res["from_netcdf_warning"].startswith("FAIL")
    wx1_crash = next((c for c in crash_res if c["format"] == "wx1"), None)
    ok_crash = wx1_crash is not None and wx1_crash["silent_corruption"] == 0 and wx1_crash["followup_blocked"] == 0
    ok_grib = all(g["2d"]["f64_identical"] == g["messages"] for g in grib_res)
    ok_nc = all(n["exact_roundtrip"] for n in nc_res)
    return 0 if (ok_pytest and ok_reg and ok_crash and ok_grib and ok_nc) else 1


if __name__ == "__main__":
    raise SystemExit(main())
