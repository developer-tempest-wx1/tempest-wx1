# Copyright 2026 Jordy van Gils
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import wx1


pytestmark = pytest.mark.skipif(wx1._core is None, reason="WX-1 PyO3 extension has not been built")


def test_native_numpy_roundtrip_chunked_f32_is_bit_exact(tmp_path):
    source = (np.arange(24, dtype="<f4").reshape(4, 6) / np.float32(7.0)).astype("<f4")
    path = tmp_path / "numpy-roundtrip.wx1"

    wx1.write_array(
        path,
        source,
        dataset_id="numpy-roundtrip",
        variable_path="/temperature",
        dimensions=["y", "x"],
        chunk_shape=[2, 3],
        dimension_roles=["latitude", "longitude"],
        attributes={"units": "K", "standard_name": "air_temperature"},
    )

    with wx1.open_file(path) as file:
        assert file.dataset_id == "numpy-roundtrip"
        restored = file.read_array("/temperature")
        np.testing.assert_array_equal(restored.view("u4"), source.view("u4"))
        assert restored.dtype == np.dtype("<f4")
        assert file.variables[0]["attributes"]["units"] == "K"

        selection = file.read_array_slice("/temperature", [1, 2], [2, 3])
        np.testing.assert_array_equal(selection, source[1:3, 2:5])


def test_native_numpy_roundtrip_preserves_big_endian_payload(tmp_path):
    source = np.array([[1.25, -2.5], [3.75, 4.5]], dtype=">f4")
    path = tmp_path / "big-endian.wx1"
    wx1.write_array(
        path,
        source,
        dataset_id="big-endian",
        variable_path="/field",
        dimensions=["y", "x"],
        chunk_shape=[1, 2],
    )
    with wx1.open_file(path) as file:
        restored = file.read_array("/field")
        assert restored.dtype == np.dtype(">f4")
        np.testing.assert_array_equal(restored.view(">u4"), source.view(">u4"))


def test_native_xarray_dask_chunked_read_and_reduction(tmp_path):
    da = pytest.importorskip("dask.array")
    source = np.arange(96, dtype="<f4").reshape(4, 6, 4) / np.float32(11.0)
    path = tmp_path / "native-dask.wx1"
    wx1.write_array(
        path,
        source,
        dataset_id="native-dask",
        variable_path="/temperature",
        dimensions=["time", "y", "x"],
        chunk_shape=[1, 3, 2],
        dimension_roles=["time", "latitude", "longitude"],
    )

    dataset = wx1.open_dataset(path, chunks={"time": 1, "y": 3, "x": 2})
    try:
        assert isinstance(dataset["temperature"].data, da.Array)
        actual = dataset["temperature"].mean().compute().item()
        assert actual == pytest.approx(float(source.mean()))
    finally:
        dataset.close()


@pytest.mark.parametrize(
    "source",
    [
        np.array([[0, 1, 65535], [12, 42, 1024]], dtype="<u2"),
        np.array([[1.0, -2.0], [3.5, 4.25]], dtype="<f8"),
        np.array([[True, False], [False, True]], dtype=np.bool_),
    ],
    ids=["uint16", "float64", "bool8"],
)
def test_native_numpy_roundtrip_preserves_supported_dtype_and_values(tmp_path, source):
    path = tmp_path / f"dtype-{source.dtype.name}.wx1"
    wx1.write_array(
        path,
        source,
        dataset_id=f"dtype-{source.dtype.name}",
        variable_path="/field",
        dimensions=["y", "x"],
        chunk_shape=[1, 2],
    )
    with wx1.open_file(path) as file:
        restored = file.read_array("/field")
        assert restored.dtype == source.dtype
        np.testing.assert_array_equal(restored, source)
        if source.dtype.kind in ("u", "i", "f"):
            unsigned = f"u{source.dtype.itemsize}"
            np.testing.assert_array_equal(restored.view(unsigned), source.view(unsigned))


def test_lazy_open_and_explicit_validate(tmp_path):
    source = np.arange(24, dtype="<f4").reshape(4, 6)
    path = tmp_path / "lazy-open.wx1"
    wx1.write_array(
        path,
        source,
        dataset_id="lazy-open-test",
        variable_path="/temperature",
        dimensions=["y", "x"],
        chunk_shape=[2, 3],
    )

    with wx1.open_file(path) as file:
        assert file.dataset_id == "lazy-open-test"
        # Explicit full validation passes on intact file
        file.validate()
        restored = file.read_array("/temperature")
        np.testing.assert_array_equal(restored, source)


def test_adr0024_restart_profile_forbids_lossy_and_preserves_bit_exact_with_byteshuffle(tmp_path):
    source = np.random.RandomState(42).normal(280.0, 10.0, (32, 32)).astype("<f4")
    bad_path = tmp_path / "bad-restart.wx1"
    with pytest.raises(ValueError, match="lossy quantization is strictly forbidden in restart profile"):
        wx1.write_array(
            bad_path,
            source,
            dataset_id="bad-restart",
            variable_path="/t2m",
            dimensions=["y", "x"],
            profile="restart",
            lossy_error_bound=0.01,
        )

    good_path = tmp_path / "good-restart-shuffle.wx1"
    wx1.write_array(
        good_path,
        source,
        dataset_id="good-restart",
        variable_path="/t2m",
        dimensions=["y", "x"],
        chunk_shape=[16, 16],
        profile="restart",
        codec="zstd_shuffle",
    )
    with wx1.open_file(good_path) as file:
        file.validate()
        restored = file.read_array("/t2m")
        np.testing.assert_array_equal(restored.view("u4"), source.view("u4"))
        attrs = file.variables[0]["attributes"]
        assert attrs["wx1:lossy"] == "false"
        assert attrs["wx1:profile"] == "restart"


def test_adr0024_output_profile_lossy_quantization_and_special_floats(tmp_path):
    rng = np.random.RandomState(123)
    source = rng.uniform(250.0, 310.0, (32, 32)).astype("<f4")
    # Inject special IEEE 754 values
    source[0, 0] = np.nan
    source[0, 1] = np.inf
    source[0, 2] = -np.inf
    source[0, 3] = np.float32(-0.0)
    source[0, 4] = np.float32(0.0)

    lossy_path = tmp_path / "lossy-output.wx1"
    eps = 0.01
    wx1.write_array(
        lossy_path,
        source,
        dataset_id="lossy-output",
        variable_path="/t2m",
        dimensions=["y", "x"],
        chunk_shape=[16, 16],
        profile="output",
        lossy_error_bound=eps,
        lossy_metric="max_abs_error",
        codec="zstd_shuffle",
    )

    with wx1.open_file(lossy_path) as file:
        file.validate()
        restored = file.read_array("/t2m")
        attrs = file.variables[0]["attributes"]
        assert attrs["wx1:lossy"] == "true"
        assert attrs["wx1:profile"] == "output"
        assert attrs["wx1:quantization_type"] == "linear_absolute"
        assert float(attrs["wx1:error_bound"]) == pytest.approx(eps)
        assert float(attrs["wx1:max_measured_abs_error"]) <= eps

        # Verify special floats
        assert np.isnan(restored[0, 0])
        assert np.isposinf(restored[0, 1])
        assert np.isneginf(restored[0, 2])
        assert restored[0, 3].view("u4") == np.float32(-0.0).view("u4")
        assert restored[0, 4].view("u4") == np.float32(0.0).view("u4")

        # Verify finite elements strictly obey declared error bound
        finite_mask = np.isfinite(source)
        diffs = np.abs(restored[finite_mask] - source[finite_mask])
        assert np.max(diffs) <= eps


def test_adr0025_parallel_workers_and_async_writer(tmp_path):
    rng = np.random.RandomState(2025)
    source = rng.normal(285.0, 5.0, (4, 32, 32)).astype("<f4")
    p1 = tmp_path / "w1.wx1"
    p4 = tmp_path / "w4.wx1"

    wx1.write_array(
        p1,
        source,
        dataset_id="parallel-test",
        variable_path="/t2m",
        dimensions=["z", "y", "x"],
        chunk_shape=[1, 16, 16],
        codec="zstd_shuffle",
        workers=1,
    )
    wx1.write_array(
        p4,
        source,
        dataset_id="parallel-test",
        variable_path="/t2m",
        dimensions=["z", "y", "x"],
        chunk_shape=[1, 16, 16],
        codec="zstd_shuffle",
        workers=4,
    )
    assert p1.read_bytes() == p4.read_bytes()

    async_path = tmp_path / "async_nwp.wx1"
    with wx1.AsyncWriter(
        async_path,
        dataset_id="nwp-async",
        variable_path="/t2m",
        spatial_dimensions=["y", "x"],
        spatial_shape=[32, 32],
        spatial_chunk_shape=[16, 16],
        codec="zstd_shuffle",
        workers=4,
        max_pending_generations=2,
        durability="buffered",
    ) as writer:
        for step in range(4):
            writer.submit_step(source[step])
        tel = writer.flush()
        assert tel["submitted_generations"] == 4
        assert tel["committed_generations"] == 4

    with wx1.open_file(async_path) as f:
        assert f.generation == 4
        f.validate()
        restored = f.read_array("/t2m")
        np.testing.assert_array_equal(restored.view("u4"), source.view("u4"))


def test_adr0026_python_cubed_sphere_halos_and_tile_read(tmp_path):
    # C16 cubed-sphere with halo_width=2 -> allocated shape (6, 20, 20)
    rng = np.random.RandomState(2026)
    source = rng.normal(290.0, 4.0, (6, 20, 20)).astype("<f4")
    cs_path = tmp_path / "cs_c16.wx1"

    wx1.write_array(
        cs_path,
        source,
        dataset_id="cs-c16",
        variable_path="/t",
        dimensions=["tile", "y", "x"],
        chunk_shape=[1, 20, 20],
        codec="zstd_shuffle",
        workers=4,
        grid_family="cubed_sphere",
        c_resolution=16,
        halo_width=2,
        stagger_horizontal="center",
        tile_dimension="tile",
    )

    with wx1.open_file(cs_path) as f:
        f.validate()
        topos = f.grid_topologies
        assert "primary_grid" in topos
        assert topos["primary_grid"]["family"] == "cubed_sphere"
        assert topos["primary_grid"]["c_resolution"] == 16
        assert topos["primary_grid"]["num_tiles"] == 6

        plan2 = f.plan_tile_slice("/t", 2, include_halos=False)
        assert plan2["chunk_ids"] == (3,)
        assert plan2["shape"] == (1, 16, 16)

        tile2_int = f.read_tile_slice("/t", 2, include_halos=False)
        assert tile2_int.shape == (1, 16, 16)
        np.testing.assert_array_equal(tile2_int[0], source[2, 2:18, 2:18])

        tile2_full = f.read_tile_slice("/t", 2, include_halos=True)
        assert tile2_full.shape == (1, 20, 20)
        np.testing.assert_array_equal(tile2_full[0], source[2])


def test_write_dataset_no_variable_attribute_leak_to_global_attrs(tmp_path):
    xr = pytest.importorskip("xarray")
    path = tmp_path / "no_attr_leak.wx1"
    data = np.linspace(270.0, 290.0, 6, dtype=np.float32).reshape(2, 3)
    ds = xr.Dataset(
        data_vars={
            "t2m": (("y", "x"), data, {"units": "K", "long_name": "2m temperature"}),
            "msl": (("y", "x"), data * 350.0, {"units": "Pa", "long_name": "Mean sea level pressure"}),
        },
        coords={
            "y": ("y", np.array([50.0, 52.0], dtype=np.float32), {"units": "degrees_north"}),
            "x": ("x", np.array([4.0, 5.0, 6.0], dtype=np.float32), {"units": "degrees_east"}),
        },
        attrs={"title": "Clean Global Metadata"},
    )
    wx1.write_dataset(path, ds, dataset_id="no-leak-test", codec="zstd_shuffle", workers=1)

    with wx1.open_file(path) as f:
        assert f.attributes["title"] == "Clean Global Metadata"
        assert "units" not in f.attributes
        assert "long_name" not in f.attributes

    with wx1.open_dataset(path) as reopened:
        assert reopened.attrs["title"] == "Clean Global Metadata"
        assert "units" not in reopened.attrs
        assert "long_name" not in reopened.attrs
        assert reopened["t2m"].attrs["units"] == "K"
        assert reopened["msl"].attrs["units"] == "Pa"

    # Also verify no leak even when global attributes are empty
    path_empty = tmp_path / "no_attr_leak_empty.wx1"
    ds_no_global = ds.copy()
    ds_no_global.attrs = {}
    wx1.write_dataset(path_empty, ds_no_global, dataset_id="no-leak-empty", workers=1)
    with wx1.open_dataset(path_empty) as reopened_empty:
        assert "units" not in reopened_empty.attrs
        assert "long_name" not in reopened_empty.attrs


def test_write_dataset_preserves_dimension_and_nondim_coordinates(tmp_path):
    xr = pytest.importorskip("xarray")
    path = tmp_path / "coords_roundtrip.wx1"
    ny, nx = 3, 4
    y_m = np.array([0.0, 1500.0, 3000.0], dtype=np.float64)
    x_m = np.array([0.0, 1500.0, 3000.0, 4500.0], dtype=np.float64)
    lat2d = np.linspace(51.0, 51.03, ny * nx, dtype=np.float64).reshape(ny, nx)
    lon2d = np.linspace(5.0, 5.06, ny * nx, dtype=np.float64).reshape(ny, nx)
    lead_hours = np.array([0, 3], dtype=np.int32)
    t_vals = np.array(["2026-11-21T00:00:00", "2026-11-21T03:00:00"], dtype="datetime64[s]")
    field = np.arange(2 * ny * nx, dtype=np.float32).reshape(2, ny, nx)

    ds = xr.Dataset(
        data_vars={"t2m": (("time", "y", "x"), field, {"units": "K"})},
        coords={
            "time": ("time", t_vals),
            "y": ("y", y_m, {"units": "m"}),
            "x": ("x", x_m, {"units": "m"}),
            "lead_hours": ("time", lead_hours, {"units": "hour"}),
            "lat2d": (("y", "x"), lat2d, {"units": "degrees_north", "standard_name": "latitude"}),
            "lon2d": (("y", "x"), lon2d, {"units": "degrees_east", "standard_name": "longitude"}),
        },
    )
    wx1.write_dataset(path, ds, dataset_id="coords-rt", codec="zstd_shuffle", workers=1)

    with wx1.open_file(path) as f:
        coord_vars = set(f.coordinate_variables)
        assert {"/time", "/y", "/x", "/lead_hours", "/lat2d", "/lon2d"}.issubset(coord_vars)
        coord_meta = {c["name"]: c for c in f.coordinates}
        assert coord_meta["time"]["axis"] == "time"
        assert coord_meta["lat2d"]["axis"] == "latitude"
        assert coord_meta["lon2d"]["axis"] == "longitude"
        assert coord_meta["lon2d"]["longitude_convention"] == "minus180_to_180"

    with wx1.open_dataset(path) as reopened:
        for cname in ("time", "y", "x", "lead_hours", "lat2d", "lon2d"):
            assert cname in reopened.coords
        assert "t2m" in reopened.data_vars
        np.testing.assert_array_equal(reopened.coords["lead_hours"].values, lead_hours)
        np.testing.assert_allclose(reopened.coords["lat2d"].values, lat2d)
        np.testing.assert_array_equal(reopened["t2m"].values, field)


def test_cf_datetime64_and_timedelta64_roundtrip_and_raw_bit_exact(tmp_path):
    xr = pytest.importorskip("xarray")
    path = tmp_path / "temporal_cf.wx1"
    times_s = np.array(
        ["2026-11-21T00:00:00", "2026-11-21T06:00:00", "2026-11-21T12:00:00"],
        dtype="datetime64[s]",
    )
    times_ns = np.array(
        ["2026-11-21T00:00:00.123456789", "2026-11-21T01:00:00.000000000"],
        dtype="datetime64[ns]",
    )
    steps_h = np.array([0, 6, 12], dtype="timedelta64[h]")
    t2m = np.linspace(275.0, 285.0, 6, dtype=np.float32).reshape(3, 2)

    ds = xr.Dataset(
        data_vars={"t2m": (("time", "member"), t2m, {"units": "K"})},
        coords={
            "time": ("time", times_s),
            "step": ("time", steps_h),
            "precise_time": ("member", times_ns),
        },
    )
    wx1.write_dataset(path, ds, dataset_id="temporal-cf", workers=1)

    # 1. Native reader sees bit-exact int64 (note xarray normalizes some units on Dataset creation)
    expected_step_raw = ds["step"].values.view("int64")
    expected_step_dtype = ds["step"].values.dtype
    with wx1.open_file(path) as f:
        raw_t = f.read_array("/time")
        assert raw_t.dtype == np.dtype("int64")
        np.testing.assert_array_equal(raw_t, times_s.view("int64"))
        raw_step = f.read_array("/step")
        assert raw_step.dtype == np.dtype("int64")
        np.testing.assert_array_equal(raw_step, expected_step_raw)

    # 2. xarray default (decode_times=True, decode_timedelta=True)
    with wx1.open_dataset(path) as decoded:
        assert decoded["time"].dtype == np.dtype("datetime64[s]")
        np.testing.assert_array_equal(decoded["time"].values, times_s)
        assert decoded["precise_time"].dtype == np.dtype("datetime64[ns]")
        np.testing.assert_array_equal(decoded["precise_time"].values, times_ns)
        assert decoded["step"].dtype == expected_step_dtype
        np.testing.assert_array_equal(decoded["step"].values, ds["step"].values)

    # 3. xarray raw mode (decode_times=False, decode_timedelta=False)
    with wx1.open_dataset(path, decode_times=False, decode_timedelta=False) as raw_ds:
        assert raw_ds["time"].dtype == np.dtype("int64")
        assert raw_ds["time"].attrs["units"] == "seconds since 1970-01-01 00:00:00"
        assert raw_ds["time"].attrs["calendar"] == "proleptic_gregorian"
        np.testing.assert_array_equal(raw_ds["time"].values, times_s.view("int64"))
        assert raw_ds["step"].dtype == np.dtype("int64")
        np.testing.assert_array_equal(raw_ds["step"].values, expected_step_raw)

    # 4. Direct dict write preserves raw timedelta64[h] unit ([0, 6, 12]) without xarray normalization
    path_hours = tmp_path / "temporal_hours_dict.wx1"
    wx1.write_dataset(
        path_hours,
        {"step": steps_h, "val": np.ones(3, dtype=np.float32)},
        dataset_id="hours-dict",
        dimensions={"step": ("step",), "val": ("step",)},
        coordinates=["step"],
        workers=1,
    )
    with wx1.open_file(path_hours) as fh:
        raw_h = fh.read_array("/step")
        assert raw_h.dtype == np.dtype("int64")
        np.testing.assert_array_equal(raw_h, steps_h.view("int64"))
    with wx1.open_dataset(path_hours) as ds_h:
        assert ds_h["step"].dtype == np.dtype("timedelta64[h]")
        np.testing.assert_array_equal(ds_h["step"].values, steps_h)


def test_open_mfdataset_across_multiple_wx1_files_with_dask(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("dask.array")
    files = []
    lat = np.array([50.0, 52.0], dtype=np.float32)
    lon = np.array([4.0, 6.0], dtype=np.float32)
    for idx in range(3):
        p = tmp_path / f"lead_{idx:03d}.wx1"
        t_arr = np.array([f"2026-11-21T{idx:02d}:00:00"], dtype="datetime64[s]")
        field = np.full((1, 2, 2), 280.0 + idx, dtype=np.float32)
        ds = xr.Dataset(
            data_vars={"t2m": (("time", "lat", "lon"), field, {"units": "K"})},
            coords={"time": ("time", t_arr), "lat": ("lat", lat), "lon": ("lon", lon)},
        )
        wx1.write_dataset(p, ds, dataset_id=f"mf-{idx}", workers=1)
        files.append(p)

    combined = xr.open_mfdataset(
        files,
        engine="wx1",
        combine="nested",
        concat_dim="time",
        chunks={"time": 1},
    )
    try:
        assert combined["t2m"].shape == (3, 2, 2)
        assert "time" in combined.coords
        assert combined["time"].dtype == np.dtype("datetime64[s]")
        np.testing.assert_allclose(
            combined["t2m"].mean(dim=("lat", "lon")).compute().values,
            np.array([280.0, 281.0, 282.0], dtype=np.float32),
        )
    finally:
        combined.close()


