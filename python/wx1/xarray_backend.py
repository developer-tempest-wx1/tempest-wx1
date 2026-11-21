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

"""xarray backend with lazy per-selection reads from a WX-1 snapshot."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
from xarray.backends import BackendEntrypoint
from xarray.backends.common import BackendArray
from xarray.core.indexing import (
    IndexingSupport,
    LazilyIndexedArray,
    explicit_indexing_adapter,
)

from . import open_file


# ==============================================================================
# BLOCK 1: LAZY SUB-VOLUME SLICING ADAPTER (Wx1BackendArray)
# Purpose: Intercept xarray / Dask index keys (slices, steps, integers) and
#          map them to bounded WX-1 read_array_slice calls without loading
#          unselected chunks into memory.
# ==============================================================================

class Wx1BackendArray(BackendArray):
    """BackendArray that requests only the chunks intersecting each selection."""

    def __init__(
        self,
        file,
        variable,
        *,
        decode_dtype=None,
        cf_units: str | None = None,
        cf_calendar: str | None = None,
    ):
        self._file = file
        self._path = variable["path"]
        self.shape = tuple(int(value) for value in variable["shape"])
        self._raw_dtype = np.dtype(variable["numpy_dtype"])
        self._decode_dtype = np.dtype(decode_dtype) if decode_dtype is not None else None
        self._cf_units = cf_units
        self._cf_calendar = cf_calendar
        if self._decode_dtype is not None:
            self.dtype = self._decode_dtype
        elif self._cf_units is not None:
            probe = xr.coding.times.decode_cf_datetime(
                np.zeros((1,), dtype=self._raw_dtype),
                units=self._cf_units,
                calendar=self._cf_calendar or "standard",
            )
            self.dtype = np.asarray(probe).dtype
        else:
            self.dtype = self._raw_dtype

    def __getitem__(self, key):
        return explicit_indexing_adapter(
            key,
            self.shape,
            IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def _raw_indexing_method(self, key):
        starts = []
        counts = []
        post_index = []
        for axis, (selector, length) in enumerate(zip(key, self.shape)):
            if isinstance(selector, (int, np.integer)):
                index = int(selector)
                if index < 0:
                    index += length
                if index < 0 or index >= length:
                    raise IndexError(f"index {selector} is out of bounds for axis {axis} with size {length}")
                starts.append(index)
                counts.append(1)
                post_index.append(0)
                continue
            if not isinstance(selector, slice):
                raise IndexError("WX-1 backend supports basic integer and slice indexing")
            begin, end, step = selector.indices(length)
            if step > 0:
                low, high = begin, end
                local_start = 0
                local_stop = max(0, high - low)
                local_step = step
            else:
                if begin < end:
                    low, high = 0, 0
                    local_start, local_stop, local_step = 0, 0, 1
                else:
                    low, high = end + 1, begin + 1
                    local_start = max(0, high - low - 1)
                    local_stop = None
                    local_step = step
            starts.append(low)
            counts.append(max(0, high - low))
            post_index.append(slice(local_start, local_stop, local_step))

        array = self._file.read_array_slice(self._path, starts, counts)
        selected = array[tuple(post_index)]
        if self._decode_dtype is not None:
            if self._decode_dtype.kind in ("M", "m"):
                return np.asarray(selected, dtype=np.int64).view(self._decode_dtype)
            return np.asarray(selected, dtype=self._decode_dtype)
        if self._cf_units is not None:
            decoded = xr.coding.times.decode_cf_datetime(
                np.asarray(selected),
                units=self._cf_units,
                calendar=self._cf_calendar or "standard",
            )
            return np.asarray(decoded, dtype=self.dtype)
        return selected


# ==============================================================================
# BLOCK 2: XARRAY ENGINE ENTRYPOINT (engine="wx1")
# Purpose: Integrate WX-1 natively into xarray.open_dataset, populating dataset
#          coordinates, attributes, and variables with lazy Dask-compatible backend arrays.
# ==============================================================================

class Wx1BackendEntrypoint(BackendEntrypoint):
    """xarray entrypoint registered as ``engine='wx1'``."""

    description = "Experimental WX-1 scientific container backend"

    def open_dataset(
        self,
        filename_or_obj: str | Path,
        *,
        drop_variables: str | Iterable[str] | None = None,
        decode_times: bool = True,
        decode_timedelta: bool = True,
    ) -> xr.Dataset:
        if isinstance(drop_variables, str):
            drop_variables = {drop_variables}
        path = Path(filename_or_obj)
        file = open_file(path)
        variables = {}
        for item in file.variables:
            variable_path = item["path"]
            if drop_variables and variable_path in drop_variables:
                continue
            dtype = np.dtype(
                {
                    "uint8": "u1", "int8": "i1", "uint16": "u2", "int16": "i2",
                    "uint32": "u4", "int32": "i4", "uint64": "u8", "int64": "i8",
                    "float32": "f4", "float64": "f8", "bool": "b1",
                }[item["dtype"]]
            )
            if dtype.itemsize > 1:
                dtype = dtype.newbyteorder("<" if item["byte_order"] == "little" else ">")
            item_for_backend = dict(item)
            item_for_backend["numpy_dtype"] = dtype.str

            v_attrs = dict(item["attributes"])
            v_encoding = {}
            logical_dtype = str(v_attrs.get("wx1:logical_dtype", ""))
            decode_dtype = None
            cf_units = None
            cf_calendar = None

            if decode_times and item["dtype"] == "int64" and logical_dtype.startswith("datetime64["):
                decode_dtype = np.dtype(logical_dtype)
                for k in ("units", "calendar", "wx1:logical_dtype"):
                    if k in v_attrs:
                        v_encoding[k] = v_attrs.pop(k)
            elif decode_timedelta and item["dtype"] == "int64" and logical_dtype.startswith("timedelta64["):
                decode_dtype = np.dtype(logical_dtype)
                for k in ("units", "wx1:logical_dtype"):
                    if k in v_attrs:
                        v_encoding[k] = v_attrs.pop(k)
            elif decode_times and isinstance(v_attrs.get("units"), str) and " since " in v_attrs["units"]:
                cf_units = v_attrs.pop("units")
                cf_calendar = v_attrs.pop("calendar", "standard")
                v_encoding["units"] = cf_units
                v_encoding["calendar"] = cf_calendar

            backend_array = Wx1BackendArray(
                file,
                item_for_backend,
                decode_dtype=decode_dtype,
                cf_units=cf_units,
                cf_calendar=cf_calendar,
            )
            lazy_data = LazilyIndexedArray(backend_array)
            var_obj = xr.Variable(
                tuple(item["dimensions"]),
                lazy_data,
                attrs=v_attrs,
                encoding=v_encoding,
            )
            clean_name = item.get("name") or variable_path.lstrip("/")
            if drop_variables and clean_name in drop_variables:
                continue
            variables[clean_name or variable_path] = var_obj

        if hasattr(file, "_native") and hasattr(file._native, "attributes"):
            raw_global = {
                str(k): str(v)
                for k, v in dict(file._native.attributes()).items()
                if not str(k).startswith("wx1:__")
            }
        else:
            raw_global = dict(file.attributes)

        dataset = xr.Dataset(
            data_vars=variables,
            attrs={"wx1_dataset_id": file.dataset_id, "wx1_generation": file.generation, **raw_global},
        )
        raw_coord_vars = tuple(file.coordinate_variables)
        if raw_coord_vars:
            coord_names_set = {p.lstrip("/") for p in raw_coord_vars}
            coordinate_paths = [
                name for name in dataset.variables
                if name in coord_names_set or name in raw_coord_vars
            ]
        else:
            # Backwards-compatible heuristic for legacy v0.1.7 containers without Coordinate records
            coordinate_paths = [
                name for name, var in dataset.variables.items()
                if var.ndim == 1 and var.dims[0] == name
            ]
        if coordinate_paths:
            dataset = dataset.set_coords(coordinate_paths)
        dataset.set_close(file.close)
        return dataset

    def guess_can_open(self, filename_or_obj):
        return str(filename_or_obj).lower().endswith(".wx1")
