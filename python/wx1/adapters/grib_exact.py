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

"""Message-exact GRIB archiving on top of the WX-1 integer lattice (``grib_2d`` / ``grib_delta``).

Archive managers keep *messages*, not just values.  This module stores, per GRIB message,

* the exact quantisation integers ``X`` (the values that GRIB packing stored) compressed with the
  2-D Lorenzo predictor + ZigZag + zstd/shuffle, and
* every byte of the message that is *not* the packed data section (all GRIB1 sections 0-4 header
  bytes / all GRIB2 sections 0-6, i.e. every MARS key, local definition, grid and bitmap
  definition) verbatim, de-duplicated across messages and zlib-compressed,

and rebuilds the message on export:

* **GRIB1 ``grid_simple``** (e.g. ERA5 from the CDS): the packed bit stream is regenerated from ``X``;
  the rebuilt message is **byte-identical** to the original (verified with SHA-256).
* **GRIB2** (``grid_ccsds`` / AEC, complex packing, ...): the stored sections 0-6 are used as an ecCodes
  template and the data section is re-encoded by ecCodes/libaec with the recorded packing keys.  With
  ECMWF open-data messages the result is byte-identical; whenever the re-encode would not reproduce
  the *values* bit-exactly, the original message is stored verbatim instead (``kind == "raw"``), so
  the round trip is always lossless at message level.  ``verify_roundtrip`` reports which route
  each message used and whether the bytes are identical.

Only the public functions ``write_grib_messages``, ``read_grib_messages`` and ``verify_roundtrip`` are
intended for users.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_G2_PACK_KEYS = (
    "decimalScaleFactor",
    "bitsPerValue",
    "orderOfSpatialDifferencing",
    "ccsdsFlags",
    "ccsdsBlockSize",
    "ccsdsRsi",
)


# ----------------------------------------------------------------------------- predictor primitives
def _zigzag16(d: np.ndarray) -> np.ndarray:
    i = d.view(np.int16).astype(np.int32)
    return ((i << 1) ^ (i >> 15)).astype(np.uint16)


def _unzigzag16(zz: np.ndarray) -> np.ndarray:
    return (zz >> np.uint16(1)) ^ (np.uint16(0) - (zz & np.uint16(1)))


def predict_forward(X: np.ndarray, mode: str = "2d") -> np.ndarray:
    """Lattice integers (``uint16``, shape ``[..., H, W]``) -> ZigZag residuals (same shape/dtype)."""
    X = np.ascontiguousarray(X, dtype=np.uint16)
    if mode == "int":
        return X
    d1 = np.empty_like(X)
    d1[..., 0] = X[..., 0]
    d1[..., 1:] = X[..., 1:] - X[..., :-1]
    if mode == "1d":
        zz = _zigzag16(d1)
        zz[..., 0] = X[..., 0]
        return zz
    if mode != "2d":
        raise ValueError(f"unknown predictor mode {mode!r}")
    d2 = np.empty_like(d1)
    d2[..., 0, :] = d1[..., 0, :]
    d2[..., 1:, :] = d1[..., 1:, :] - d1[..., :-1, :]
    zz = _zigzag16(d2)
    zz[..., 0, 0] = X[..., 0, 0]
    return zz


def predict_inverse(zz: np.ndarray, mode: str = "2d") -> np.ndarray:
    zz = np.ascontiguousarray(zz, dtype=np.uint16)
    if mode == "int":
        return zz
    d = _unzigzag16(zz)
    if mode == "1d":
        d[..., 0] = zz[..., 0]
        return np.cumsum(d, axis=-1, dtype=np.uint16)
    d[..., 0, 0] = zz[..., 0, 0]
    np.cumsum(d, axis=-2, dtype=np.uint16, out=d)
    np.cumsum(d, axis=-1, dtype=np.uint16, out=d)
    return d


# ----------------------------------------------------------------------------- bit packing (GRIB1 simple)
def _unpack_bits(buf: bytes, n: int, bpv: int) -> np.ndarray:
    if bpv == 0:
        return np.zeros(n, dtype=np.uint32)
    if bpv == 8:
        return np.frombuffer(buf, np.uint8, n).astype(np.uint32)
    if bpv == 16:
        return np.frombuffer(buf, ">u2", n).astype(np.uint32)
    nbytes = (n * bpv + 7) // 8
    bits = np.unpackbits(np.frombuffer(buf, np.uint8, nbytes))[: n * bpv].reshape(n, bpv)
    weights = (np.uint64(1) << np.arange(bpv - 1, -1, -1, dtype=np.uint64))
    return (bits.astype(np.uint64) * weights).sum(axis=1).astype(np.uint32)


def _pack_bits(X: np.ndarray, bpv: int) -> bytes:
    if bpv == 0:
        return b""
    if bpv == 8:
        return X.astype(np.uint8).tobytes()
    if bpv == 16:
        return X.astype(">u2").tobytes()
    shifts = np.arange(bpv - 1, -1, -1, dtype=np.uint64)
    bits = ((X.astype(np.uint64)[:, None] >> shifts) & np.uint64(1)).astype(np.uint8).ravel()
    return np.packbits(bits).tobytes()


def _eccodes():
    try:
        import eccodes

        return eccodes
    except ImportError as exc:  # pragma: no cover
        raise ImportError("grib_exact requires the 'grib2' extra (eccodes)") from exc


def _lattice_values(X: np.ndarray, R: float, E: int, D: int) -> np.ndarray:
    """Exactly the formula ecCodes uses to decode simple/CCSDS/complex packing (float64)."""
    y = R + X.astype(np.float64) * (2.0 ** E)
    return y * (10.0 ** (-D)) if D else y


# ----------------------------------------------------------------------------- extraction / rebuild
def _strip_g2(raw: bytes, off7: int) -> bytes:
    head = bytearray(raw[:off7])
    out = bytearray(bytes(head) + struct.pack(">IB", 5, 7) + b"7777")
    out[8:16] = struct.pack(">Q", len(out))
    return bytes(out)


def extract_field(raw: bytes, verify: bool = True) -> dict[str, Any]:
    """Split one GRIB message into ``{header, suffix, lattice integers, packing keys}``.

    Returns ``kind == "raw"`` (message kept verbatim) for anything that is not exactly representable.
    """
    ec = _eccodes()
    gid = ec.codes_new_from_message(raw)
    try:
        edition = int(ec.codes_get(gid, "edition"))
        ni = int(ec.codes_get(gid, "Ni")) if ec.codes_is_defined(gid, "Ni") else 0
        nj = int(ec.codes_get(gid, "Nj")) if ec.codes_is_defined(gid, "Nj") else 0
        packing = str(ec.codes_get(gid, "packingType"))
        bpv = int(ec.codes_get(gid, "bitsPerValue"))
        has_bitmap = bool(ec.codes_is_defined(gid, "bitmapPresent") and ec.codes_get(gid, "bitmapPresent"))
        rec: dict[str, Any] = {"kind": "raw", "edition": edition, "raw": raw, "ni": ni, "nj": nj}
        if ni <= 0 or nj <= 0 or bpv > 16 or packing in ("spectral_complex", "spectral_simple"):
            return rec
        R = float(ec.codes_get_double(gid, "referenceValue"))
        E = int(ec.codes_get(gid, "binaryScaleFactor"))
        D = int(ec.codes_get(gid, "decimalScaleFactor"))
        mask = None
        if has_bitmap:
            ec.codes_set_double(gid, "missingValue", float("nan"))
        vals = ec.codes_get_values(gid).reshape(nj, ni)
        if has_bitmap:
            mask = ~np.isnan(vals)
        if edition == 1:
            if packing != "grid_simple":
                return rec
            n = int(ec.codes_get(gid, "numberOfCodedValues"))
            off = int(ec.codes_get(gid, "offsetBeforeData"))
            Xp = _unpack_bits(raw[off:], n, bpv)
            nbytes = (n * bpv + 7) // 8
            header, suffix = raw[:off], raw[off + nbytes:]
            if mask is None:
                X = Xp.reshape(nj, ni)
            else:
                X = np.zeros((nj, ni), dtype=np.uint32)
                X[mask] = Xp
            keys: dict[str, Any] = {}
        else:
            if has_bitmap:
                return rec
            off7 = int(ec.codes_get(gid, "offsetSection7"))
            header, suffix = raw[:off7], b"7777"
            scale = 2.0 ** E
            X = np.rint((vals * (10.0 ** D) - R) / scale)
            if X.min() < 0 or X.max() > 65535:
                return rec
            X = X.astype(np.uint32)
            keys = {k: int(ec.codes_get(gid, k)) for k in _G2_PACK_KEYS if ec.codes_is_defined(gid, k)}
        rec.update(
            kind="lattice", header=header, suffix=suffix, bpv=bpv, R=R, E=E, D=D,
            mask=mask, X=X.astype(np.uint16), keys=keys, packing=packing,
        )
        if verify:
            if mask is None:
                ok = np.array_equal(_lattice_values(rec["X"], R, E, D), vals)
            else:
                ok = np.array_equal(_lattice_values(rec["X"][mask], R, E, D), vals[mask])
            if not ok:
                return {"kind": "raw", "edition": edition, "raw": raw, "ni": ni, "nj": nj}
            if edition == 2:
                rebuilt = rebuild_message(rec, rec["X"])
                g2 = ec.codes_new_from_message(rebuilt)
                try:
                    again = ec.codes_get_values(g2)
                finally:
                    ec.codes_release(g2)
                if not np.array_equal(again.reshape(nj, ni), vals):
                    return {"kind": "raw", "edition": edition, "raw": raw, "ni": ni, "nj": nj}
        return rec
    finally:
        ec.codes_release(gid)


def rebuild_message(rec: dict[str, Any], X: np.ndarray) -> bytes:
    """Rebuild a GRIB message from stored header/suffix/keys and the lattice integers ``X``."""
    if rec["kind"] == "raw":
        return rec["raw"]
    R, E, D, bpv = rec["R"], rec["E"], rec["D"], rec["bpv"]
    mask = rec.get("mask")
    if rec["edition"] == 1:
        Xp = X[mask] if mask is not None else X.ravel()
        return rec["header"] + _pack_bits(Xp.astype(np.uint32), bpv) + rec["suffix"]
    ec = _eccodes()
    gid = ec.codes_new_from_message(_strip_g2(rec["header"], len(rec["header"])))
    try:
        for k in ("decimalScaleFactor", "bitsPerValue", "orderOfSpatialDifferencing",
                  "ccsdsFlags", "ccsdsBlockSize", "ccsdsRsi"):
            if k in rec["keys"]:
                try:
                    ec.codes_set(gid, k, int(rec["keys"][k]))
                except Exception:
                    pass
        ec.codes_set_values(gid, _lattice_values(X, R, E, D).ravel())
        return ec.codes_get_message(gid)
    finally:
        ec.codes_release(gid)


# ----------------------------------------------------------------------------- container I/O
def _b64z(obj: bytes) -> str:
    return base64.b64encode(zlib.compress(obj, 9)).decode("ascii")


def _unb64z(s: str) -> bytes:
    return zlib.decompress(base64.b64decode(s.encode("ascii")))


def write_grib_messages(path: str | Path, messages: Sequence[bytes], *, mode: str = "2d",
                        workers: int = 4, dataset_id: str = "grib_messages") -> dict[str, Any]:
    """Store GRIB messages (all on the same grid) in a WX-1 file; returns a small statistics dict."""
    import wx1

    recs = [extract_field(m) for m in messages]
    shapes = {(r["nj"], r["ni"]) for r in recs if r["kind"] == "lattice"}
    if len(shapes) > 1:
        raise ValueError(f"all lattice messages must share one grid, got {sorted(shapes)}")
    nj, ni = next(iter(shapes)) if shapes else (1, 1)
    n = len(recs)
    X = np.zeros((n, nj, ni), dtype=np.uint16)
    pool: dict[bytes, int] = {}
    blobs: list[bytes] = []

    def _intern(b: bytes) -> int:
        if b not in pool:
            pool[b] = len(blobs)
            blobs.append(b)
        return pool[b]

    meta = []
    for i, r in enumerate(recs):
        if r["kind"] == "raw":
            meta.append({"kind": "raw", "raw": _intern(r["raw"])})
            continue
        X[i] = r["X"]
        m = {"kind": "lattice", "edition": r["edition"], "header": _intern(r["header"]),
             "suffix": _intern(r["suffix"]), "bpv": r["bpv"], "R": r["R"], "E": r["E"], "D": r["D"],
             "keys": r["keys"]}
        if r["mask"] is not None:
            m["mask"] = _intern(np.packbits(r["mask"].ravel()).tobytes())
        meta.append(m)
    zz = predict_forward(X, mode)
    payload = zz.astype(np.uint8) if int(zz.max()) < 256 else zz
    blob = {"mode": mode, "shape": [nj, ni], "fields": meta,
            "blobs": [base64.b64encode(b).decode("ascii") for b in blobs]}
    wx1.write_array(
        str(path), payload, dataset_id=dataset_id, variable_path="/fields",
        dimensions=["field", "latitude", "longitude"], chunk_shape=[1, nj, ni],
        codec="zstd_shuffle", workers=workers, durability="buffered",
        attributes={"grib:exact_records": _b64z(json.dumps(blob).encode()), "grib:exact_mode": mode},
    )
    return {"fields": n, "lattice": sum(r["kind"] == "lattice" for r in recs),
            "raw_fallback": sum(r["kind"] == "raw" for r in recs),
            "file_bytes": Path(path).stat().st_size, "source_bytes": sum(len(m) for m in messages)}


def read_grib_messages(path: str | Path) -> list[bytes]:
    """Inverse of :func:`write_grib_messages`; returns the GRIB messages in their original order."""
    import wx1

    with wx1.open_file(str(path)) as f:
        v = next(v for v in f.variables if v["path"] == "/fields")
        blob = json.loads(_unb64z(v["attributes"]["grib:exact_records"]))
        zz = f.read_array("/fields", copy=False).astype(np.uint16, copy=False)
    blobs = [base64.b64decode(b) for b in blob["blobs"]]
    nj, ni = blob["shape"]
    X = predict_inverse(zz.copy(), blob["mode"])
    out = []
    for i, m in enumerate(blob["fields"]):
        if m["kind"] == "raw":
            out.append(blobs[m["raw"]])
            continue
        mask = None
        if "mask" in m:
            mask = np.unpackbits(np.frombuffer(blobs[m["mask"]], np.uint8))[: nj * ni].astype(bool).reshape(nj, ni)
        rec = {"kind": "lattice", "edition": m["edition"], "header": blobs[m["header"]],
               "suffix": blobs[m["suffix"]], "bpv": m["bpv"], "R": m["R"], "E": m["E"], "D": m["D"],
               "keys": m["keys"], "mask": mask}
        out.append(rebuild_message(rec, X[i]))
    return out


def verify_roundtrip(messages: Sequence[bytes], path: str | Path, *, mode: str = "2d") -> dict[str, Any]:
    """Write, read back and compare byte-for-byte (SHA-256) and value-for-value (float64)."""
    ec = _eccodes()
    stats = write_grib_messages(path, messages, mode=mode)
    back = read_grib_messages(path)
    identical = sum(hashlib.sha256(a).digest() == hashlib.sha256(b).digest() for a, b in zip(messages, back))
    values_equal = 0
    for a, b in zip(messages, back):
        ga, gb = ec.codes_new_from_message(a), ec.codes_new_from_message(b)
        try:
            for g in (ga, gb):
                if ec.codes_is_defined(g, "bitmapPresent") and ec.codes_get(g, "bitmapPresent"):
                    ec.codes_set_double(g, "missingValue", float("nan"))
            va, vb = ec.codes_get_values(ga), ec.codes_get_values(gb)
            values_equal += bool(np.array_equal(va, vb, equal_nan=True))
        finally:
            ec.codes_release(ga)
            ec.codes_release(gb)
    stats.update(messages=len(messages), byte_identical=identical, values_float64_identical=values_equal)
    return stats


def read_field(path: str | Path, index: int, *, dtype: str = "float64") -> np.ndarray:
    """Decode ONE field (``index`` in original message order) of a :func:`write_grib_messages` file.

    Only that field's chunk is read.  Returns the physical values (bit-identical to the ecCodes decode for
    ``dtype="float64"``; masked points are NaN).  The stored chunk holds predictor residuals, so slicing
    ``/fields`` directly does NOT give data values - use this function (or ``predict_inverse``).
    """
    import wx1

    with wx1.open_file(str(path)) as f:
        v = next(v for v in f.variables if v["path"] == "/fields")
        blob = json.loads(_unb64z(v["attributes"]["grib:exact_records"]))
        nj, ni = blob["shape"]
        m = blob["fields"][index]
        if m["kind"] != "lattice":
            raise ValueError("field is stored verbatim (raw fallback); use read_grib_messages()")
        z = f.read_array_slice("/fields", [index, 0, 0], [1, nj, ni])[0].astype(np.uint16, copy=False)
    X = predict_inverse(z.copy(), blob["mode"])
    out = _lattice_values(X, m["R"], m["E"], m["D"])
    if "mask" in m:
        mask = np.unpackbits(np.frombuffer(base64.b64decode(blob["blobs"][m["mask"]]), np.uint8))[: nj * ni].astype(bool).reshape(nj, ni)
        out = np.where(mask, out, np.nan)
    return out.astype(dtype, copy=False)
