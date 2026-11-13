"""Writing point clouds as PCD (PCL, Open3D, CloudCompare) or PLY files.

World-frame clouds keep float64 coordinates (map-frame values reach
thousands of metres, where float32 has only millimetre precision); other
clouds are written as float32.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from kitscenes.points import PointCloud

_PCD_TYPES = {"f": "F", "i": "I", "u": "U", "b": "U"}
_PLY_TYPES = {
    np.dtype(np.float32): "float", np.dtype(np.float64): "double", np.dtype(np.int8): "char",
    np.dtype(np.uint8): "uchar", np.dtype(np.int16): "short", np.dtype(np.uint16): "ushort",
    np.dtype(np.int32): "int", np.dtype(np.uint32): "uint", np.dtype(np.bool_): "uchar",
}


def _table(cloud: PointCloud, fields: Optional[Sequence[str]], measured_only: bool) -> np.ndarray:
    keep = cloud.measured if measured_only else np.ones(len(cloud), bool)
    xyz_dtype = np.float64 if cloud.frame == "world" else np.float32
    names = list(cloud.attrs) if fields is None else list(fields)
    if cloud.source_index is not None and fields is None:
        names.append("source")
    columns = {"x": cloud.xyz[keep, 0], "y": cloud.xyz[keep, 1], "z": cloud.xyz[keep, 2]}
    for name in names:
        values = cloud.source_index if name == "source" else cloud.attrs[name]
        values = np.asarray(values)[keep]
        if values.dtype == np.bool_:
            values = values.astype(np.uint8)
        if values.dtype == np.int64 or values.dtype == np.uint64:  # not in every PCD/PLY reader
            values = values.astype(np.float64)
        columns[name] = values
    table = np.empty(int(keep.sum()), dtype=[(k, xyz_dtype if k in "xyz" else v.dtype) for k, v in columns.items()])
    for k, v in columns.items():
        table[k] = v
    return table


def write_pcd(path: Path | str, cloud: PointCloud, fields: Optional[Sequence[str]] = None, *, measured_only: bool = True) -> Path:
    """Write a binary PCD (v0.7) file.

    Args:
        fields: Attributes to include besides x, y, z (default: all, plus
            ``source`` for fused clouds).
        measured_only: Leave out rows without a measurement.
    """
    table = _table(cloud, fields, measured_only)
    names = table.dtype.names
    header = "\n".join([
        "# .PCD v0.7 - written by kitscenes",
        "VERSION 0.7",
        "FIELDS " + " ".join(names),
        "SIZE " + " ".join(str(table.dtype[n].itemsize) for n in names),
        "TYPE " + " ".join(_PCD_TYPES[table.dtype[n].kind] for n in names),
        "COUNT " + " ".join("1" for _ in names),
        f"WIDTH {len(table)}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {len(table)}",
        "DATA binary",
    ]) + "\n"
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(table.astype(table.dtype.newbyteorder("<")).tobytes())
    return path


def write_ply(path: Path | str, cloud: PointCloud, fields: Optional[Sequence[str]] = None, *, measured_only: bool = True) -> Path:
    """Write a binary little-endian PLY file (vertices only); arguments as :func:`write_pcd`."""
    table = _table(cloud, fields, measured_only)
    lines = ["ply", "format binary_little_endian 1.0", "comment written by kitscenes", f"element vertex {len(table)}"]
    lines += [f"property {_PLY_TYPES[table.dtype[n]]} {n}" for n in table.dtype.names]
    lines.append("end_header")
    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode("ascii"))
        fh.write(table.astype(table.dtype.newbyteorder("<")).tobytes())
    return path
