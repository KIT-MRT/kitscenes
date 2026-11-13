"""Reading the sensor Parquet files for :class:`~kitscenes.frame.Frame`.

Lidar coordinates go from the 5 mm grid to metres, and the no-return rows
(reflectivity -1 at the origin) are dropped unless asked for.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from kitscenes.points import ScanLayout, scan_layout

# A file path, or the raw bytes of a parquet file (e.g. a member read from a zip archive).
ParquetSource = Union[Path, str, bytes, bytearray, memoryview]

_LIDAR_POINT_TYPES = frozenset({"PointLGDataset", "PointKITScenes"})
_RADAR_POINT_TYPES = frozenset({"PointLGDatasetRadar", "PointKITScenesRadar"})


def _read_table(source: ParquetSource, columns: Optional[Sequence[str]] = None) -> pa.Table:
    if isinstance(source, (bytes, bytearray, memoryview)):
        source = pa.BufferReader(source)
    else:
        source = str(source)
    return pq.read_table(source, columns=list(columns) if columns is not None else None)


def _describe(source: ParquetSource) -> str:
    return "<in-memory parquet>" if isinstance(source, (bytes, bytearray, memoryview)) else str(source)


def _no_return_mask(reflectivity: np.ndarray, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Rows that are a fired ray without a return: (0, 0, 0) with reflectivity -1.

    Written for the Hesai sensors by the extraction tool and for the Velodyne
    by its unpacker, with the real firing time. The reflectivity-0 rows at the
    origin (slots without a firing, zeroed near-field returns, empty grid
    cells) are not no-returns.
    """
    # Coordinates are discretized and the marker is written as exactly -1, so
    # exact comparisons are equivalent to np.isclose here and ~4x faster.
    return (reflectivity == -1.0) & (x == 0.0) & (y == 0.0) & (z == 0.0)


def _read_lidar_table(
    path: ParquetSource, columns: Optional[Sequence[str]], drop_no_returns: bool,
) -> pa.Table:
    if columns is None:
        return _read_table(path)
    needed = list(columns)
    if drop_no_returns:
        needed += [c for c in ("x", "y", "z", "reflectivity") if c not in needed]
    return _read_table(path, needed)


def _lidar_columns_from_table(
    table: pa.Table,
    path: ParquetSource,
    *,
    drop_no_returns: bool,
    columns: Optional[Sequence[str]] = None,
    with_row: bool = False,
) -> dict[str, np.ndarray]:
    metadata = table.schema.metadata or {}

    point_type_name = metadata.get(b"point_type_name", b"").decode()
    if point_type_name not in _LIDAR_POINT_TYPES:
        raise RuntimeError(
            f"Expected LiDAR point type, got {point_type_name!r} in {_describe(path)}"
        )

    resolution = float(metadata.get(b"discretization_resolution", b"0.005"))
    arrays = {col: table.column(col).to_numpy() for col in table.column_names}
    for axis in ("x", "y", "z"):
        if axis in arrays:
            arrays[axis] = (arrays[axis] * resolution).astype(np.float32)

    keep = None
    if drop_no_returns and all(c in arrays for c in ("x", "y", "z", "reflectivity")):
        no_return = _no_return_mask(arrays["reflectivity"], arrays["x"], arrays["y"], arrays["z"])
        if no_return.any():
            keep = ~no_return

    out_columns = list(columns) if columns is not None else table.column_names
    # to_numpy() may return read-only views of Arrow buffers; hand out writable arrays.
    out = {
        col: arrays[col][keep] if keep is not None else _writable(arrays[col])
        for col in out_columns
    }
    if with_row:
        out["row"] = np.flatnonzero(keep).astype(np.uint32) if keep is not None \
            else np.arange(table.num_rows, dtype=np.uint32)
    return out


def _writable(array: np.ndarray) -> np.ndarray:
    return array if array.flags.writeable else array.copy()


def load_lidar_columns(
    path: ParquetSource,
    *,
    drop_no_returns: bool = True,
    columns: Optional[Sequence[str]] = None,
) -> dict[str, np.ndarray]:
    """A lidar sweep (file path, or the bytes of a Parquet file) as one array per column, writable.

    Args:
        drop_no_returns: Drop the no-return rows (reflectivity -1 at the
            origin). The ground-segmentation labels have one row per row
            that is left.
        columns: Only decode these columns (e.g. ``("x", "y", "z",
            "timestamp")``). The filter's inputs are read as needed and dropped
            again.
    """
    return _lidar_columns_from_table(
        _read_lidar_table(path, columns, drop_no_returns),
        path,
        drop_no_returns=drop_no_returns,
        columns=columns,
    )


def load_lidar_sweep(
    path: ParquetSource, *, drop_no_returns: bool = True,
) -> tuple[dict[str, np.ndarray], Optional[ScanLayout]]:
    """All columns of a sweep plus ``row`` (uint32, each row's index in the stored file), and its scan layout.

    The layout comes from the ``sensor_id`` column and the ``pcl_width`` /
    ``pcl_height`` metadata (:func:`kitscenes.points.scan_layout`); None for
    unorganized sweeps.
    """
    table = _read_table(path)
    columns = _lidar_columns_from_table(table, path, drop_no_returns=drop_no_returns, with_row=True)
    metadata = table.schema.metadata or {}
    sensor_id = None
    if "sensor_id" in table.column_names and table.num_rows:
        sensor_id = int(table.column("sensor_id")[0].as_py())
    try:
        width = int(metadata.get(b"pcl_width", b"0"))
        height = int(metadata.get(b"pcl_height", b"0"))
    except ValueError:
        width = height = 0
    return columns, scan_layout(sensor_id, width, height, table.num_rows)


def load_radar_columns(path: ParquetSource) -> dict[str, np.ndarray]:
    """A radar file (path or bytes) as one array per column."""
    table = _read_table(path)
    metadata = table.schema.metadata or {}
    point_type_name = metadata.get(b"point_type_name", b"").decode()
    if point_type_name and point_type_name not in _RADAR_POINT_TYPES:
        raise RuntimeError(f"Expected radar point type, got {point_type_name!r} in {_describe(path)}")
    return {col: _writable(table.column(col).to_numpy()) for col in table.column_names}


def load_ground_mask_parquet(path: ParquetSource) -> np.ndarray:
    """Load a per-point ground segmentation file as a boolean array.

    The file holds one ``isground`` row per row of the matching lidar sweep
    without its no-return rows (reflectivity -1 at the origin), in the order
    :func:`load_lidar_columns` returns them. Rows at the origin with
    reflectivity 0 (slots without a point) have a label too, which means
    nothing.
    """
    table = _read_table(path)
    if "isground" not in table.column_names:
        raise RuntimeError(f"No 'isground' column in ground segmentation file {_describe(path)}")
    return table.column("isground").to_numpy().astype(bool)
