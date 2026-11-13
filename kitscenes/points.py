"""Point clouds as columns: an (N, 3) xyz array plus per-point attributes.

The points of a lidar sweep are in the order of the stored file, and
``attrs["row"]`` is each point's row in it. ``frame.lidar(name, rows=True)``
returns every stored row. What a row at the sensor origin means depends on
the sensor (see :attr:`PointCloud.measured`, :attr:`PointCloud.no_return`
and docs/sensors.md, "Lidar rows"):

- **Hesai PandarXT-32** (``lidar_corner_*``) and **Hesai OT128**
  (``lidar_top``, 2026 generation): one row per firing slot, column-major,
  ``row = (column * 2 + return) * rings + ring``. A fired ray without a
  return is stored as (0, 0, 0) with reflectivity -1 and its firing time
  (a *no-return*). The OT128 also has (0, 0, 0) rows with reflectivity 0:
  slots its high-performance firing pattern leaves out (exactly a quarter
  of the rows; rings 0-23 and 88-127 fire every second column; nominal
  column time) and returns closer than 1.4 m, which the extraction zeroes.
- **Velodyne VLS-128 AP** (``lidar_top``, 2025 generation): a fixed azimuth
  grid, ring-major, ``row = ring * columns + column``. No-returns as for the
  Hesai; grid cells no firing fell into are (0, 0, 0) with reflectivity 0
  and timestamp 0.
- **Seyond Falcon K1** (``lidar_front/left/rear/right``): an unorganized
  cloud of returns only, no rows at the origin.

``frame.lidar(name)`` returns the measured points only; ``rows=True`` adds
the slots without a point and ``no_returns=True`` the no-return rows.
"""

from __future__ import annotations

import dataclasses
from types import MappingProxyType
from typing import Literal, Mapping, Optional, Union

import numpy as np

from kitscenes.schema import Provenance

#: ``sensor_id`` of the stored lidar rows -> (model, order, returns, azimuth step in degrees per column).
#: The steps are nominal and were checked on the release data: columns turn clockwise seen from
#: above (azimuth decreasing), so the steps are negative.
_ORGANIZED_SENSORS = {
    0: ("Velodyne VLS-128 AP", "ring-major", 1, -360.0 / 1708),
    1: ("Hesai PandarXT-32", "column-major", 2, -0.18),
    2: ("Hesai PandarXT-32", "column-major", 2, -0.18),
    7: ("Hesai OT128", "column-major", 2, -0.1),
}


@dataclasses.dataclass(frozen=True)
class ScanLayout:
    """How the rows of an organized lidar sweep map to firing slots (return, ring, column).

    Ring 0 is the top laser (highest elevation) for every model; columns
    follow the scan clockwise seen from above. The Hesai sweeps are
    column-major with both returns of a column next to each other, the
    Velodyne sweeps ring-major. Both store ``pcl_width`` x ``pcl_height`` in
    the parquet metadata the same way, so a plain ``reshape(height, width)``
    is right for the Velodyne and scrambles a Hesai sweep; use
    :meth:`position` or :meth:`PointCloud.range_image`.

    Attributes:
        model: ``"Hesai OT128"``, ``"Hesai PandarXT-32"`` or ``"Velodyne VLS-128 AP"``.
        order: ``"column-major"`` (Hesai: ``row = (column * returns + return)
            * rings + ring``) or ``"ring-major"`` (Velodyne: ``row = ring *
            columns + column``).
        rings: Lasers (H): 128 or 32.
        columns: Azimuth steps (W): OT128 3600 (360 degrees), XT32 1500 or
            1504 (only 270 degrees are stored, the side towards the car is
            not), Velodyne 1708 fixed azimuth cells (360 degrees).
        returns: Returns per firing slot: 2 for the Hesai (return 1 has bit 8
            of ``sensor_specific_data`` set), 1 for the Velodyne.
        azimuth_step_deg: Nominal azimuth step per column, negative because
            the columns turn clockwise: OT128 -0.1, XT32 -0.18, Velodyne
            -360 / 1708.
    """

    model: str
    order: Literal["column-major", "ring-major"]
    rings: int
    columns: int
    returns: int
    azimuth_step_deg: float

    @property
    def num_rows(self) -> int:
        """Rows of a stored sweep: ``returns * rings * columns``."""
        return self.returns * self.rings * self.columns

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(returns, rings, columns)``, the shape of :meth:`PointCloud.range_image`."""
        return self.returns, self.rings, self.columns

    def position(self, row: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(ring, column, return)`` of stored rows (int64 arrays like *row*)."""
        row = np.asarray(row, dtype=np.int64)
        if self.order == "column-major":
            ring = row % self.rings
            slot = row // self.rings
            return ring, slot // self.returns, slot % self.returns
        return row // self.columns, row % self.columns, np.zeros_like(row)

    def row(self, ring: np.ndarray, column: np.ndarray, return_: np.ndarray | int = 0) -> np.ndarray:
        """The stored row of firing slots (the inverse of :meth:`position`)."""
        ring, column, return_ = (np.asarray(a, dtype=np.int64) for a in (ring, column, return_))
        if self.order == "column-major":
            return (column * self.returns + return_) * self.rings + ring
        return ring * self.columns + column


def scan_layout(sensor_id: Optional[int], width: int, height: int, num_rows: int) -> Optional[ScanLayout]:
    """The layout of a stored sweep from its ``sensor_id`` and ``pcl_width``/``pcl_height`` metadata.

    None for unorganized clouds (Seyond, ``height == 1``), an unknown
    sensor, or metadata that does not match the row count.
    """
    if sensor_id is None or height <= 1 or int(sensor_id) not in _ORGANIZED_SENSORS:
        return None
    model, order, returns, step = _ORGANIZED_SENSORS[int(sensor_id)]
    if width * height != num_rows or width % returns:
        return None
    return ScanLayout(model, order, height, width // returns, returns, step)


@dataclasses.dataclass(frozen=True, eq=False)
class PointCloud:
    """Points of one sensor (or several, fused) at one frame.

    Attributes:
        xyz: (N, 3) float32 positions in :attr:`frame`.
        attrs: Per-point columns, each (N,): for lidar ``reflectivity``,
            ``timestamp`` (firing time in seconds since the scene start),
            ``ring``, ``sensor_id``, ``sensor_specific_data`` as stored, and
            ``row`` (uint32, the row's index in the stored sweep, before the
            no-return rows are dropped).
        measured: (N,) bool, True where the row holds a measured point.
            False for rows at the sensor origin (0, 0, 0): firing slots that
            never fired, OT128 returns closer than 1.4 m that the extraction
            zeroed, empty Velodyne grid cells, and (with ``no_returns=True``)
            fired rays without a return. Only a cloud read with
            ``frame.lidar(name, rows=True)`` (or a radar cloud) has such rows;
            they stay at the origin in every frame and :meth:`compact` drops them.
        frame: ``"base"`` (vehicle reference frame at :attr:`t_ns`),
            ``"road"`` (:attr:`kitscenes.Scene.T_road_base`), ``"sensor"`` or
            ``"world"`` (the frame of the poses, for accumulated clouds).
        t_ns: Frame time in nanoseconds since the scene start.
        sensor: Sensor name, or the names of the fused sensors.
        motion_compensated: Whether the points were moved to :attr:`t_ns`
            with the ego motion (deskewed).
        reason: Why they were not, else None.
        source_index: For fused clouds, (N,) uint8 index into :attr:`sensor`.
        provenance: How the cloud was made (fused clouds are geometry-derived).
        layout: How ``attrs["row"]`` maps to firing slots, for the Hesai and
            Velodyne sweeps (:class:`ScanLayout`); None for the Seyond lidars,
            fused and accumulated clouds and radar.
    """

    xyz: np.ndarray
    attrs: Mapping[str, np.ndarray]
    measured: np.ndarray
    frame: str
    t_ns: int
    sensor: Union[str, tuple[str, ...]]
    motion_compensated: bool = False
    reason: Optional[str] = None
    source_index: Optional[np.ndarray] = None
    provenance: Provenance = Provenance.SOURCE
    layout: Optional[ScanLayout] = None

    def __post_init__(self) -> None:
        n = len(self.xyz)
        if self.xyz.shape != (n, 3):
            raise ValueError(f"xyz must be (N, 3), got {self.xyz.shape}")
        for name, column in self.attrs.items():
            if len(column) != n:
                raise ValueError(f"Attribute {name!r} has {len(column)} rows, xyz has {n}.")
        if self.measured.shape != (n,):
            raise ValueError(f"measured must be ({n},), got {self.measured.shape}")
        if self.frame not in ("base", "road", "sensor", "world"):
            raise ValueError(f"frame must be 'base', 'road', 'sensor' or 'world', got {self.frame!r}")

    def __len__(self) -> int:
        return len(self.xyz)

    def __repr__(self) -> str:
        comp = "compensated" if self.motion_compensated else f"not compensated ({self.reason})"
        no_return = int(self.no_return.sum())
        extra = f", no_return={no_return}" if no_return else ""
        return (
            f"{type(self).__name__}(sensor={self.sensor!r}, n={len(self)}, measured={int(self.measured.sum())}"
            f"{extra}, frame={self.frame!r}, t_ns={self.t_ns}, {comp})"
        )

    def __getitem__(self, name: str) -> np.ndarray:
        """A column by name: ``"x"``, ``"y"``, ``"z"`` or an attribute."""
        axis = "xyz".find(name) if len(name) == 1 else -1
        if axis >= 0:
            return self.xyz[:, axis]
        return self.attrs[name]

    # -- typed columns: the arrays of ``attrs`` under a name the IDE knows ----------

    def _column(self, name: str) -> np.ndarray:
        try:
            return self.attrs[name]
        except KeyError:
            raise AttributeError(
                f"This {type(self).__name__} of {self.sensor!r} has no {name!r} column; it has {sorted(self.attrs)}"
            ) from None

    @property
    def reflectivity(self) -> np.ndarray:
        """(N,) float32 lidar reflectivity, ``attrs["reflectivity"]`` (-1 marks a no-return row)."""
        return self._column("reflectivity")

    @property
    def ring(self) -> np.ndarray:
        """(N,) uint8 laser ring of each lidar row, ``attrs["ring"]``."""
        return self._column("ring")

    @property
    def ground(self) -> np.ndarray:
        """(N,) bool ground-segmentation label of each row, ``attrs["ground"]`` (``frame.lidar(name, labels=True)``)."""
        return self._column("ground")

    @property
    def row(self) -> np.ndarray:
        """(N,) uint32 index of each point's row in its stored sweep, ``attrs["row"]`` (see :attr:`layout`)."""
        return self._column("row")

    @property
    def frame_index(self) -> np.ndarray:
        """(N,) frame index each point of an accumulated cloud comes from, ``attrs["frame_index"]``."""
        return self._column("frame_index")

    @property
    def point_t_ns(self) -> np.ndarray:
        """(N,) int64 firing (lidar) or detection (radar) time of each point in nanoseconds since the scene start.

        ``attrs["timestamp"]`` holds the same times in seconds (float64), as
        stored. :attr:`t_ns` is the frame time the cloud is compensated to.
        """
        return np.rint(np.asarray(self._column("timestamp"), dtype=np.float64) * 1e9).astype(np.int64)

    @property
    def no_return(self) -> np.ndarray:
        """(N,) bool: rows that are a fired ray without a return (reflectivity -1 at the origin).

        Only a sweep loaded with ``frame.lidar(name, rows=True, no_returns=True)`` has
        such rows (Hesai and Velodyne; they carry their firing time, ring and
        ``row``, xyz is 0); everywhere else this is all False.
        """
        reflectivity = self.attrs.get("reflectivity")
        if reflectivity is None:
            return np.zeros(len(self), dtype=bool)
        return ~self.measured & (np.asarray(reflectivity) == -1)

    def select(self, mask: np.ndarray) -> "PointCloud":
        """The rows where *mask* is True (a bool or index array)."""
        return dataclasses.replace(
            self,
            xyz=self.xyz[mask],
            attrs={k: v[mask] for k, v in self.attrs.items()},
            measured=self.measured[mask],
            source_index=None if self.source_index is None else self.source_index[mask],
        )

    def compact(self) -> "PointCloud":
        """Only the measured rows (what ``frame.lidar(name)`` returns without ``rows=True``)."""
        return self if bool(self.measured.all()) else self.select(self.measured)

    def transformed(self, T: np.ndarray, frame: str) -> "PointCloud":
        """The cloud with its measured points moved by the (4, 4) transform *T* into *frame*.

        E.g. ``cloud.transformed(frame.T_world_base, "world")`` for a base-frame
        cloud. Rows without a measurement stay at the origin.
        """
        from kitscenes.rig import transform_points

        xyz = np.zeros_like(self.xyz)
        xyz[self.measured] = transform_points(np.asarray(T, dtype=np.float64), self.xyz[self.measured])
        return dataclasses.replace(self, xyz=xyz, frame=frame)

    def _layout_rows(self) -> tuple[ScanLayout, np.ndarray]:
        if self.layout is None:
            raise ValueError(
                f"{self.sensor!r} has no scan layout: only single Hesai and Velodyne sweeps are organized "
                "(the Seyond lidars store unorganized clouds; fused and accumulated clouds mix sweeps)."
            )
        if "row" not in self.attrs:
            raise ValueError("The cloud has no 'row' attribute (the stored row of each point).")
        return self.layout, np.asarray(self.attrs["row"])

    def scan_position(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(ring, column, return)`` of every row, from ``attrs["row"]`` and :attr:`layout`.

        The ring equals the stored ``ring`` column; the column is the azimuth
        step in scan order (see :class:`ScanLayout`).

        Raises:
            ValueError: Without a layout (Seyond, fused or accumulated clouds).
        """
        layout, row = self._layout_rows()
        return layout.position(row)

    def range_image(self, value: Union[str, np.ndarray] = "range", *, fill: float = np.nan) -> np.ndarray:
        """The sweep as a ``(returns, rings, columns)`` image (:attr:`ScanLayout.shape`).

        Row ``r`` of the image is ring ``r`` (ring 0, the top laser, at the
        top); columns follow the scan clockwise. Built from ``attrs["row"]``,
        so it is right for the column-major Hesai and the ring-major Velodyne
        sweeps alike, and for clouds with rows dropped (by :meth:`select` or
        the no-return filter): their slots get *fill*.

        Args:
            value: ``"range"`` (distance from the sensor of the measured
                rows; *fill* elsewhere; needs ``frame="sensor"``: load with
                ``frame.lidar(name, frame="sensor", deskew=False, rows=True)`` for the
                ranges as measured), the name of an attribute (e.g.
                ``"reflectivity"``, ``"timestamp"``) or ``"x"``/``"y"``/``"z"``,
                or an (N,) array, one value per row.
            fill: Value of the slots without a row (and of unmeasured rows for
                ``"range"``). With NaN the image is floating point, else it
                has the dtype of the values (*fill* must fit it).

        Raises:
            ValueError: Without a layout (Seyond, fused or accumulated
                clouds), or ``"range"`` of a cloud not in the sensor frame.
        """
        layout, row = self._layout_rows()
        if isinstance(value, str) and value == "range":
            if self.frame != "sensor":
                raise ValueError(
                    f"range_image('range') needs the sensor frame, the cloud is in the {self.frame} frame: "
                    "load it with frame.lidar(name, frame='sensor', deskew=False)."
                )
            values = np.where(self.measured, np.linalg.norm(self.xyz, axis=1), fill)
        else:
            values = np.asarray(self[value] if isinstance(value, str) else value)
            if values.shape != (len(self),):
                raise ValueError(f"value must have one entry per row ({len(self)},), got {values.shape}")
        out = np.full(layout.shape, fill, dtype=np.result_type(values.dtype, np.float32) if np.isnan(fill)
                      else values.dtype)
        ring, column, return_ = layout.position(row)
        out[return_, ring, column] = values
        return out

    def to_structured(self) -> np.ndarray:
        """One structured array with fields ``x``, ``y``, ``z`` and the attributes."""
        fields = [("x", self.xyz.dtype), ("y", self.xyz.dtype), ("z", self.xyz.dtype)]
        fields += [(k, v.dtype) for k, v in self.attrs.items()]
        out = np.empty(len(self), dtype=fields)
        for i, axis in enumerate("xyz"):
            out[axis] = self.xyz[:, i]
        for k, v in self.attrs.items():
            out[k] = v
        return out


@dataclasses.dataclass(frozen=True, eq=False, repr=False)
class RadarCloud(PointCloud):
    """Radar detections, ego-motion compensated (see :func:`kitscenes.motion.compensate_radar_points`).

    ``motion_compensated`` refers to the positions; ``range``/``azimuth``/
    ``elevation`` are recomputed from them and stay relative to the radar.

    Attributes:
        doppler_compensated: Whether ``range_rate`` has the ego radial
            velocity removed (static returns ~0 m/s).
        capture_t_ns: Capture time of the cloud (median of the per-detection
            times), if known.
    """

    doppler_compensated: bool = False
    capture_t_ns: Optional[int] = None

    @property
    def range(self) -> np.ndarray:
        """(N,) float32 distance from the radar in metres, ``attrs["range"]``."""
        return self._column("range")

    @property
    def azimuth(self) -> np.ndarray:
        """(N,) float32 azimuth in the radar frame in radians, ``attrs["azimuth"]``."""
        return self._column("azimuth")

    @property
    def elevation(self) -> np.ndarray:
        """(N,) float32 elevation in the radar frame in radians, ``attrs["elevation"]``."""
        return self._column("elevation")

    @property
    def range_rate(self) -> np.ndarray:
        """(N,) float32 radial velocity in m/s, ``attrs["range_rate"]`` (ego motion removed if :attr:`doppler_compensated`)."""
        return self._column("range_rate")

    @property
    def rcs(self) -> np.ndarray:
        """(N,) float32 radar cross section as stored, ``attrs["rcs"]``."""
        return self._column("rcs")


def readonly(cloud: PointCloud) -> PointCloud:
    """Make a cloud immutable in place: read-only arrays and attribute mapping (for clouds shared through a cache)."""
    object.__setattr__(cloud, "attrs", MappingProxyType(dict(cloud.attrs)))
    arrays = [cloud.xyz, cloud.measured, *cloud.attrs.values()]
    if cloud.source_index is not None:
        arrays.append(cloud.source_index)
    for array in arrays:
        array.flags.writeable = False
    return cloud
