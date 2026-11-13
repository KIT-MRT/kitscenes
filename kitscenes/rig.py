"""The sensor rig of a scene, from ``calibration/calib.json``.

Every sensor has ``T_base_sensor``: the (4, 4) transform that maps points
from its own frame into the vehicle's reference (base) frame,
``T_to_reference`` in ``calib.json``. Transforms are named ``T_<to>_<from>``
throughout the API (``T_world_base`` maps base-frame points into the world). The base frame is x forward, y left, z up. Cameras use the
OpenCV convention (x right, y down, z forward) and are distortion-free
pinholes with one focal length.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Mapping, Optional

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.constants import CAMERA_NAMES, LIDAR_NAMES, RADAR_NAMES, SensorNameLike, sensor_name
from kitscenes.errors import CalibrationError
from kitscenes.messages import IMU_TO_BASE

_SUPPORTED_CAMERA_MODEL = "pinhole"
_REQUIRED_CAMERA_KEYS = frozenset({"T_to_reference", "camera_model", "intrinsics", "resolution"})
_PINHOLE_INTRINSIC_KEYS = frozenset({"focal_length", "principal_point_u", "principal_point_v"})


#: The ``calib.json`` entry of the IMU.
IMU_ENTRY = "gnss_ins_imu"
#: A ``gnss_ins_imu`` rotation further than this from :data:`kitscenes.messages.IMU_TO_BASE` is invalid (degrees).
MAX_IMU_DEVIATION_DEG = 5.0


def imu_deviation_deg(rotation: np.ndarray) -> float:
    """Angle in degrees between an IMU-to-base rotation and the nominal axes."""
    cos = (np.trace(np.asarray(rotation) @ IMU_TO_BASE.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply a (4, 4) transform to (N, 3) points; keeps float32 input as float32."""
    xyz = np.asarray(xyz)
    dtype = np.float32 if xyz.dtype == np.float32 else np.float64
    return (xyz @ T[:3, :3].T.astype(dtype) + T[:3, 3].astype(dtype)).astype(dtype, copy=False)


def _readonly(array) -> np.ndarray:
    """A read-only float64 copy of *array*."""
    out = np.array(array, dtype=np.float64, copy=True)
    out.flags.writeable = False
    return out


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid (4, 4) transform."""
    inv = np.eye(4)
    inv[:3, :3] = T[:3, :3].T
    inv[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return inv


def _transform(entry: dict, name: str) -> np.ndarray:
    try:
        T = np.array(entry["T_to_reference"], dtype=np.float64)
    except KeyError:
        raise CalibrationError(f"Calibration for {name!r} has no T_to_reference.") from None
    if T.shape != (4, 4):
        raise CalibrationError(f"T_to_reference of {name!r} has shape {T.shape}, expected (4, 4).")
    return T


@dataclasses.dataclass(frozen=True, eq=False)
class Sensor:
    """A calibrated sensor.

    Attributes:
        name: Sensor name, e.g. ``"lidar_top"``.
        T_base_sensor: (4, 4) float64, maps sensor-frame points into the base frame.
    """

    __repr__ = compact_repr

    name: str
    T_base_sensor: np.ndarray

    def __post_init__(self) -> None:
        # Shared by every frame of a cached scene: a read-only copy, so an in-place edit cannot leak.
        object.__setattr__(self, "T_base_sensor", _readonly(self.T_base_sensor))

    @property
    def T_sensor_base(self) -> np.ndarray:
        """(4, 4) float64, maps base-frame points into the sensor frame (the inverse of :attr:`T_base_sensor`)."""
        return invert_transform(self.T_base_sensor)


@dataclasses.dataclass(frozen=True, eq=False)
class Lidar(Sensor):
    """A calibrated lidar."""

    __repr__ = compact_repr


@dataclasses.dataclass(frozen=True, eq=False)
class Radar(Sensor):
    """A calibrated radar."""

    __repr__ = compact_repr


@dataclasses.dataclass(frozen=True, eq=False)
class Camera(Sensor):
    """A calibrated pinhole camera.

    Attributes:
        K: (3, 3) float64 intrinsic matrix.
        image_size: ``(width, height)`` in pixels.
    """

    __repr__ = compact_repr

    K: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    image_size: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "K", _readonly(self.K))

    def project(
        self, xyz_base: np.ndarray, *, min_depth: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project base-frame points into the image (the API's one pinhole, see :mod:`kitscenes.projection`).

        float32 points are projected in float32 (millions of lidar points),
        anything else in float64.

        Returns:
            ``(uv, depth, visible)``: (N, 2) pixel coordinates (pixel centres
            on integers, as OpenCV), (N,) depth along the optical axis in
            metres, and (N,) bool for points at least *min_depth* in front of
            the camera and inside the image (``-0.5 <= u < width - 0.5``,
            likewise v). ``uv`` of points closer than *min_depth* is NaN.
        """
        xyz = np.asarray(xyz_base)
        dtype = np.float32 if xyz.dtype == np.float32 else np.float64
        xyz_cam = transform_points(self.T_sensor_base, xyz.astype(dtype, copy=False))
        depth = xyz_cam[:, 2]
        front = depth >= min_depth
        safe = np.where(front, depth, dtype(1.0))
        p = xyz_cam @ self.K.T.astype(dtype)
        uv = p[:, :2] / safe[:, None]
        uv[~front] = np.nan
        w, h = self.image_size
        visible = front & (uv[:, 0] >= -0.5) & (uv[:, 0] < w - 0.5) & (uv[:, 1] >= -0.5) & (uv[:, 1] < h - 0.5)
        return uv, depth, visible

    def scaled(self, factor: float) -> "Camera":
        """The camera for an image resized by *factor* (e.g. 0.5 for half size).

        Pixel centres are on integers (see :meth:`project`), so the principal
        point maps as ``(c + 0.5) * factor - 0.5``: the centre of the image's
        top-left pixel stays the centre of the resized one's (area resizing,
        as the JPEG DCT reduce of :meth:`kitscenes.Frame.image` does).
        """
        K = self.K.copy()
        K[:2] *= factor
        K[:2, 2] += 0.5 * factor - 0.5
        w, h = self.image_size
        return dataclasses.replace(self, K=K, image_size=(round(w * factor), round(h * factor)))

    def cropped(self, x0: int, y0: int, width: int, height: int) -> "Camera":
        """The camera for the image crop ``[y0:y0 + height, x0:x0 + width]``."""
        K = self.K.copy()
        K[0, 2] -= x0
        K[1, 2] -= y0
        return dataclasses.replace(self, K=K, image_size=(int(width), int(height)))

    @classmethod
    def from_calib_entry(cls, name: str, entry: dict) -> "Camera":
        """Build from a ``calib.json`` camera entry.

        Raises:
            CalibrationError: If the entry is incomplete, not a pinhole, or has
                intrinsics beyond the pinhole set (a distortion coefficient or a
                second focal length would give a plausible but wrong projection).
        """
        missing = _REQUIRED_CAMERA_KEYS - set(entry)
        if missing:
            raise CalibrationError(f"Calibration for camera {name!r} is missing {sorted(missing)} in calib.json.")
        if entry["camera_model"] != _SUPPORTED_CAMERA_MODEL:
            raise CalibrationError(
                f"Camera {name!r} uses camera_model {entry['camera_model']!r}; only "
                f"{_SUPPORTED_CAMERA_MODEL!r} is supported."
            )
        intr = entry["intrinsics"]
        if set(intr) != _PINHOLE_INTRINSIC_KEYS:
            raise CalibrationError(
                f"Camera {name!r} has intrinsics {sorted(intr)}; a pinhole model expects exactly "
                f"{sorted(_PINHOLE_INTRINSIC_KEYS)}."
            )
        f, cu, cv = (float(intr[k]) for k in ("focal_length", "principal_point_u", "principal_point_v"))
        K = np.array([[f, 0.0, cu], [0.0, f, cv], [0.0, 0.0, 1.0]])
        res = entry["resolution"]
        return cls(name, _transform(entry, name), K, (int(res["width"]), int(res["height"])))


class _Sensors(dict):
    """Sensors of one kind by name; a missing name says which ones there are."""

    def __init__(self, kind: str, sensors: Mapping) -> None:
        super().__init__(sensors)
        self.kind = kind

    def __missing__(self, name: str):
        raise KeyError(f"No {self.kind} {name!r} in this scene's calib.json; it has {sorted(self) or 'none'}")


@dataclasses.dataclass(frozen=True, eq=False)
class Rig:
    """All calibrated sensors of a scene.

    ``calib.json`` has a valid entry for every sensor with data (the vehicle
    carries all 19 and the release ships them all); :attr:`kitscenes.Scene.rig`
    raises a :class:`~kitscenes.errors.CalibrationError` if it does not.

    Attributes:
        cameras, lidars, radars: Sensors by name, in the order of
            :data:`kitscenes.constants.CAMERA_NAMES` etc.
        transforms: ``T_base_sensor`` of every ``calib.json`` entry that
            has one, including the GNSS antennas and the IMU.
    """

    __repr__ = compact_repr

    cameras: Mapping[str, Camera]
    lidars: Mapping[str, Lidar]
    radars: Mapping[str, Radar]
    transforms: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        for name, kind in (("cameras", "camera"), ("lidars", "lidar"), ("radars", "radar")):
            sensors = getattr(self, name)
            if not isinstance(sensors, _Sensors):
                object.__setattr__(self, name, _Sensors(kind, sensors))

    def sensor(self, name: SensorNameLike) -> Sensor:
        """The calibrated sensor of that name.

        Raises:
            KeyError: For an unknown sensor name.
            CalibrationError: If ``calib.json`` has no entry for it.
        """
        name = sensor_name(name)
        kind = _KINDS.get(name)
        if kind is None:
            raise KeyError(f"Unknown sensor {name!r}")
        sensors = {"camera": self.cameras, "lidar": self.lidars, "radar": self.radars}[kind]
        if name not in sensors:
            raise CalibrationError(f"calib.json has no entry for {name!r}")
        return sensors[name]

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return (*self.cameras, *self.lidars, *self.radars)

    @property
    def imu_to_base(self) -> Optional[np.ndarray]:
        """(3, 3) rotation from the IMU's axes to the base frame's, from ``gnss_ins_imu``.

        None if ``calib.json`` has no such entry or its rotation is more than
        :data:`MAX_IMU_DEVIATION_DEG` from the nominal axes (an identity
        rotation, say).
        """
        T = self.transforms.get(IMU_ENTRY)
        if T is None or not imu_deviation_deg(T[:3, :3]) <= MAX_IMU_DEVIATION_DEG:
            return None
        return np.array(T[:3, :3])  # a copy: rig.transforms are read-only

    @classmethod
    def from_calib(cls, calib: dict, source: str = "calib.json") -> "Rig":
        """Build from the parsed ``calib.json``.

        Raises:
            CalibrationError: If it is not a JSON object or has an invalid
                entry (no (4, 4) ``T_to_reference``, a camera that is not a
                distortion-free pinhole); *source* names the file.
        """
        if not isinstance(calib, dict):
            raise CalibrationError(f"{source} is not a JSON object")

        def build(name: str, make):
            try:
                return make(name, calib[name])
            except CalibrationError as err:
                raise CalibrationError(f"{source}: {err}") from None
            except (KeyError, TypeError, ValueError) as err:
                raise CalibrationError(f"{source}: invalid entry for {name!r}: {err!r}") from None

        cameras, lidars, radars = (
            {n: build(n, make) for n in names if n in calib}
            for names, make in (
                (CAMERA_NAMES, Camera.from_calib_entry),
                (LIDAR_NAMES, lambda n, e: Lidar(n, _transform(e, n))),
                (RADAR_NAMES, lambda n, e: Radar(n, _transform(e, n))),
            )
        )
        transforms = {  # read-only: shared by every user of a cached scene
            n: _readonly(build(n, lambda n, e: _transform(e, n)))
            for n, e in calib.items() if isinstance(e, dict) and "T_to_reference" in e
        }
        return cls(cameras, lidars, radars, transforms)

    @classmethod
    def from_json_bytes(cls, data: bytes, source: str = "calib.json") -> "Rig":
        """Build from the bytes of ``calib.json``.

        Raises:
            CalibrationError: If it is not valid JSON, or as :meth:`from_calib`.
        """
        try:
            calib = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as err:
            raise CalibrationError(f"{source} is not valid JSON: {err}") from None
        return cls.from_calib(calib, source)

    @classmethod
    def empty(cls) -> "Rig":
        return cls({}, {}, {}, {})


#: Sensor kind by name.
_KINDS = {
    **{n: "camera" for n in CAMERA_NAMES}, **{n: "lidar" for n in LIDAR_NAMES}, **{n: "radar" for n in RADAR_NAMES},
}
