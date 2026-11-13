"""Dataset-wide constants: sensor names and environment variables.

File names and the scene layout are in :mod:`kitscenes._layout`.
"""

from __future__ import annotations

from enum import Enum
from typing import Final, Iterable, Literal, Union

# ---------------------------------------------------------------------------
# Sensor names
# ---------------------------------------------------------------------------
# Every method that takes a sensor name accepts the plain string or a member of
# the name enums (``frame.lidar("lidar_top")`` == ``frame.lidar(LidarName.TOP)``);
# the members are strings themselves. Names the API returns (dict keys,
# ``cloud.sensor``, sample keys) are always plain strings, so pickles and JSON
# written from them load without kitscenes.


class _SensorName(str, Enum):
    """A sensor name; equal to, hashed like and formatted as its string value."""

    def __str__(self) -> str:
        return self.value

    __format__ = str.__format__


class CameraName(_SensorName):
    """The camera names (``CameraName.RING_FRONT == "camera_ring_front"``)."""

    BASE_FRONT_CENTER = "camera_base_front_center"
    BASE_FRONT_LEFT_RECT = "camera_base_front_left_rect"
    BASE_FRONT_RIGHT_RECT = "camera_base_front_right_rect"
    RING_FRONT = "camera_ring_front"
    RING_FRONT_LEFT = "camera_ring_front_left"
    RING_FRONT_RIGHT = "camera_ring_front_right"
    RING_REAR = "camera_ring_rear"
    RING_REAR_LEFT = "camera_ring_rear_left"
    RING_REAR_RIGHT = "camera_ring_rear_right"


class LidarName(_SensorName):
    """The lidar names (``LidarName.TOP == "lidar_top"``)."""

    TOP = "lidar_top"
    FRONT = "lidar_front"
    LEFT = "lidar_left"
    RIGHT = "lidar_right"
    REAR = "lidar_rear"
    CORNER_LEFT = "lidar_corner_left"
    CORNER_RIGHT = "lidar_corner_right"


class RadarName(_SensorName):
    """The radar names (``RadarName.FRONT == "radar_front"``)."""

    FRONT = "radar_front"
    LEFT = "radar_left"
    RIGHT = "radar_right"


# The same names as Literal types, for completion and type checking of plain strings
# (a Literal cannot be built from the enum; a test keeps them in sync).
CameraNameStr = Literal[
    "camera_base_front_center",
    "camera_base_front_left_rect",
    "camera_base_front_right_rect",
    "camera_ring_front",
    "camera_ring_front_left",
    "camera_ring_front_right",
    "camera_ring_rear",
    "camera_ring_rear_left",
    "camera_ring_rear_right",
]
LidarNameStr = Literal[
    "lidar_top",
    "lidar_front",
    "lidar_left",
    "lidar_right",
    "lidar_rear",
    "lidar_corner_left",
    "lidar_corner_right",
]
RadarNameStr = Literal["radar_front", "radar_left", "radar_right"]

#: What a parameter that takes a sensor name accepts.
CameraNameLike = Union[CameraName, CameraNameStr]
LidarNameLike = Union[LidarName, LidarNameStr]
RadarNameLike = Union[RadarName, RadarNameStr]
SensorNameLike = Union[CameraNameLike, LidarNameLike, RadarNameLike]

CAMERA_BASE_NAMES: Final[tuple[str, ...]] = tuple(m.value for m in CameraName if m.name.startswith("BASE_"))
CAMERA_RING_NAMES: Final[tuple[str, ...]] = tuple(m.value for m in CameraName if m.name.startswith("RING_"))
CAMERA_NAMES: Final[tuple[str, ...]] = CAMERA_BASE_NAMES + CAMERA_RING_NAMES
LIDAR_NAMES: Final[tuple[str, ...]] = tuple(m.value for m in LidarName)
RADAR_NAMES: Final[tuple[str, ...]] = tuple(m.value for m in RadarName)
ALL_SENSOR_NAMES: Final[tuple[str, ...]] = CAMERA_NAMES + LIDAR_NAMES + RADAR_NAMES


def sensor_name(name: str) -> str:
    """*name* as a plain string (the value of a name enum member, a string unchanged).

    Raises:
        TypeError: For anything else than a string.
    """
    if isinstance(name, _SensorName):
        return name.value
    if isinstance(name, str):
        return name
    raise TypeError(f"A sensor name must be a string or a kitscenes name enum member, got {type(name).__name__}")


def sensor_names(names: Union[str, Iterable[str]]) -> tuple[str, ...]:
    """Sensor names as a tuple of plain strings; one name may be given alone."""
    return (sensor_name(names),) if isinstance(names, str) else tuple(sensor_name(n) for n in names)

# ---------------------------------------------------------------------------
# The vehicle
# ---------------------------------------------------------------------------

#: Lidar returns inside this box around the base origin (half extents, metres) are the car's own body.
EGO_BOX: Final[tuple[float, float, float]] = (3.6, 1.5, 2.6)

#: Height of the base origin above the road (m) per lidar generation (:attr:`kitscenes.splits.SceneInfo.lidar`):
#: where the road frame (``scene.T_road_base``) sits. How it was measured: docs/conventions.md, "The road frame".
ROAD_HEIGHT_M: Final[dict[str, float]] = {"hesai": 2.055, "velodyne": 2.040}

#: :data:`ROAD_HEIGHT_M` for a scene of unknown lidar generation (one the scene table does not list).
DEFAULT_ROAD_HEIGHT_M: Final[float] = 2.05

# ---------------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------------

KITSCENES_ROOT_ENV_VAR: Final[str] = "KITSCENES_ROOT"

#: Top-level folders of the city maps: the Lanelet2 HD maps and the OpenStreetMap SD maps.
HD_MAPS_DIR: Final[str] = "lanelet2_hd_maps"
SD_MAPS_DIR: Final[str] = "openstreetmap_sd_maps"
