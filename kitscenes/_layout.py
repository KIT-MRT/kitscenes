"""File layout of a released scene: every member name the API reads.

A scene is the folder ``<root>/data/<split>/<scene_id>/`` (splits:
:mod:`kitscenes.splits`). Member names are relative to the scene folder and
always use ``/``, so they work the same for a folder and an archive.
"""

from __future__ import annotations

from typing import Final

from kitscenes.constants import CAMERA_NAMES, LIDAR_NAMES, RADAR_NAMES

DATA_SUBDIR: Final[str] = "data"

REFERENCE_TIMESTAMPS: Final[str] = "timestamp.reference.txt"
POSES: Final[str] = "poses.txt"
CALIBRATION: Final[str] = "calibration/calib.json"
IMAGE_CORRECTIONS: Final[str] = "calibration/image_corrections.json"
MAP_OSM: Final[str] = "maps/map.osm"
MAP_ORIGIN: Final[str] = "maps/origin.json"
GROUND_SEGMENTATION: Final[str] = "processed/ground_segmentation"
INSTANCE_MASKS: Final[str] = "processed/seamseg_instances"

FRAME_INDEX_WIDTH: Final[int] = 10
CAMERA_EXTENSION: Final[str] = ".jpg"
POINT_CLOUD_EXTENSION: Final[str] = ".parquet"


def sensor_kind(sensor: str) -> str:
    """``"camera"``, ``"lidar"`` or ``"radar"``.

    Raises:
        KeyError: For a name that is no KITScenes sensor.
    """
    if sensor in CAMERA_NAMES:
        return "camera"
    if sensor in LIDAR_NAMES:
        return "lidar"
    if sensor in RADAR_NAMES:
        return "radar"
    raise KeyError(
        f"Unknown sensor {sensor!r}. Valid names: {list(CAMERA_NAMES + LIDAR_NAMES + RADAR_NAMES)}"
    )


def frame_stem(index: int) -> str:
    """File stem of a frame index: ``0000000042``."""
    return f"{index:0{FRAME_INDEX_WIDTH}d}"


def sensor_member(sensor: str, index: int) -> str:
    """Member name of one sensor frame, e.g. ``lidar_top/0000000042.parquet``."""
    ext = CAMERA_EXTENSION if sensor_kind(sensor) == "camera" else POINT_CLOUD_EXTENSION
    return f"{sensor}/{frame_stem(index)}{ext}"


def ground_label_member(lidar: str, index: int) -> str:
    """Member name of one ground-segmentation label file."""
    return f"{GROUND_SEGMENTATION}/{lidar}/{frame_stem(index)}{POINT_CLOUD_EXTENSION}"


def instance_member(camera: str, index: int) -> str:
    """Member name of one camera's instance-mask file."""
    return f"{INSTANCE_MASKS}/{camera}/{frame_stem(index)}.json"


def frame_index_of(member_name: str) -> int | None:
    """Frame index of a sensor file name (``0000000042.jpg`` -> 42), else None."""
    stem, _, ext = member_name.rpartition("/")[2].partition(".")
    return int(stem) if stem.isdigit() and ext else None
