"""Frame poses of a scene in EPSG:25832, the CRS of the aerial imagery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from kitscenes.aerial._geo import heading_from_rotation, scene_xy_to_epsg25832
from kitscenes.errors import MapUnavailable

if TYPE_CHECKING:
    from kitscenes.map import MapOrigin
    from kitscenes.scene import Scene

#: Crop folder of scenes whose folder is not a release split.
NO_SPLIT = "_flat"


@dataclass(frozen=True)
class ScenePoses:
    """The frames of one scene placed in EPSG:25832.

    Attributes:
        scene_id, split: The scene and its release folder (:data:`NO_SPLIT` if none).
        lat, lon: Its map origin (its city's for the test splits, :func:`placement_origin`).
        indices: (M,) frame indices with a pose.
        xy: (M, 2) EPSG:25832 easting, northing of the base frame.
        yaw: (M,) heading in radians.
    """

    scene_id: str
    split: str
    lat: float
    lon: float
    indices: np.ndarray
    xy: np.ndarray
    yaw: np.ndarray

    def position(self, index: int) -> int:
        """Row of frame *index* in :attr:`indices`, or -1 if it has no pose."""
        k = int(np.searchsorted(self.indices, index))
        return k if k < len(self.indices) and self.indices[k] == index else -1


def placement_origin(scene: "Scene") -> "MapOrigin":
    """The origin of *scene*'s pose frame: its ``maps/origin.json``, or, for a scene without one whose poses are in
    the map frame (the test and map_change_test splits ship no maps), its city's origin
    (:data:`kitscenes.map.origin.CITY_ORIGINS`).

    Raises:
        MapUnavailable: Neither (the test-e2e splits: poses relative to the first pose).
    """
    from kitscenes.map.origin import CITY_ORIGINS

    try:
        return scene.origin
    except MapUnavailable:
        info = scene.info
        city = info.city if info is not None else None
        if scene.pose_frame == "local_utm" and city in CITY_ORIGINS:
            return CITY_ORIGINS[city]
        raise


def scene_poses(scene: "Scene") -> ScenePoses:
    """Pose of every frame of *scene* at its frame time, in EPSG:25832.

    Frames whose time lies outside the poses (or in a gap) are left out.

    Raises:
        MapUnavailable: Without a map origin, own or of its city (:func:`placement_origin`; the test-e2e splits).
        ValueError: Without usable poses.
    """
    origin = placement_origin(scene)
    trajectory, why = scene._trajectory
    if trajectory is None:
        raise ValueError(f"Scene {scene.id} has no usable poses: {why}")
    times = scene.timestamps_ns / 1e9
    indices = np.arange(len(times))
    try:
        rotations, translations = trajectory.interpolate(times)
    except ValueError:  # some frame lies outside the poses: keep the others
        keep = []
        for i, t in enumerate(times):
            try:
                trajectory.interpolate(np.array([t]))
                keep.append(i)
            except ValueError:
                pass
        indices = np.array(keep, dtype=np.int64)
        if not len(indices):
            raise ValueError(f"Scene {scene.id}: no frame time lies within its poses.") from None
        rotations, translations = trajectory.interpolate(times[indices])
    return ScenePoses(
        scene_id=scene.id,
        split=scene.release_split or NO_SPLIT,
        lat=origin.lat,
        lon=origin.lon,
        indices=indices,
        xy=scene_xy_to_epsg25832(origin, translations),
        yaw=heading_from_rotation(rotations),
    )
