"""What the crops objects share: the crop of a frame (``crops[frame]``), and crops cut at the frame's pose."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional

from kitscenes.aerial._crop import TileCoverageError
from kitscenes.errors import AerialUnavailable, Availability

if TYPE_CHECKING:
    from kitscenes.aerial._poses import ScenePoses
    from kitscenes.frame import Frame
    from kitscenes.scene import Scene


class FrameCrops:
    """Crops by frame: ``crops[frame]`` and :meth:`at`.

    A subclass has ``cache_id``, ``_has_crop_for(scene, index)`` and ``_crop_for(scene, index)``.
    """

    #: The name of the data in :class:`~kitscenes.errors.AerialUnavailable` and ``availability()``.
    _what = "aerial"

    def at(self, frame: "Frame", *, missing: str = "raise") -> Any:
        """The crop of *frame*, as :meth:`kitscenes.Frame.aerial` returns it with these crops attached.

        Args:
            missing: ``"raise"`` or ``"none"`` (return None if this frame has no crop).

        Raises:
            AerialUnavailable: With ``state`` ``WITHHELD_BY_SPLIT`` (test-e2e
                splits: poses relative to the first pose) or ``DROPPED`` (no
                crop for this frame), unless ``missing="none"``.
        """
        if missing not in ("raise", "none"):
            raise ValueError(f"missing must be 'raise' or 'none', got {missing!r}")
        scene = frame.scene
        err = scene._aerial_unavailable(frame.index, features=self._what == "aerial_features", source=self)
        if err is None:
            try:
                return frame._cached((self._what, self.cache_id), lambda: self._crop_for(scene, frame.index))
            except AerialUnavailable as dropped:
                err = dropped
        if missing == "none":
            return None
        raise err

    def __getitem__(self, frame: "Frame") -> Any:
        """``crops[frame]``: :meth:`at` (raises :class:`~kitscenes.errors.AerialUnavailable` without a crop)."""
        return self.at(frame)


class PoseCrops(FrameCrops):
    """Crops cut on the fly at the frame's pose in EPSG:25832.

    A subclass has ``covers(xy, area_id=)``, ``crop(xy, yaw, area_id=)`` and ``_area_of(scene, poses)``.
    """

    #: Scenes whose frame poses are kept per process.
    POSE_CACHE_SCENES = 64

    #: For messages: what a frame is placed on, and what the crops are cut from.
    _placed_on = "the imagery"
    _tiles_name = "the raw tiles"

    def _init_process(self) -> None:
        self._pid = os.getpid()
        self._poses: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def scene_poses(self, scene: "Scene") -> "ScenePoses":
        """The EPSG:25832 poses of *scene*'s frames (:func:`kitscenes.aerial.scene_poses`), cached per process.

        Raises:
            MapUnavailable: Without a map origin.
            ValueError: Without usable poses.
        """
        from kitscenes.aerial._poses import scene_poses

        if self._pid != os.getpid():
            self._init_process()
        key = (scene.id, str(getattr(scene.files.storage, "root", "")))
        with self._lock:
            poses = self._poses.get(key)
            if poses is not None:
                self._poses.move_to_end(key)
                return poses
        poses = scene_poses(scene)
        with self._lock:
            self._poses[key] = poses
            while len(self._poses) > self.POSE_CACHE_SCENES:
                self._poses.popitem(last=False)
        return poses

    def _area_of(self, scene: "Scene", poses: "ScenePoses") -> Optional[str]:
        raise NotImplementedError

    def _frame_pose(self, scene: "Scene", index: int):
        try:
            poses = self.scene_poses(scene)
        except (ValueError, LookupError):  # no usable poses, no map origin
            return None, None, None
        k = poses.position(index)
        if k < 0:
            return None, None, None
        return (float(poses.xy[k, 0]), float(poses.xy[k, 1])), float(poses.yaw[k]), self._area_of(scene, poses)

    def _has_crop_for(self, scene: "Scene", index: int) -> bool:
        xy, _, area_id = self._frame_pose(scene, index)
        return xy is not None and self.covers(xy, area_id=area_id)

    def _crop_for(self, scene: "Scene", index: int) -> Any:
        xy, yaw, area_id = self._frame_pose(scene, index)
        if xy is None:
            raise AerialUnavailable(self._what, index, Availability.DROPPED,
                                    f"frame {index} of scene {scene.id} has no pose to place it on {self._placed_on}")
        try:
            return self.crop(xy, yaw, area_id=area_id)
        except TileCoverageError:
            raise AerialUnavailable(
                self._what, index, Availability.DROPPED,
                f"{self._tiles_name} do not cover the crop of frame {index} of scene {scene.id}") from None
