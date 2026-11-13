"""Colour, class and instance per lidar point, from the ring cameras of the same frame.

Points are projected into each camera (:func:`kitscenes.projection.project_points`)
and read out of the image, or out of the seamseg label images
(:mod:`kitscenes.instances`; the front and rear ring cameras only). The first
camera in *cameras* that sees a point wins; the ring cameras barely overlap.

A camera colours and labels every point that projects into its image (the
projection and pixel rules are those of :mod:`kitscenes.projection`); it
does not check whether it can see the point. The lidars sit elsewhere than
the cameras, so a point hidden from a camera by something nearer takes the
colour and class of what is in front of it: a wall behind a passing car, or
the road behind a parked one, comes out in the car's colour and as "Car".

Example::

    cloud = frame.fused_lidar()
    rgb, camera = paint(frame, cloud.xyz)
    classes, instance, camera = label(frame, cloud.xyz)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from kitscenes.constants import CameraNameLike
from kitscenes.instances import UNLABELLED
from kitscenes.projection import pixel_indices, project_points

if TYPE_CHECKING:
    from kitscenes.frame import Frame
    from kitscenes.rig import Camera

#: The ring cameras, in the order they are tried.
RING_CAMERAS: tuple[str, ...] = (
    "camera_ring_front", "camera_ring_front_left", "camera_ring_front_right",
    "camera_ring_rear_right", "camera_ring_rear", "camera_ring_rear_left",
)
#: The cameras with seamseg masks.
SEGMENTED_CAMERAS: tuple[str, ...] = ("camera_ring_front", "camera_ring_rear")
#: Colour of points no camera saw.
UNSEEN = (90, 90, 90)


def _project(camera: "Camera", xyz: np.ndarray):
    """``(uv, seen)`` of the points in *camera*, in float32 (millions of points)."""
    uv, _, seen = project_points(camera, np.asarray(xyz, np.float32))
    return uv, seen


def paint(
    frame: "Frame",
    xyz: Optional[np.ndarray] = None,
    *,
    cameras: Sequence[CameraNameLike] = RING_CAMERAS,
    reduce: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """The camera colour of every point.

    Args:
        frame: The frame; its images are read.
        xyz: (N, 3) points in the frame's base frame (default: ``frame.fused_lidar().xyz``).
        cameras: Cameras to try, in order; missing images are skipped.
        reduce: Decode the images at 1/reduce size (a point needs one pixel;
            2 is about four times faster than full size).

    Returns:
        ``(rgb, camera)``: (N, 3) uint8 colours (:data:`UNSEEN` grey where no
        camera saw the point) and (N,) int8 index into *cameras* (-1: unseen).
        A point behind something nearer in a camera's view gets the colour
        of that thing (see the module notes).
    """
    xyz = frame.fused_lidar().xyz if xyz is None else np.asarray(xyz)
    rgb = np.empty((len(xyz), 3), np.uint8)
    rgb[:] = UNSEEN
    which = np.full(len(xyz), -1, np.int8)
    for index, name in enumerate(cameras):
        camera = frame.scene.files.rig_original.cameras.get(name)  # release size: same result for downscaled images
        if camera is None or not (which < 0).any():
            continue
        uv, seen = _project(camera, xyz)
        todo = np.flatnonzero(seen & (which < 0))  # only points no earlier camera coloured
        if not len(todo):
            continue
        image = frame.image(name, reduce=reduce, missing="none")
        if image is None:
            continue
        # JPEG draft decoding scales by powers of two only roughly; pixel_indices uses the real ratio.
        r, c = pixel_indices(uv[todo], image.shape[:2], camera)
        rgb[todo] = image[r, c]
        which[todo] = index
    return rgb, which


def label(
    frame: "Frame",
    xyz: Optional[np.ndarray] = None,
    *,
    cameras: Sequence[CameraNameLike] = SEGMENTED_CAMERAS,
    min_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The Mapillary Vistas class and the instance of every point, from the seamseg masks.

    Args:
        frame, xyz: As for :func:`paint`.
        cameras: Cameras to try, in order; cameras without masks are skipped.
        min_score: Ignore instances below this objectness.

    Returns:
        ``(classes, instances, camera)``: (N,) uint8 Vistas trainIds
        (:data:`kitscenes.instances.UNLABELLED` where no mask covers the
        point), (N,) uint16 instance numbers (``i + 1`` for instance ``i`` of
        that camera's masks, 0 for none; seamseg does not track, so they are
        per frame and camera) and (N,) int8 index into *cameras* of the mask
        the label came from (-1: none). A camera that sees a point on an
        unlabelled pixel leaves it to the next camera. A point behind
        something nearer in a camera's view gets the class of that thing
        (see the module notes). Model output, not annotations.
    """
    xyz = frame.fused_lidar().xyz if xyz is None else np.asarray(xyz)
    classes = np.full(len(xyz), UNLABELLED, np.uint8)
    instances = np.zeros(len(xyz), np.uint16)
    which = np.full(len(xyz), -1, np.int8)
    for index, name in enumerate(cameras):
        camera = frame.scene.files.rig_original.cameras.get(name)  # release size: same result for downscaled images
        if camera is None or not (classes == UNLABELLED).any():
            continue
        masks = frame.instances(name, missing="none")
        if masks is None:
            continue
        uv, seen = _project(camera, xyz)
        todo = np.flatnonzero(seen & (classes == UNLABELLED))
        if not len(todo):
            continue
        width, height = masks.image_size
        rows, cols = pixel_indices(uv[todo], (height, width), camera)
        found, instance = masks.lookup(rows, cols, min_score)
        classes[todo] = found
        instances[todo] = instance
        which[todo[found != UNLABELLED]] = index
    return classes, instances, which
