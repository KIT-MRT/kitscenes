"""Projecting map points into a camera image, in memory."""

from __future__ import annotations

import numpy as np

from kitscenes.rig import Camera, invert_transform, transform_points
from kitscenes.schema import EgoPose


def project_to_camera(
    xyz_map: np.ndarray, pose: EgoPose, camera: Camera, *, min_depth: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map-frame points (N, 3) -> pixels of *camera* with the vehicle at *pose*.

    Use the frame's pose (``frame.pose``), the time the image belongs to.
    Points without a known height (z NaN) come out invisible.

    Returns:
        ``(uv, depth, visible)`` as :meth:`kitscenes.rig.Camera.project`.
    """
    xyz_base = transform_points(invert_transform(pose.T_world_base), np.asarray(xyz_map, dtype=np.float64))
    uv, depth, visible = camera.project(xyz_base, min_depth=min_depth)
    visible &= np.isfinite(depth)
    return uv, depth, visible
