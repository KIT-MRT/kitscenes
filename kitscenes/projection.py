"""Projecting into the cameras: the one path the API builds on.

Painting lidar points (:mod:`kitscenes.painting`), lidar maps
(:mod:`kitscenes.mapping`, a torch twin of the same rules), map labels and
the reprojection images (:mod:`kitscenes.visualization`) all project with
these functions, so they agree to the pixel.

- **Points:** :meth:`kitscenes.rig.Camera.project` (pinhole, base frame in),
  through :func:`project_points`. There is no visibility test: a point
  behind something nearer lands on that thing's pixels like any other.
- **Lines:** :func:`project_lines` clips each line at the near plane *in 3D*
  before dividing by depth, so a line passing beside or behind the camera
  ends at the image border instead of jumping across it.
- **Pixels:** OpenCV intrinsics put pixel centres on integer coordinates, so
  the pixel of ``u`` is ``floor(u + 0.5)``, and in an image decoded at scale
  *s* it is ``floor((u + 0.5) * s)`` (:func:`pixel_indices`). A point is in
  the image for ``-0.5 <= u < width - 0.5``.

Map-frame points go through the pose first (``frame.pose``, the time the
images belong to): :func:`kitscenes.map.project_to_camera`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from kitscenes.rig import transform_points

if TYPE_CHECKING:
    from kitscenes.rig import Camera

#: Nearest depth a projected point may have (metres in front of the camera).
MIN_DEPTH = 0.1


def project_points(
    camera: "Camera",
    xyz_base: np.ndarray,
    *,
    min_depth: float = MIN_DEPTH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base-frame points -> ``(uv, depth, visible)``, as :meth:`Camera.project`.

    *visible*: at least *min_depth* in front of the camera and inside the
    image; points behind nearer ones count as visible too.
    """
    return camera.project(xyz_base, min_depth=min_depth)


def project_lines(
    camera: "Camera",
    lines: Sequence[np.ndarray],
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: Optional[float] = None,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Base-frame polylines -> their visible pieces in the image.

    Each line (N, 3) is clipped to ``min_depth <= depth`` (and
    ``depth <= max_depth``) in the camera frame, with the crossing points
    interpolated in 3D, then projected. The pieces are not clipped to the
    image rectangle; drawing libraries do that.

    Returns:
        Per input line, a list of ``(uv (M, 2), depth (M,))`` pieces, M >= 2.
    """
    T = camera.T_sensor_base
    out = []
    for line in lines:
        q = transform_points(T, np.asarray(line, dtype=np.float64))
        pieces = []
        for piece in _clip(q, min_depth, max_depth):
            p = piece @ camera.K.T
            pieces.append((p[:, :2] / p[:, 2:3], piece[:, 2].copy()))
        out.append(pieces)
    return out


def _clip(q: np.ndarray, near: float, far: Optional[float]) -> list[np.ndarray]:
    """The runs of polyline *q* (camera frame) with ``near <= z (<= far)``, ends interpolated."""
    if len(q) < 2 or not np.isfinite(q).all():
        return []
    pieces: list[np.ndarray] = []
    for piece in _clip_plane(q, near, keep_above=True):
        pieces += _clip_plane(piece, far, keep_above=False) if far is not None else [piece]
    return pieces


def _clip_plane(q: np.ndarray, plane: float, keep_above: bool) -> list[np.ndarray]:
    inside = q[:, 2] >= plane if keep_above else q[:, 2] <= plane
    pieces, current = [], []
    for i in range(len(q)):
        if i > 0 and inside[i] != inside[i - 1]:  # crosses the plane: add the crossing point
            a, b = q[i - 1], q[i]
            t = (plane - a[2]) / (b[2] - a[2])
            crossing = a + t * (b - a)
            current.append(crossing)
            if not inside[i]:
                pieces.append(np.array(current))
                current = []
        if inside[i]:
            current.append(q[i])
    if current:
        pieces.append(np.array(current))
    return [p for p in pieces if len(p) >= 2]


def pixel_indices(uv: np.ndarray, image_hw: tuple[int, int], camera: "Camera") -> tuple[np.ndarray, np.ndarray]:
    """``(rows, cols)`` of full-size pixel coordinates *uv* in an image of *image_hw* (any decode size), clamped."""
    h, w = image_hw
    cols = np.floor((uv[:, 0] + 0.5) * (w / camera.image_size[0])).astype(np.int64)
    rows = np.floor((uv[:, 1] + 0.5) * (h / camera.image_size[1])).astype(np.int64)
    return np.clip(rows, 0, h - 1), np.clip(cols, 0, w - 1)
