"""Regions of interest: a circle or an oriented box in the map frame, and a rectangle around the vehicle.

- :class:`Region` is a rectangle relative to the vehicle (metres ahead,
  behind, left, right): what :class:`kitscenes.map.MLConverter` labels, and,
  placed at a pose with :meth:`Region.around`, a :class:`Box` for map queries.
- :class:`Box` and :class:`Circle` are in the map frame. An element is in
  one if any part of its polyline (or polygon outline) is: segments that only
  cross it count, not just their vertices.
"""

from __future__ import annotations

import dataclasses
from typing import Union

import numpy as np

from kitscenes.schema import EgoPose


@dataclasses.dataclass(frozen=True)
class Circle:
    """All points within *radius* metres of *center* (x, y; a z is ignored)."""

    center: tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        c = np.asarray(self.center, dtype=np.float64).reshape(-1)
        if len(c) not in (2, 3):
            raise ValueError(f"center must be (x, y) or (x, y, z), got {self.center!r}")
        object.__setattr__(self, "center", (float(c[0]), float(c[1])))

    @classmethod
    def around(cls, pose: EgoPose, radius: float) -> "Circle":
        """A circle of *radius* metres around the vehicle at *pose*."""
        return cls((float(pose.xyz[0]), float(pose.xyz[1])), float(radius))

    def hits(self, points: np.ndarray) -> bool:
        """Whether a polyline (N, 2+) touches the circle."""
        return bool(_segment_distances(np.asarray(points)[:, :2], np.asarray(self.center, float)).min() <= self.radius)


@dataclasses.dataclass(frozen=True)
class Box:
    """An oriented box around *center* (x, y), rotated by *yaw*, in the map frame.

    Extents are measured along the box axes: ``ahead``/``behind`` along x,
    ``left``/``right`` along y. Around the vehicle: ``Region(...).around(pose)``.
    """

    center: tuple[float, float]
    yaw: float
    ahead: float
    behind: float
    left: float
    right: float

    def _local(self, points: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.asarray(points)[:, :2] - np.asarray(self.center, float)
        return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]])

    def hits(self, points: np.ndarray) -> bool:
        """Whether a polyline (N, 2+) touches the box."""
        p = self._local(points)
        lo = np.array([-self.behind, -self.right])
        hi = np.array([self.ahead, self.left])
        if len(p) == 1:
            return bool(np.all((p[0] >= lo) & (p[0] <= hi)))
        return bool(np.any(_segments_hit_aabb(p[:-1], p[1:], lo, hi)))


@dataclasses.dataclass(frozen=True)
class Region:
    """A rectangle around the vehicle, in metres along its axes (ego frame).

    What :class:`kitscenes.map.MLConverter` labels (``region=``), and the box
    of map queries around a pose (:meth:`around`)::

        region = Region(ahead=60, behind=20, left=20, right=20)
        lanes = smap.lanes(region.around(frame.pose))
        labels = MLConverter.for_scene(scene, region=region)(frame.pose)

    The converter labels a rectangle centred on the vehicle; an asymmetric
    region is labelled around its own centre and moved back into the ego
    frame, so the labels are the same as for a centred one.
    """

    ahead: float = 30.0
    behind: float = 30.0
    left: float = 15.0
    right: float = 15.0

    def __post_init__(self) -> None:
        if self.ahead + self.behind <= 0 or self.left + self.right <= 0:
            raise ValueError("Region must have a positive length and width")

    def around(self, pose: EgoPose) -> Box:
        """This rectangle around the vehicle at *pose*, aligned with its heading: a map-frame :class:`Box`."""
        return Box((float(pose.xyz[0]), float(pose.xyz[1])), pose.yaw,
                   self.ahead, self.behind, self.left, self.right)

    @property
    def half_extents(self) -> tuple[float, float]:
        """``(longitudinal, lateral)`` half extents, as the converter takes them."""
        return (self.ahead + self.behind) / 2, (self.left + self.right) / 2

    @property
    def centre(self) -> tuple[float, float]:
        """Centre of the region in the ego frame ``(x, y)``."""
        return (self.ahead - self.behind) / 2, (self.left - self.right) / 2


Roi = Union[Circle, Box]


def hits_area(roi: Roi, outline: np.ndarray) -> bool:
    """Whether a polygon (its outline, (N, 2+)) overlaps the region: the outline
    touches it, or the polygon contains the region's centre."""
    closed = np.concatenate([outline, outline[:1]])
    return roi.hits(closed) or point_in_polygon(np.asarray(roi.center, float), np.asarray(outline)[:, :2])


def point_in_polygon(q: np.ndarray, poly: np.ndarray) -> bool:
    """Even-odd rule for one point."""
    x, y = poly[:, 0], poly[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    crosses = (y > q[1]) != (yn > q[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = x + (q[1] - y) * (xn - x) / (yn - y)
    return bool(np.count_nonzero(crosses & (q[0] < x_at)) % 2)


def _segment_distances(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from point *q* to each segment of polyline *p* (or to its single point)."""
    if len(p) == 1:
        return np.linalg.norm(p - q, axis=1)
    a, b = p[:-1], p[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * ab - q, axis=1)


def _segments_hit_aabb(a: np.ndarray, b: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Liang-Barsky clipping: which segments a->b intersect the box [lo, hi]."""
    d = b - a
    t0 = np.zeros(len(a))
    t1 = np.ones(len(a))
    ok = np.ones(len(a), dtype=bool)
    for axis in range(2):
        for p, q in ((-d[:, axis], a[:, axis] - lo[axis]), (d[:, axis], hi[axis] - a[:, axis])):
            parallel = p == 0
            ok &= ~(parallel & (q < 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(parallel, 0.0, q / np.where(parallel, 1.0, p))
            entering = ~parallel & (p < 0)
            leaving = ~parallel & (p > 0)
            t0 = np.where(entering, np.maximum(t0, r), t0)
            t1 = np.where(leaving, np.minimum(t1, r), t1)
    return ok & (t0 <= t1)
