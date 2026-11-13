"""Polyline helpers: arc length, interpolation, resampling, clipping to a region (no shapely)."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from kitscenes.map.roi import Box, Circle, Roi


def arc_length(p: np.ndarray) -> np.ndarray:
    """(N,) cumulative length along a polyline (N, D)."""
    if len(p) == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))])


def point_at(p: np.ndarray, s: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Points at arc lengths *at* along polyline *p* with cumulative lengths *s* (clamped to the ends)."""
    at = np.clip(np.asarray(at, float), s[0], s[-1])
    return np.column_stack([np.interp(at, s, p[:, k]) for k in range(p.shape[1])])


def heading_at(p: np.ndarray, s: np.ndarray, at: float, span: float = 0.0) -> float:
    """Direction (radians) of polyline *p* at arc length *at*; over ``[at - span, at + span]`` if *span* > 0."""
    if span > 0:
        a, b = point_at(p, s, [at - span, at + span])
        if np.hypot(*(b - a)[:2]) > 1e-6:
            return float(math.atan2(b[1] - a[1], b[0] - a[0]))
    i = int(np.clip(np.searchsorted(s, at, side="right") - 1, 0, len(p) - 2))
    while i < len(p) - 2 and s[i + 1] - s[i] < 1e-9:
        i += 1
    d = p[i + 1] - p[i]
    return float(math.atan2(d[1], d[0]))


def resample(p: np.ndarray, n: int) -> np.ndarray:
    """*n* points evenly spaced along polyline *p* (both ends included)."""
    if len(p) == 0:
        return np.zeros((0, p.shape[1] if p.ndim == 2 else 3))
    s = arc_length(p)
    if s[-1] <= 0 or n == 1:
        return np.repeat(p[:1], n, axis=0)
    return point_at(p, s, np.linspace(0.0, s[-1], n))


def densify(p: np.ndarray, step: float) -> np.ndarray:
    """The polyline with points at most *step* apart (original vertices kept)."""
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(math.ceil(np.hypot(*(b - a)[:2]) / step)))
        t = (np.arange(1, n + 1) / n)[:, None]
        out.append(a + t * (b - a))
    return np.concatenate(out)


def contains(roi: Roi, points: np.ndarray) -> np.ndarray:
    """(N,) whether each point (N, 2+) lies inside the region (border included)."""
    q = np.asarray(points, float)[:, :2]
    if isinstance(roi, Circle):
        return np.hypot(*(q - np.asarray(roi.center)).T) <= roi.radius
    loc = roi._local(q)
    return (loc[:, 0] >= -roi.behind) & (loc[:, 0] <= roi.ahead) & (loc[:, 1] >= -roi.right) & (loc[:, 1] <= roi.left)


def bounding_box(roi: Roi) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned map-frame bounds ``(lo, hi)`` of a region."""
    if isinstance(roi, Circle):
        c = np.asarray(roi.center)
        return c - roi.radius, c + roi.radius
    c, s = math.cos(roi.yaw), math.sin(roi.yaw)
    corners = np.array([[roi.ahead, roi.left], [roi.ahead, -roi.right], [-roi.behind, roi.left], [-roi.behind, -roi.right]])
    world = corners @ np.array([[c, s], [-s, c]]) + np.asarray(roi.center)
    return world.min(axis=0), world.max(axis=0)


def _segment_interval(a: np.ndarray, b: np.ndarray, roi: Roi) -> Optional[tuple[float, float]]:
    """The parameter interval ``[t0, t1]`` of segment a->b (2D) inside the region, None if it misses it."""
    if isinstance(roi, Circle):
        c = np.asarray(roi.center)
        d = b - a
        f = a - c
        A = float(d @ d)
        B = 2 * float(f @ d)
        C = float(f @ f) - roi.radius ** 2
        if A < 1e-18:
            return (0.0, 1.0) if C <= 0 else None
        disc = B * B - 4 * A * C
        if disc < 0:
            return None
        r = math.sqrt(disc)
        t0, t1 = max(0.0, (-B - r) / (2 * A)), min(1.0, (-B + r) / (2 * A))
        return (t0, t1) if t0 <= t1 else None
    la, lb = roi._local(np.array([a, b]))
    lo = np.array([-roi.behind, -roi.right])
    hi = np.array([roi.ahead, roi.left])
    d = lb - la
    t0, t1 = 0.0, 1.0
    for axis in range(2):
        for p, q in ((-d[axis], la[axis] - lo[axis]), (d[axis], hi[axis] - la[axis])):
            if p == 0:
                if q < 0:
                    return None
                continue
            r = q / p
            if p < 0:
                t0 = max(t0, r)
            else:
                t1 = min(t1, r)
            if t0 > t1:
                return None
    return t0, t1


def clip(points: np.ndarray, roi: Roi) -> list[np.ndarray]:
    """The pieces of polyline *points* (N, D; x, y used) inside *roi*, in order along it.

    Segments that only touch the border (zero length inside) give no piece; a
    single point gives itself if it is inside.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) == 1:
        return [p.copy()] if contains(roi, p)[0] else []
    pieces: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for a, b in zip(p[:-1], p[1:]):
        iv = _segment_interval(a[:2], b[:2], roi)
        if iv is None or iv[1] - iv[0] <= 1e-12:
            if current:
                pieces.append(np.array(current))
                current = []
            continue
        t0, t1 = iv
        start = a + t0 * (b - a)
        end = a + t1 * (b - a)
        if current and t0 > 1e-12:
            pieces.append(np.array(current))
            current = []
        if not current:
            current = [start]
        current.append(end)
        if t1 < 1 - 1e-12:
            pieces.append(np.array(current))
            current = []
    if current:
        pieces.append(np.array(current))
    return [q for q in pieces if len(q) >= 2 and arc_length(q)[-1] > 1e-9]


def roi_contains_box(roi: Roi, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Which axis-aligned boxes (lo, hi: (N, 2)) overlap the region's bounding box."""
    rlo, rhi = bounding_box(roi)
    return (lo[:, 0] <= rhi[0]) & (hi[:, 0] >= rlo[0]) & (lo[:, 1] <= rhi[1]) & (hi[:, 1] >= rlo[1])


__all__ = ["arc_length", "point_at", "heading_at", "resample", "densify", "clip", "contains", "bounding_box",
           "Box", "Circle"]
