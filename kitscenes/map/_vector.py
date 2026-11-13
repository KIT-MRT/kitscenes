"""Geometry behind the map-label options (numpy only): crosswalk polygons, the height band, piece order.

- Crossings: crosswalk lanelets that share a boundary edge are one
  crossing; their polygons are merged by cancelling the shared edges.
- Box clipping of a polygon (Sutherland-Hodgman, exact for the convex
  crosswalk outlines; z is interpolated along the cut edges) and of a
  polyline (Liang-Barsky, into the pieces inside the box).
- The height band: polylines cut where they leave a z range, like the box
  cuts them at its border.
- Arc-length position of a cut piece on its uncut line, for piece indices.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Iterable, Optional, Sequence

import numpy as np

#: Crosswalk polygons smaller than this after clipping (m^2) are dropped, as in the AV2 map extractors of
#: StreamMapNet and MapTracker.
MIN_CROSSWALK_AREA = 1.0
#: Pieces shorter than this (m) are dropped, as the converter does when it cuts a line at the box.
MIN_PIECE_LENGTH = 0.1


@dataclasses.dataclass(frozen=True)
class Crossing:
    """One pedestrian crossing: the merged outline of crosswalk lanelets that share boundary edges.

    Attributes:
        lane_ids: The crosswalk lanelets (sorted).
        ring: (N, 3) map-frame outline, counter-clockwise, not closed (first != last).
        zebra: Whether one of its boundaries is a ``zebra_marking``.
        lo, hi: x-y bounding box of the ring.
    """

    lane_ids: tuple[int, ...]
    ring: np.ndarray
    zebra: bool
    lo: np.ndarray
    hi: np.ndarray


def signed_area(xy: np.ndarray) -> float:
    """Shoelace area of a ring (N, 2+), positive counter-clockwise."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _ring_keys(ring: np.ndarray) -> list[tuple[int, int]]:
    """Vertex keys on a millimetre grid: shared map nodes have the same coordinates, hence the same key."""
    return [tuple(k) for k in np.round(ring[:, :2] * 1000.0).astype(np.int64).tolist()]


def _clean_ring(points: np.ndarray) -> np.ndarray:
    """*points* as a ring: consecutive duplicates and a closing duplicate removed, counter-clockwise."""
    pts = np.asarray(points, dtype=np.float64)
    keys = _ring_keys(pts)
    keep = [i for i in range(len(pts)) if i == 0 or keys[i] != keys[i - 1]]
    pts = pts[keep]
    if len(pts) > 1 and _ring_keys(pts[:1]) == _ring_keys(pts[-1:]):
        pts = pts[:-1]
    if len(pts) >= 3 and signed_area(pts) < 0:
        pts = pts[::-1]
    return pts


def crossings(lanes: Iterable, zebra_lines: set[int]) -> list[Crossing]:
    """The crossings of a map from its crosswalk lanes (:class:`kitscenes.map.Lane`).

    Lanes whose outlines share an edge (two consecutive vertices, i.e. the
    same two map nodes) form one crossing. Where the merged outline is not
    one simple ring (lanes that touch in a more complex way), the lanes stay
    separate crossings.
    """
    lanes = sorted(lanes, key=lambda lane: lane.id)
    rings = {lane.id: _clean_ring(lane.polygon) for lane in lanes}
    rings = {i: r for i, r in rings.items() if len(r) >= 3}
    zebra = {lane.id: lane.left in zebra_lines or lane.right in zebra_lines for lane in lanes if lane.id in rings}

    edges_of: dict[int, set] = {}
    for i, ring in rings.items():
        keys = _ring_keys(ring)
        edges_of[i] = {frozenset((keys[k], keys[(k + 1) % len(keys)])) for k in range(len(keys))}
    parent = {i: i for i in rings}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_edge: dict[frozenset, int] = {}
    for i in rings:
        for edge in edges_of[i]:
            if edge in by_edge:
                a, b = find(by_edge[edge]), find(i)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                by_edge[edge] = i
    groups: dict[int, list[int]] = {}
    for i in rings:
        groups.setdefault(find(i), []).append(i)

    out = []
    for members in groups.values():
        merged = _merge([rings[i] for i in members]) if len(members) > 1 else rings[members[0]]
        parts = [(tuple(members), merged)] if merged is not None else [((i,), rings[i]) for i in members]
        for ids, ring in parts:
            out.append(Crossing(tuple(sorted(ids)), ring, any(zebra[i] for i in ids),
                                ring[:, :2].min(axis=0), ring[:, :2].max(axis=0)))
    return sorted(out, key=lambda c: c.lane_ids)


def _merge(rings: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """The union of counter-clockwise rings that share edges, or None if it is not one simple ring."""
    point: dict[tuple, np.ndarray] = {}
    directed: list[tuple[tuple, tuple]] = []
    for ring in rings:
        keys = _ring_keys(ring)
        for k, key in enumerate(keys):
            point.setdefault(key, ring[k])
            directed.append((key, keys[(k + 1) % len(keys)]))
    present = set(directed)
    if len(present) != len(directed):  # an edge twice in the same direction: overlapping lanes
        return None
    remaining = [e for e in directed if (e[1], e[0]) not in present]
    nxt: dict[tuple, tuple] = {}
    for a, b in remaining:
        if a in nxt:  # two ways on from one vertex: not one simple ring
            return None
        nxt[a] = b
    start = min(nxt)
    ring, key = [start], nxt[start]
    while key != start:
        if key not in nxt or len(ring) > len(nxt):
            return None
        ring.append(key)
        key = nxt[key]
    if len(ring) != len(nxt):  # a second ring (a hole, or a separate part)
        return None
    out = np.array([point[k] for k in ring])
    return out if signed_area(out) > 0 else None


def clip_ring(ring: np.ndarray, half_x: float, half_y: float) -> np.ndarray:
    """A ring (N, 3+) in a box frame clipped to ``|x| <= half_x, |y| <= half_y`` (Sutherland-Hodgman).

    All coordinates of a new vertex are interpolated along the edge it cuts,
    so z follows the outline. Exact for convex rings; a ring the box cuts into
    several parts comes out as one with zero-width bridges along the border.
    """
    pts = np.asarray(ring, dtype=np.float64)
    for axis, sign, limit in ((0, 1.0, half_x), (0, -1.0, half_x), (1, 1.0, half_y), (1, -1.0, half_y)):
        if len(pts) == 0:
            break
        d = limit - sign * pts[:, axis]  # >= 0 inside
        nxt_pts, nxt_d = np.roll(pts, -1, axis=0), np.roll(d, -1)
        out = []
        for p, q, dp, dq in zip(pts, nxt_pts, d, nxt_d):
            if dp >= 0:
                out.append(p)
            if (dp >= 0) != (dq >= 0):
                out.append(p + (q - p) * (dp / (dp - dq)))
        pts = np.array(out).reshape(-1, ring.shape[1])
    return pts


def clip_polyline(line: np.ndarray, half_x: float, half_y: float) -> list[np.ndarray]:
    """The parts of a polyline (N, 2+) in a box frame inside ``|x| <= half_x, |y| <= half_y``, cut at the border.

    Other coordinates are interpolated along the cut segments. Parts shorter
    than :data:`MIN_PIECE_LENGTH` are dropped.
    """
    pts = np.asarray(line, dtype=np.float64)
    if len(pts) < 2:
        return []
    a, d = pts[:-1], np.diff(pts, axis=0)
    t0, t1 = np.zeros(len(d)), np.ones(len(d))
    hit = np.ones(len(d), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-d[:, 0], a[:, 0] + half_x), (d[:, 0], half_x - a[:, 0]),
                     (-d[:, 1], a[:, 1] + half_y), (d[:, 1], half_y - a[:, 1])):
            r = q / p
            hit &= ~((p == 0) & (q < 0))
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
    hit &= t0 <= t1
    pieces: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for k in np.flatnonzero(hit):
        start, end = a[k] + t0[k] * d[k], a[k] + t1[k] * d[k]
        if not current:  # entering the box (an open piece always continues: it ended inside)
            current = [start]
        current.append(end)
        if t1[k] < 1:
            pieces.append(np.array(current))
            current = []
    if current:
        pieces.append(np.array(current))
    return [p for p in pieces if length(p) >= MIN_PIECE_LENGTH]


def close_ring(ring: np.ndarray) -> np.ndarray:
    """A ring as a closed polyline (first == last, exactly), starting at its smallest (x, y) vertex."""
    start = int(np.lexsort((ring[:, 1], ring[:, 0]))[0])
    ring = np.roll(ring, -start, axis=0)
    return np.vstack([ring, ring[:1]])


def cut_to_band(line: np.ndarray, z: np.ndarray, lo: float, hi: float) -> list[np.ndarray]:
    """The parts of *line* (N, D) whose height *z* (N,) is within ``[lo, hi]``, cut at the band's limits.

    Parts shorter than :data:`MIN_PIECE_LENGTH` are dropped.
    """
    inside = (z >= lo) & (z <= hi)
    if inside.all():
        return [line]
    pieces: list[np.ndarray] = []
    current: list[np.ndarray] = []
    for k in range(len(line)):
        if k > 0 and z[k] != z[k - 1]:
            za, zb = z[k - 1], z[k]
            # the band's limits this segment crosses, in the order it crosses them
            for t in sorted(t for t in ((lo - za) / (zb - za), (hi - za) / (zb - za)) if 0 < t < 1):
                p = line[k - 1] + t * (line[k] - line[k - 1])
                if current:  # leaving the band
                    current.append(p)
                    pieces.append(np.array(current))
                    current = []
                else:  # entering it
                    current = [p]
        if inside[k]:
            current.append(line[k])
        elif current:
            pieces.append(np.array(current))
            current = []
    if current:
        pieces.append(np.array(current))
    return [p for p in pieces if len(p) >= 2 and length(p) >= MIN_PIECE_LENGTH]


def length(line: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())


def arc_position(line: np.ndarray, point: np.ndarray) -> float:
    """Arc length (x-y) along *line* of the point on it nearest to *point*."""
    xy, q = np.asarray(line, dtype=np.float64)[:, :2], np.asarray(point, dtype=np.float64)[:2]
    if len(xy) == 1:
        return 0.0
    a, b = xy[:-1], xy[1:]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    dist = np.linalg.norm(a + t[:, None] * ab - q, axis=1)
    k = int(np.argmin(dist))
    seg = np.linalg.norm(ab, axis=1)
    return float(seg[:k].sum() + t[k] * seg[k])


def instance_key(cls: str, ids: Sequence[int], piece: int) -> int:
    """A stable signed 64-bit hash of an element key ``(class, ids, piece)`` (the same in every process)."""
    text = f"{cls}|{','.join(str(int(i)) for i in ids)}|{int(piece)}".encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "little", signed=True)
