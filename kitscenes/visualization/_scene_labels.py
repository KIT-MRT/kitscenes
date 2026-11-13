"""Map labels of a scene, computed once and cut per frame (for the video renderer).

:class:`SceneLabels` asks the ML converter once per scene (scene chaining), keeps the lines in the map
frame and cuts them to the region at each pose with numpy, which takes a fraction of the converter's time
per frame. The result matches ``MLConverter`` with ``chaining="scene"`` except for the exact cut point at
the region's border. The lines keep the map's own points (no resampling).
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from kitscenes.map import MapLabels, Region
    from kitscenes.scene import Scene
    from kitscenes.schema import EgoPose


def _liang_barsky(ax, ay, dx, dy, hx, hy, t0, t1, hit):
    """Narrow ``[t0, t1]`` of the segments ``a + t d`` to ``|x| <= hx, |y| <= hy`` (``hit``: anything left)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-dx, ax + hx), (dx, hx - ax), (-dy, ay + hy), (dy, hy - ay)):
            r = q / p
            hit = hit & ~((p == 0) & (q < 0))
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
    return t0, t1, hit & (t0 <= t1)


@dataclasses.dataclass
class _Polylines:
    """Polylines of a scene in the map frame, flattened: vertices, offsets, class and id per line."""

    vertices: np.ndarray  # (N, 3) float64, map frame
    starts: np.ndarray  # (L + 1,) int64: line k is vertices[starts[k]:starts[k + 1]]
    classes: np.ndarray  # (L,) int: index into the class names
    ids: list  # per line: map ids (tuple for lines, int for traffic elements)
    seg_a: np.ndarray = None  # (S,) index of each segment's first vertex (segments never span two lines)
    seg_line: np.ndarray = None  # (S,) line of each segment

    def __post_init__(self) -> None:
        counts = np.diff(self.starts)
        segs = np.maximum(counts - 1, 0)
        self.seg_line = np.repeat(np.arange(len(counts)), segs)
        first = np.repeat(self.starts[:-1], segs)
        within = np.arange(len(self.seg_line)) - np.repeat(np.cumsum(segs) - segs, segs)
        self.seg_a = (first + within).astype(np.int64)

    @classmethod
    def of(cls, groups: dict, names: Sequence[str], ids: dict, to_map) -> "_Polylines":
        verts, starts, classes, all_ids = [], [0], [], []
        for c, name in enumerate(names):
            for line, line_id in zip(groups.get(name, []), ids.get(name, [])):
                line = np.asarray(line, np.float64)
                if line.ndim != 2 or len(line) == 0:
                    continue
                verts.append(to_map(line[:, :3]))
                starts.append(starts[-1] + len(line))
                classes.append(c)
                all_ids.append(line_id)
        vertices = np.concatenate(verts) if verts else np.zeros((0, 3))
        return cls(vertices, np.asarray(starts, np.int64), np.asarray(classes, np.int64), all_ids)

    def clip(self, R: np.ndarray, t: np.ndarray, box: tuple, rect: Optional[tuple] = None):
        """Segments in the ego frame (``R``, ``t``: the pose) clipped to *box* ``(cx, cy, hx, hy)``.

        *rect* ``(cx, cy, yaw, hx, hy)``: also clip to this rectangle in the map's x-y (the converter's).
        Returns ``(ego vertices, t0, t1, hit)`` per segment (Liang-Barsky, as ``clip_polyline``).
        """
        ego = (self.vertices - t) @ R
        cx, cy, hx, hy = box
        a = ego[self.seg_a]
        d = ego[self.seg_a + 1] - a
        t0, t1 = np.zeros(len(d)), np.ones(len(d))
        hit = np.ones(len(d), dtype=bool)
        t0, t1, hit = _liang_barsky(a[:, 0] - cx, a[:, 1] - cy, d[:, 0], d[:, 1], hx, hy, t0, t1, hit)
        if rect is not None:
            mx, my, yaw, rx, ry = rect
            c, s = np.cos(yaw), np.sin(yaw)
            am = self.vertices[self.seg_a, :2] - (mx, my)
            dm = self.vertices[self.seg_a + 1, :2] - self.vertices[self.seg_a, :2]
            t0, t1, hit = _liang_barsky(c * am[:, 0] + s * am[:, 1], -s * am[:, 0] + c * am[:, 1],
                                        c * dm[:, 0] + s * dm[:, 1], -s * dm[:, 0] + c * dm[:, 1], rx, ry, t0, t1, hit)
        return ego, t0, t1, hit

    def pieces(self, R: np.ndarray, t: np.ndarray, box: tuple, min_length: float, rect: Optional[tuple] = None):
        """The parts of every line inside *box* (and *rect*): ``[(line index, (n, 3) ego points), ...]`` in line order."""
        if not len(self.seg_a):
            return []
        ego, t0, t1, hit = self.clip(R, t, box, rect)
        k = np.flatnonzero(hit)
        if not len(k):
            return []
        a = ego[self.seg_a[k]]
        d = ego[self.seg_a[k] + 1] - a
        S = a + t0[k, None] * d
        E = a + t1[k, None] * d
        line = self.seg_line[k]
        # a piece starts at the first kept segment of a line, after a gap, or where the box cut the line
        new = np.ones(len(k), dtype=bool)
        new[1:] = (k[1:] != k[:-1] + 1) | (line[1:] != line[:-1]) | (t1[k[:-1]] < 1) | (t0[k[1:]] > 0)
        piece = np.cumsum(new) - 1
        n_pieces = int(piece[-1]) + 1
        out = np.empty((len(k) + n_pieces, 3))
        rows = np.arange(len(k)) + piece + 1
        out[rows] = E
        out[rows[new] - 1] = S[new]
        first_seg = np.flatnonzero(new)
        bounds = first_seg + np.arange(n_pieces)  # where each piece starts in out
        seg_len = np.linalg.norm((E - S)[:, :2], axis=1) if min_length > 0 else None
        lengths = np.add.reduceat(seg_len, first_seg) if seg_len is not None else None
        result = []
        for p, chunk in enumerate(np.split(out, bounds[1:])):
            if lengths is not None and lengths[p] < min_length:
                continue
            result.append((int(line[first_seg[p]]), chunk))
        return result

    def present(self, poses: Sequence[tuple], box: tuple) -> np.ndarray:
        """(L,) bool: lines with a part inside *box* at any of the poses ``(R, t)``."""
        out = np.zeros(len(self.classes), dtype=bool)
        if not len(self.seg_a):
            return out
        for R, t in poses:
            hit = self.clip(R, t, box)[3]
            out[self.seg_line[hit]] = True
            if out.all():
                break
        return out


@dataclasses.dataclass
class SceneLabels:
    """The map labels of a scene in the map frame, cut per frame (:meth:`__call__`); picklable, no lanelet2 objects.

    Attributes:
        region: The labelled region.
        hidden_ids: Traffic signs drawn by neither view (unknown subtype).
        classes: Line classes with a part in the region at some frame of the scene.
        te_classes: The same for traffic elements (hidden ones left out).
        edges: Some traffic element has an edge (to another element or a lane) at some frame.
        map_changed: The scene is a map_change scene.
    """

    region: "Region"
    line_names: tuple
    te_names: tuple
    lines: _Polylines
    tes: _Polylines
    te_to_te: np.ndarray  # (E, 2) global traffic element indices
    te_to_centerline: np.ndarray  # (E, 2) global traffic element index, global line index
    hidden_ids: frozenset
    classes: frozenset
    te_classes: frozenset
    edges: bool
    map_changed: bool = False
    dims: int = 3

    @classmethod
    def build(cls, scene: "Scene", region: "Region", grouping: str = "default",
              poses: Optional[Sequence["EgoPose"]] = None) -> "SceneLabels":
        """Ask the converter once for the whole scene (scene chaining) and keep the labels in the map frame.

        Args:
            poses: The poses of the frames to render (for :attr:`classes`); default: every frame with a pose.

        Raises:
            KITScenesError: MapUnavailable, LabelsUnavailable, ... as ``MLConverter``.
            ValueError: If the scene has no poses.
        """
        from kitscenes.map import MLConverter, Region
        from kitscenes.map.labels import _rotation, scene_chain_area
        from kitscenes.visualization import style

        if poses is None:
            from kitscenes.visualization.reprojection import _pose_or_none

            poses = [p for p in (_pose_or_none(scene.frame(k)) for k in range(len(scene))) if p is not None]
        if not poses:
            raise ValueError(f"scene {scene.id} has no poses")
        area = scene_chain_area(scene, region)
        anchor = poses[0]
        reach = area.radius + float(np.hypot(*(np.asarray(anchor.xyz[:2]) - np.asarray(area.center)))) + 5.0
        info = scene.split_info
        conv = MLConverter(scene.map, grouping=grouping, region=Region(reach, reach, reach, reach), sampling=None,
                           te_sampling=None, chaining=area,
                           map_changed=info is not None and info.family == "map_change")
        whole = conv(anchor)
        R0, t0 = _rotation(anchor), np.asarray(anchor.xyz, np.float64)

        def to_map(points):
            return points @ R0.T + t0

        line_names = tuple(whole.lines)
        te_names = tuple(whole.traffic_elements)
        lines = _Polylines.of(whole.lines, line_names, whole.line_ids, to_map)
        tes = _Polylines.of(whole.traffic_elements, te_names, whole.traffic_element_ids, to_map)
        # edges by global index: (class, index in class) -> position in the flattened lists
        te_global, n = {}, 0
        for name in te_names:
            for i, e in enumerate(whole.traffic_elements[name]):
                if np.asarray(e).ndim == 2 and len(e):
                    te_global[(name, i)] = n
                    n += 1
        line_global, n = {}, 0
        for name in line_names:
            for i, e in enumerate(whole.lines[name]):
                if np.asarray(e).ndim == 2 and len(e):
                    line_global[(name, i)] = n
                    n += 1
        centerline = "Centerline" if "Centerline" in line_names else None
        te_to_te = [(te_global[(a, i)], te_global[(b, j)]) for a, i, b, j in whole.te_to_te
                    if (a, i) in te_global and (b, j) in te_global]
        te_to_cl = [(te_global[(a, i)], line_global[(centerline, k)]) for a, i, k in whole.te_to_centerline
                    if centerline and (a, i) in te_global and (centerline, k) in line_global]
        hidden = frozenset(style.unknown_sign_ids(scene.map))
        out = cls(region, line_names, te_names, lines, tes, np.asarray(te_to_te, np.int64).reshape(-1, 2),
                  np.asarray(te_to_cl, np.int64).reshape(-1, 2), hidden, frozenset(), frozenset(), False,
                  whole.map_changed, whole.dims)
        out._presence(poses)
        return out

    @property
    def _box(self) -> tuple:
        r = self.region
        return ((r.ahead - r.behind) / 2, (r.left - r.right) / 2, (r.ahead + r.behind) / 2, (r.left + r.right) / 2)

    def _converter_rect(self, pose: "EgoPose") -> tuple:
        """The rectangle the converter cuts in (map x-y, turned by its yaw; ``MLConverter._query``)."""
        from kitscenes.map.labels import _converter_angles

        yaw = _converter_angles(pose)[0]
        r = self.region
        dx, dy = (r.ahead - r.behind) / 2, (r.left - r.right) / 2
        c, s = np.cos(yaw), np.sin(yaw)
        x, y = float(pose.xyz[0]), float(pose.xyz[1])
        return x + c * dx - s * dy, y + s * dx + c * dy, yaw, (r.ahead + r.behind) / 2, (r.left + r.right) / 2

    def _presence(self, poses: Sequence["EgoPose"]) -> None:
        from kitscenes.map.labels import _rotation
        from kitscenes.visualization import style

        frames = [(_rotation(p), np.asarray(p.xyz, np.float64)) for p in poses]
        box = self._box
        lines = self.lines.present(frames, box)
        tes = self.tes.present(frames, box)
        tes &= np.array([i not in self.hidden_ids for i in self.tes.ids], dtype=bool).reshape(-1)
        self.classes = frozenset(self.line_names[c] for c in np.unique(self.lines.classes[lines]))
        self.te_classes = frozenset(self.te_names[c] for c in np.unique(self.tes.classes[tes]))
        arrow = np.array([self.te_names[c] in style.ARROW_TYPES for c in self.tes.classes], dtype=bool).reshape(-1)
        a, b = self.te_to_te.T
        e1 = (tes[a] & tes[b] & ~arrow[a] & ~arrow[b]).any() if len(a) else False
        a, k = self.te_to_centerline.T
        e2 = (tes[a] & lines[k] & ~arrow[a]).any() if len(a) else False
        self.edges = bool(e1 or e2)

    def __call__(self, pose: "EgoPose") -> "MapLabels":
        """The labels at *pose*, in its ego frame, cut to the region (as ``MLConverter``, without lane segments)."""
        from kitscenes.map import MapLabels
        from kitscenes.map._vector import MIN_PIECE_LENGTH
        from kitscenes.map.labels import _rotation

        R, t = _rotation(pose), np.asarray(pose.xyz, np.float64)
        box = self._box
        rect = self._converter_rect(pose)
        lines = {name: [] for name in self.line_names}
        line_ids = {name: [] for name in self.line_names}
        first_piece = {}  # global line -> index in its class (its first piece, or the identical line kept)
        seen: dict = {}
        for k, points in self.lines.pieces(R, t, box, MIN_PIECE_LENGTH, rect):
            name = self.line_names[self.lines.classes[k]]
            key = (name, points.shape, points.tobytes())
            index = seen.get(key)
            if index is None:  # a line identical to an earlier one of its class is dropped (as MLConverter)
                index = seen[key] = len(lines[name])
                lines[name].append(points[:, :self.dims])
                line_ids[name].append(self.lines.ids[k])
            first_piece.setdefault(k, index)
        tes = {name: [] for name in self.te_names}
        te_ids = {name: [] for name in self.te_names}
        te_at = {}
        for k, points in self.tes.pieces(R, t, box, 0.0, rect):
            name = self.te_names[self.tes.classes[k]]
            te_at.setdefault(k, (name, len(tes[name])))
            tes[name].append(points[:, :self.dims])
            te_ids[name].append(self.tes.ids[k])
        te_to_te = [te_at[a] + te_at[b] for a, b in self.te_to_te.tolist() if a in te_at and b in te_at]
        te_to_cl = sorted({te_at[a] + (first_piece[k],) for a, k in self.te_to_centerline.tolist()
                           if a in te_at and k in first_piece})
        order = {name: i for i, name in enumerate(self.te_names)}
        te_to_cl.sort(key=lambda e: (order[e[0]], e[1], e[2]))
        return MapLabels(lines=lines, line_ids=line_ids, traffic_elements=tes, traffic_element_ids=te_ids,
                         te_to_centerline=te_to_cl, te_to_te=te_to_te, pose=pose, region=self.region, dims=self.dims,
                         map_changed=self.map_changed,
                         untyped_sign_ids=self.hidden_ids & {i for ids in te_ids.values() for i in ids})
