"""An HD map (Lanelet2 ``.osm``) as typed numpy elements, with region queries.

The map is loaded with lanelet2 on first use and converted once into plain
elements (:mod:`kitscenes.map.elements`); queries filter those. The lanelet2
objects stay available (:attr:`SceneMap.lanelet_map`, ``routing_graph``,
``traffic_rules``). A SceneMap pickles to its path and origin.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
import threading
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.map import _lanelet2
from kitscenes.map.elements import Lane, LaneKind, LaneMatch, MapLine, StopLine, TrafficLight, TrafficSign
from kitscenes.map.origin import MapOrigin
from kitscenes.map.roi import Roi, hits_area
from kitscenes.schema import EgoPose

logger = logging.getLogger(__name__)


class MapProblemWarning(UserWarning):
    """A map loads, but lanelet2 found problems in it (see :attr:`SceneMap.routing_problems`)."""

_NO_HEIGHT = 1e30  # the maps store -FLT_MAX for "no height"
_LIGHT_TYPES = ("traffic_light", "traffic_light_pedestrians")


def _points(linestring) -> np.ndarray:
    pts = np.array([[p.x, p.y, p.z] for p in linestring], dtype=np.float64).reshape(-1, 3)
    pts[np.abs(pts[:, 2]) > _NO_HEIGHT, 2] = np.nan
    pts.flags.writeable = False  # shared by every query of the cached map
    return pts


def _attrs(primitive) -> dict[str, str]:
    a = primitive.attributes
    return {k: str(a[k]) for k in a.keys()}


@dataclasses.dataclass(frozen=True, eq=False)
class MapPatch:
    """The elements of a map in a region, optionally moved into another frame.

    Attributes:
        frame: ``"map"``, ``"ego"`` (base frame at a pose) or ``"custom"``.
    """

    __repr__ = compact_repr

    lanes: list[Lane]
    lines: list[MapLine]
    stop_lines: list[StopLine]
    traffic_signs: list[TrafficSign]
    traffic_lights: list[TrafficLight]
    frame: str = "map"

    def transform(self, T: np.ndarray, frame: str = "custom") -> "MapPatch":
        """All points mapped by the (4, 4) transform *T*.

        Points without a known height keep z = NaN; for their x/y the median
        height of the patch is used (the maps store heights above the
        ellipsoid, so 0 would shift them by metres under any pitch).
        """
        finite = [p[:, 2][np.isfinite(p[:, 2])] for p in self._all_points()]
        fill = float(np.median(np.concatenate(finite))) if any(len(f) for f in finite) else 0.0

        def tf(p: np.ndarray) -> np.ndarray:
            unknown = np.isnan(p[:, 2])
            q = p.copy()
            q[unknown, 2] = fill
            out = q @ T[:3, :3].T + T[:3, 3]
            out[unknown, 2] = np.nan
            return out

        return MapPatch(
            lanes=[dataclasses.replace(
                lane, left_points=tf(lane.left_points), right_points=tf(lane.right_points),
                centerline=tf(lane.centerline)) for lane in self.lanes],
            lines=[dataclasses.replace(line, points=tf(line.points)) for line in self.lines],
            stop_lines=[dataclasses.replace(s, points=tf(s.points)) for s in self.stop_lines],
            traffic_signs=[dataclasses.replace(s, points=tf(s.points)) for s in self.traffic_signs],
            traffic_lights=[dataclasses.replace(s, points=tf(s.points)) for s in self.traffic_lights],
            frame=frame,
        )

    def _all_points(self) -> list[np.ndarray]:
        pts = [p for lane in self.lanes for p in (lane.left_points, lane.right_points, lane.centerline)]
        pts += [e.points for group in (self.lines, self.stop_lines, self.traffic_signs, self.traffic_lights) for e in group]
        return pts

    def in_ego(self, pose: EgoPose) -> "MapPatch":
        """The patch in the vehicle's base frame at *pose* (x forward, y left, z up).

        The base frame's origin is about 2 m above the road in the release
        (measured), so road elements end up at z of about -2 m.
        """
        T_world_base = pose.T_world_base
        T = np.eye(4)
        T[:3, :3] = T_world_base[:3, :3].T
        T[:3, 3] = -T_world_base[:3, :3].T @ T_world_base[:3, 3]
        return self.transform(T, "ego")


class SceneMap:
    """An HD map in a local UTM frame.

    Args:
        osm_path: Lanelet2 ``.osm`` file (scene map or city map).
        origin: The frame origin; use the scene's origin to get the frame of
            its ``poses.txt``.

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    def __init__(self, osm_path: Path | str, origin: MapOrigin) -> None:
        self.path = Path(osm_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Map file not found: {self.path}")
        self.origin = origin
        self._lock = threading.Lock()

    @classmethod
    def from_scene_dir(cls, scene_dir: Path | str) -> "SceneMap":
        """The map of a scene folder (``maps/map.osm`` + ``maps/origin.json``)."""
        scene_dir = Path(scene_dir)
        origin = MapOrigin.from_json((scene_dir / "maps" / "origin.json").read_text())
        return cls(scene_dir / "maps" / "map.osm", origin)

    def __repr__(self) -> str:
        state = f"{len(self._index.lanes)} lanes" if "_index" in self.__dict__ else "not loaded"
        return f"SceneMap({str(self.path)!r}, origin=({self.origin.lat:.5f}, {self.origin.lon:.5f}), {state})"

    def __getstate__(self) -> dict:
        return {"path": self.path, "origin": self.origin}

    def __setstate__(self, state: dict) -> None:
        self.path = state["path"]
        self.origin = state["origin"]
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # lanelet2 objects (escape hatch)
    # ------------------------------------------------------------------

    @cached_property
    def lanelet_map(self):
        """The ``lanelet2.core.LaneletMap``."""
        lanelet2 = _lanelet2.require()
        from lanelet2.io import Origin, loadRobust
        from lanelet2.projection import UtmProjector

        projector = UtmProjector(Origin(self.origin.lat, self.origin.lon))
        ll2_map, errors = loadRobust(str(self.path), projector)
        if errors:
            logger.warning("%s: lanelet2 reported %d load errors (first: %s)", self.path.name, len(errors), errors[0])
        del lanelet2
        return ll2_map

    @cached_property
    def traffic_rules(self):
        """German traffic rules for vehicles."""
        lanelet2 = _lanelet2.require()
        return lanelet2.traffic_rules.create(
            lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle,
        )

    @cached_property
    def routing_graph(self):
        """Vehicle routing graph.

        Two lanes with a bicycle lane between them are left/right neighbours
        when both bike-lane bounds may be crossed (dashed markings), see
        :data:`kitscenes.map._lanelet2.ROUTING_GRAPH_CONFIG`. This needs a
        lanelet2 with that feature; others warn once and leave it out.

        Maps with routing problems (e.g. an ``adjacentLeft`` relation that is
        not mutual) still load: the problems are listed in
        :attr:`routing_problems` and reported with a :class:`MapProblemWarning`.
        Routes through the affected lanelets may be missing or odd; the other
        queries are unaffected.
        """
        lanelet2 = _lanelet2.require()
        kwargs = {}
        if _lanelet2.has_bike_lane_lane_change():
            kwargs["configuration"] = dict(_lanelet2.ROUTING_GRAPH_CONFIG)
        else:
            _lanelet2.warn_no_bike_lane_lane_change("SceneMap.routing_graph")
        graph = lanelet2.routing.RoutingGraph(self.lanelet_map, self.traffic_rules, **kwargs)
        problems = [str(p) for p in graph.checkValidity(False)]  # True would raise on the first one
        self.__dict__["routing_problems"] = problems
        if problems:
            listing = "\n".join(f"  - {p}" for p in problems)
            logger.warning("%s: the routing graph has %d problems:\n%s", self.path, len(problems), listing)
            warnings.warn(
                f"MAP PROBLEMS in {self.path.name}: the routing graph has {len(problems)} problem(s); the map is "
                f"loaded anyway, but routing near them may be wrong (SceneMap.routing_problems lists them). "
                f"First: {problems[0]}",
                MapProblemWarning,
                stacklevel=3,
            )
        return graph

    @cached_property
    def routing_problems(self) -> list[str]:
        """The problems lanelet2 found in the routing graph (empty for a sound map)."""
        self.routing_graph
        return self.__dict__["routing_problems"]

    # ------------------------------------------------------------------
    # Conversion (once)
    # ------------------------------------------------------------------

    @cached_property
    def _index(self) -> "_Index":
        with self._lock:
            if "_index" in self.__dict__:  # another thread finished it
                return self.__dict__["_index"]
            return _Index.build(self)

    def lane(self, lane_id: int) -> Lane:
        return self._index.lanes_by_id[lane_id]

    def line(self, line_id: int) -> MapLine:
        return self._index.lines_by_id[line_id]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def lanes(self, roi: Optional[Roi] = None, kinds: Optional[Iterable[LaneKind | str]] = None) -> list[Lane]:
        """Lanes overlapping *roi* (all if None), optionally of the given kinds."""
        wanted = None if kinds is None else {LaneKind(k) for k in kinds}
        idx = self._index
        candidates = idx.candidates(idx.lane_boxes, roi)
        return [
            idx.lanes[i] for i in candidates
            if (wanted is None or idx.lanes[i].kind in wanted)
            and (roi is None or hits_area(roi, idx.lanes[i].polygon))
        ]

    def drivable_lanes(self, roi: Optional[Roi] = None) -> list[Lane]:
        """Lanes a vehicle may drive on (roads, bus lanes), overlapping *roi*.

        Their union (``lane.polygon``) is the lanes' footprint, not the drivable
        space up to the curbs: the ``drivable_space_border`` lines do not close
        into areas in the release maps.
        """
        return [lane for lane in self.lanes(roi) if lane.vehicle]

    def lines(self, roi: Optional[Roi] = None, types: Optional[Iterable[str]] = None) -> list[MapLine]:
        """Linestrings touching *roi*, optionally only of the given ``type`` tags."""
        wanted = None if types is None else set(types)
        idx = self._index
        out = []
        for i in idx.candidates(idx.line_boxes, roi):
            line = idx.lines[i]
            if (wanted is None or line.type in wanted) and (roi is None or roi.hits(line.points)):
                out.append(line)
        return out

    def stop_lines(self, roi: Optional[Roi] = None) -> list[StopLine]:
        return [s for s in self._index.stop_lines if roi is None or roi.hits(s.points)]

    def traffic_signs(self, roi: Optional[Roi] = None, *, require_height: bool = False) -> list[TrafficSign]:
        """Traffic signs, including those no regulatory element refers to (most) and ``subtype="unknown"``."""
        return [
            s for s in self._index.traffic_signs
            if (roi is None or roi.hits(s.points)) and (s.has_height or not require_height)
        ]

    def traffic_lights(self, roi: Optional[Roi] = None) -> list[TrafficLight]:
        return [s for s in self._index.traffic_lights if roi is None or roi.hits(s.points)]

    def query(self, roi: Roi) -> MapPatch:
        """Everything in *roi*, in the map frame; ``.in_ego(pose)`` moves it to the vehicle frame."""
        return MapPatch(
            lanes=self.lanes(roi), lines=self.lines(roi), stop_lines=self.stop_lines(roi),
            traffic_signs=self.traffic_signs(roi), traffic_lights=self.traffic_lights(roi),
        )

    def match(self, pose: EgoPose, *, max_distance: float = 1.0, max_heading_error: float = np.pi / 2) -> list[LaneMatch]:
        """Lanes a vehicle at *pose* may be driving on, best first.

        All lanes within *max_distance* that a vehicle may use in the pose's
        direction (``lanelet2.matching`` plus the traffic rules: no crosswalks,
        bike lanes or one-way lanes against their direction), whose direction
        differs from the heading by at most *max_heading_error*. Sorted by
        distance, then heading error.
        """
        lanelet2 = _lanelet2.require()
        x, y, yaw = float(pose.xyz[0]), float(pose.xyz[1]), pose.yaw
        obj = lanelet2.matching.Object2d()
        obj.pose = lanelet2.matching.Pose2d(x, y, yaw)
        candidates = lanelet2.matching.removeNonRuleCompliantMatches(
            lanelet2.matching.getDeterministicMatches(self.lanelet_map, obj, max_distance), self.traffic_rules,
        )
        out = []
        for c in candidates:
            lane = self._index.lanes_by_id[c.lanelet.id]
            centerline = lane.centerline[::-1] if c.lanelet.inverted() else lane.centerline
            error = _heading_error(centerline, x, y, yaw)
            if error <= max_heading_error:
                out.append(LaneMatch(lane.id, bool(c.lanelet.inverted()), float(c.distance), error))
        out.sort(key=lambda m: (m.distance, m.heading_error))
        return out

    def route(self, from_lane: int, to_lane: int) -> Optional[list[int]]:
        """Lane IDs of the shortest vehicle route, or None if there is none."""
        ll = self.lanelet_map.laneletLayer
        path = self.routing_graph.shortestPath(ll[from_lane], ll[to_lane])
        return None if path is None else [lanelet.id for lanelet in path]


def _heading_error(centerline: np.ndarray, x: float, y: float, yaw: float) -> float:
    """Angle between *yaw* and the polyline direction at its segment closest to (x, y)."""
    if len(centerline) < 2:
        return float("nan")
    a, b = centerline[:-1, :2], centerline[1:, :2]
    ab = b - a
    q = np.array([x, y])
    t = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12), 0.0, 1.0)
    i = int(np.argmin(np.linalg.norm(a + t[:, None] * ab - q, axis=1)))
    direction = float(np.arctan2(ab[i, 1], ab[i, 0]))
    return float(abs((direction - yaw + np.pi) % (2 * np.pi) - np.pi))


def _bbox(points: np.ndarray) -> np.ndarray:
    return np.concatenate([points[:, :2].min(axis=0), points[:, :2].max(axis=0)]) if len(points) else np.full(4, np.nan)


@dataclasses.dataclass
class _Index:
    lanes: list[Lane]
    lanes_by_id: dict[int, Lane]
    lane_boxes: np.ndarray
    lines: list[MapLine]
    lines_by_id: dict[int, MapLine]
    line_boxes: np.ndarray
    stop_lines: list[StopLine]
    traffic_signs: list[TrafficSign]
    traffic_lights: list[TrafficLight]

    @staticmethod
    def candidates(boxes: np.ndarray, roi: Optional[Roi]) -> Sequence[int]:
        """Elements whose bounding box overlaps the region's bounding box."""
        if roi is None:
            return range(len(boxes))
        if hasattr(roi, "radius"):
            r = roi.radius
            lo = np.asarray(roi.center) - r
            hi = np.asarray(roi.center) + r
        else:
            reach = np.hypot(max(roi.ahead, roi.behind), max(roi.left, roi.right))
            lo = np.asarray(roi.center) - reach
            hi = np.asarray(roi.center) + reach
        keep = (boxes[:, 0] <= hi[0]) & (boxes[:, 2] >= lo[0]) & (boxes[:, 1] <= hi[1]) & (boxes[:, 3] >= lo[1])
        return np.flatnonzero(keep)

    @classmethod
    def build(cls, smap: SceneMap) -> "_Index":
        m, rules = smap.lanelet_map, smap.traffic_rules
        graph = smap.routing_graph

        lines = sorted(
            (MapLine(ls.id, *_type_subtype(ls), _points(ls), _attrs(ls)) for ls in m.lineStringLayer),
            key=lambda line: line.id,
        )
        lines_by_id = {line.id: line for line in lines}

        # Regulatory elements: what they refer to, their stop lines, their lanes.
        reg_lanes: dict[int, list[int]] = {}
        lanes: list[Lane] = []
        for llt in m.laneletLayer:
            for reg in llt.regulatoryElements:
                reg_lanes.setdefault(reg.id, []).append(llt.id)
        for llt in sorted(m.laneletLayer, key=lambda x: x.id):
            attrs = _attrs(llt)
            vehicle = bool(rules.canPass(llt))
            succ = pred = ()
            left_n = right_n = None
            if vehicle:
                succ = tuple(x.id for x in graph.following(llt))
                pred = tuple(x.id for x in graph.previous(llt))
                left, right = graph.left(llt), graph.right(llt)
                left_n = None if left is None else left.id
                right_n = None if right is None else right.id
            speed = float(rules.speedLimit(llt).speedLimitMPS) if vehicle else None
            lanes.append(Lane(
                id=llt.id,
                kind=LaneKind.of(attrs.get("subtype", "")),
                left=llt.leftBound.id,
                right=llt.rightBound.id,
                left_points=_points(llt.leftBound),
                right_points=_points(llt.rightBound),
                centerline=_points(llt.centerline),
                speed_limit_mps=speed if speed and np.isfinite(speed) and speed > 0 else None,
                vehicle=vehicle,
                one_way=vehicle and not bool(rules.canPass(llt.invert())),
                successors=succ,
                predecessors=pred,
                left_neighbor=left_n,
                right_neighbor=right_n,
                regulatory_ids=tuple(r.id for r in llt.regulatoryElements),
                attributes=attrs,
            ))

        refers: dict[int, list[int]] = {}       # line id -> reg ids that refer to it
        reg_stop_lines: dict[int, list[int]] = {}  # reg id -> stop line ids
        stop_line_lanes: dict[int, set[int]] = {}
        for reg in m.regulatoryElementLayer:
            params = reg.parameters
            for role in params.keys():
                for p in params[role]:
                    pid = getattr(p, "id", None)
                    if pid is None or pid not in lines_by_id:
                        continue
                    if role == "ref_line":
                        reg_stop_lines.setdefault(reg.id, []).append(pid)
                        stop_line_lanes.setdefault(pid, set()).update(reg_lanes.get(reg.id, ()))
                    elif role == "refers":
                        refers.setdefault(pid, []).append(reg.id)

        stop_ids = {line.id for line in lines if line.type == "stop_line"} | set(stop_line_lanes)
        stop_lines = [
            StopLine(i, lines_by_id[i].points, tuple(sorted(stop_line_lanes.get(i, ())))) for i in sorted(stop_ids)
        ]

        def lanes_of(regs: Iterable[int]) -> tuple[int, ...]:
            return tuple(sorted({lane for r in regs for lane in reg_lanes.get(r, ())}))

        signs = [
            TrafficSign(l.id, l.type, l.subtype, l.points, l.attributes,
                        regulatory_ids=tuple(refers.get(l.id, ())), lane_ids=lanes_of(refers.get(l.id, ())))
            for l in lines if l.type == "traffic_sign"
        ]
        lights = [
            TrafficLight(l.id, l.type, l.subtype, l.points, l.attributes,
                         pedestrian=l.type == "traffic_light_pedestrians",
                         stop_line_ids=tuple(sorted({s for r in refers.get(l.id, ()) for s in reg_stop_lines.get(r, ())})),
                         lane_ids=lanes_of(refers.get(l.id, ())))
            for l in lines if l.type in _LIGHT_TYPES
        ]
        return cls(
            lanes=lanes,
            lanes_by_id={lane.id: lane for lane in lanes},
            lane_boxes=np.array([_bbox(lane.polygon) for lane in lanes]).reshape(-1, 4),
            lines=lines,
            lines_by_id=lines_by_id,
            line_boxes=np.array([_bbox(line.points) for line in lines]).reshape(-1, 4),
            stop_lines=stop_lines,
            traffic_signs=signs,
            traffic_lights=lights,
        )


def _type_subtype(ls) -> tuple[str, str]:
    a = ls.attributes
    return (str(a["type"]) if "type" in a else "", str(a["subtype"]) if "subtype" in a else "")
