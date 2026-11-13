"""Routes on the SD map: :class:`SDRoute`, its maneuvers (:class:`SDManeuver`) and turn directions.

A route is a sequence of roads (:class:`~kitscenes.sdmap.SDRoad`) from a
position on the first to a position on the last, with its polyline in the
map frame. Its maneuvers are found at the junctions it passes where there is
a choice (:class:`~kitscenes.sdmap.SDJunction`: graph nodes merged across
short internal pieces, so a left turn across a dual carriageway is one
maneuver). The turn angle is measured between the route's direction over the
``approach_m`` before the junction and over the ``approach_m`` after it, and
binned like OSRM's guidance (:class:`Direction`).
"""

from __future__ import annotations

import dataclasses
import math
from enum import Enum
from functools import cached_property
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.constants import DEFAULT_ROAD_HEIGHT_M
from kitscenes.sdmap._geometry import arc_length, heading_at, point_at
from kitscenes.sdmap.elements import T_base_world, to_ego
from kitscenes.schema import EgoPose

if TYPE_CHECKING:
    from kitscenes.navigation import Command, CommandRule
    from kitscenes.sdmap.sd_map import SDMap

#: Distance over which the approach and exit directions of a maneuver are measured (metres).
APPROACH_M = 20.0


class Direction(str, Enum):
    """Turn direction of a maneuver, from the turn angle (left positive), in OSRM's bins:
    straight up to 20°, slight 20-40°, turn 40-120°, sharp 120-170°, U-turn beyond 170°."""

    STRAIGHT = "straight"
    SLIGHT_LEFT = "slight_left"
    LEFT = "left"
    SHARP_LEFT = "sharp_left"
    SLIGHT_RIGHT = "slight_right"
    RIGHT = "right"
    SHARP_RIGHT = "sharp_right"
    UTURN = "uturn"

    @classmethod
    def of(cls, angle_rad: float) -> "Direction":
        """The direction of a turn by *angle_rad* (left positive)."""
        a = math.degrees(abs(angle_rad))
        if a <= 20:
            return cls.STRAIGHT
        if a > 170:
            return cls.UTURN
        left = angle_rad > 0
        if a <= 40:
            return cls.SLIGHT_LEFT if left else cls.SLIGHT_RIGHT
        if a <= 120:
            return cls.LEFT if left else cls.RIGHT
        return cls.SHARP_LEFT if left else cls.SHARP_RIGHT

    @property
    def is_left(self) -> bool:
        return self in (Direction.SLIGHT_LEFT, Direction.LEFT, Direction.SHARP_LEFT)

    @property
    def is_right(self) -> bool:
        return self in (Direction.SLIGHT_RIGHT, Direction.RIGHT, Direction.SHARP_RIGHT)


#: ``turn:lanes`` indications that serve each direction.
_LANE_INDICATIONS = {
    Direction.STRAIGHT: {"through", "none"},
    Direction.SLIGHT_LEFT: {"slight_left", "through", "left"},
    Direction.LEFT: {"left", "slight_left", "sharp_left"},
    Direction.SHARP_LEFT: {"sharp_left", "left"},
    Direction.SLIGHT_RIGHT: {"slight_right", "through", "right"},
    Direction.RIGHT: {"right", "slight_right", "sharp_right"},
    Direction.SHARP_RIGHT: {"sharp_right", "right"},
    Direction.UTURN: {"reverse", "left"},
}


@dataclasses.dataclass(frozen=True)
class SDManeuver:
    """A maneuver along an :class:`SDRoute`.

    Attributes:
        kind: ``"depart"``, ``"arrive"``, ``"turn"`` (a junction with a choice,
            not straight on), ``"continue"`` (straight on through a junction
            with a choice), ``"fork"`` (two exits both within 40° of straight
            on), ``"roundabout"`` (the whole passage, entry to exit),
            ``"on_ramp"``, ``"off_ramp"``, ``"merge"`` (onto a motorway or trunk
            from a link), ``"u_turn"`` (back onto the same piece of road).
        direction: The :class:`Direction` of :attr:`angle_rad`; for a fork,
            ``SLIGHT_LEFT`` / ``SLIGHT_RIGHT`` by the side of the chosen exit.
        angle_rad: Turn angle, left positive: the route's direction after the
            junction minus its direction before it.
        s_m: Arc length along the route where the maneuver starts (where the route enters the junction).
        end_s_m: Where it ends (leaves the junction; the roundabout exit).
        xy: (2,) map-frame position at :attr:`s_m`.
        junction_id: The junction (None for depart and arrive).
        options: Exits the route could have taken there (U-turns not counted).
        control: The junction's control (``"traffic_signals"``, ...; ``"none"``).
        roundabout_exit: For a roundabout, the number of the exit taken.
        road_in, road_out: The roads before and after the junction.
        lanes_valid: For the approach road's ``turn:lanes``, whether each lane
            (left to right) serves this direction; None without the tag.
        straightest: Whether the route takes the exit closest to straight on
            (False: a "keep left / right" onto a branch while another one goes
            straighter); None where there was no choice.
    """

    __repr__ = compact_repr

    kind: str
    direction: Direction
    angle_rad: float
    s_m: float
    end_s_m: float
    xy: np.ndarray
    junction_id: Optional[int] = None
    options: int = 0
    control: str = "none"
    roundabout_exit: Optional[int] = None
    road_in: Optional[int] = None
    road_out: Optional[int] = None
    lanes_valid: Optional[tuple[bool, ...]] = None
    straightest: Optional[bool] = None

    @property
    def is_decision(self) -> bool:
        """Whether the driver had to choose here (turn, continue, fork, roundabout, ramps)."""
        return self.kind not in ("depart", "arrive", "merge")


Pose = Union[EgoPose, np.ndarray]


def _xy_yaw(pose: Pose) -> tuple[np.ndarray, Optional[float]]:
    if isinstance(pose, EgoPose):
        return np.asarray(pose.xyz[:2], float), pose.yaw
    p = np.asarray(pose, float).reshape(-1)
    return p[:2], (float(p[2]) if len(p) > 2 else None)


@dataclasses.dataclass(frozen=True, eq=False)
class SDRoute:
    """A route on an SD map (map frame), see :meth:`SDMap.route` and :meth:`SDMap.route_along`.

    Attributes:
        road_ids: The roads, in order.
        start_s_m: Where the route starts on the first road (arc length from the road's start).
        end_s_m: Where it ends on the last road.
        points: (M, 2) map-frame polyline.
        s_m: (M,) arc length of each point from the route's start.
        length_m: Its length.
        engine: How it was found: ``"builtin"`` (:meth:`SDMap.route`) or ``"matched"`` (map matching of a trajectory).
        sd_map: The map it is on.
    """

    __repr__ = compact_repr

    road_ids: tuple[int, ...]
    start_s_m: float
    end_s_m: float
    points: np.ndarray
    s_m: np.ndarray
    length_m: float
    engine: str = "builtin"
    sd_map: Optional["SDMap"] = dataclasses.field(default=None, repr=False)

    @classmethod
    def from_roads(cls, sd_map: "SDMap", road_ids, start_s_m: float, end_s_m: float, engine: str = "builtin") -> "SDRoute":
        """The route over *road_ids* from *start_s_m* on the first to *end_s_m* on the last."""
        road_ids = tuple(int(r) for r in road_ids)
        g = sd_map._graph
        parts = []
        for k, r in enumerate(road_ids):
            p = g.road_points(r)
            s = arc_length(p)
            lo = start_s_m if k == 0 else 0.0
            hi = end_s_m if k == len(road_ids) - 1 else s[-1]
            lo, hi = float(np.clip(lo, 0, s[-1])), float(np.clip(hi, 0, s[-1]))
            inner = p[(s > lo) & (s < hi)]
            piece = np.concatenate([point_at(p, s, [lo]), inner, point_at(p, s, [hi])])
            parts.append(piece if not parts else piece[1:])
        points = np.concatenate(parts) if parts else np.zeros((0, 2))
        keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]) if len(points) else []
        points = points[keep] if len(points) else points
        if len(points) == 1:
            points = np.repeat(points, 2, axis=0)
        s = arc_length(points)
        points.flags.writeable = False
        s.flags.writeable = False
        return cls(road_ids, float(start_s_m), float(end_s_m), points, s, float(s[-1]) if len(s) else 0.0,
                   engine, sd_map)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def point_at(self, s_m) -> np.ndarray:
        """(2,) map-frame point at arc length *s_m* (clamped to the route), or (K, 2) for an array."""
        out = point_at(self.points, self.s_m, np.atleast_1d(s_m))
        return out[0] if np.ndim(s_m) == 0 else out

    def heading_at(self, s_m: float, span_m: float = 0.0) -> float:
        """Direction of the route at *s_m* in radians (map frame); averaged over ±*span_m* if given."""
        return heading_at(self.points, self.s_m, float(s_m), span_m)

    def project(self, pose: Pose, *, near_s_m: Optional[float] = None, window_m: float = 50.0) -> float:
        """Arc length of the point of the route closest to the pose (EgoPose or x, y).

        Args:
            near_s_m: Search only within *window_m* of this arc length (for
                routes that pass a place twice, and for poses along the route in order).
        """
        xy, _ = _xy_yaw(pose)
        p, s = self.points, self.s_m
        a, b = p[:-1], p[1:]
        ab = b - a
        l2 = np.einsum("ij,ij->i", ab, ab)
        t = np.clip(np.einsum("ij,ij->i", xy - a, ab) / np.maximum(l2, 1e-12), 0.0, 1.0)
        d = np.hypot(*(a + t[:, None] * ab - xy).T)
        if near_s_m is not None:
            far = (s[1:] < near_s_m - window_m) | (s[:-1] > near_s_m + window_m)
            if not far.all():
                d = np.where(far, np.inf, d)
        i = int(np.argmin(d))
        return float(s[i] + t[i] * np.sqrt(l2[i]))

    def distance_to(self, pose: Pose) -> float:
        """Distance of the pose to the route polyline in metres."""
        xy, _ = _xy_yaw(pose)
        return float(np.hypot(*(self.point_at(self.project(xy)) - xy)))

    def in_ego(self, pose: EgoPose, *, road_height_m: float = DEFAULT_ROAD_HEIGHT_M) -> np.ndarray:
        """(M, 3) route polyline in the base frame at *pose*, on the road (see :func:`kitscenes.sdmap.elements.to_ego`)."""
        return to_ego(self.points, pose, road_height_m)

    def road_at(self, s_m: float) -> int:
        """The road at arc length *s_m*."""
        k = int(np.searchsorted(self._road_starts, s_m, side="right") - 1)
        return self.road_ids[int(np.clip(k, 0, len(self.road_ids) - 1))]

    @cached_property
    def _road_starts(self) -> np.ndarray:
        """Arc length where each road starts on the route (the first at 0)."""
        g = self.sd_map._graph
        lengths = np.array([float(g.road_length[r]) for r in self.road_ids])
        starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]]) - self.start_s_m
        starts[0] = 0.0
        return starts

    def speed_limits(self, unit: str = "mps") -> np.ndarray:
        """(M,) speed limit at each point (of the road the point's segment is on); NaN unknown, inf none.

        In m/s, or in km/h with ``unit="kph"``.
        """
        if unit not in ("kph", "mps"):
            raise ValueError(f"unit must be 'kph' or 'mps', not {unit!r}")
        out = np.full(len(self.points), np.nan)
        for i, s in enumerate(self.s_m):
            v = self.sd_map.road(self.road_at(min(float(s), self.length_m - 1e-6) if self.length_m > 0 else 0.0)).speed_limit_mps
            out[i] = np.nan if v is None else v
        return out * 3.6 if unit == "kph" else out

    # ------------------------------------------------------------------
    # Maneuvers and navigation
    # ------------------------------------------------------------------

    @cached_property
    def free_run_m(self) -> float:
        """How far the road goes on past the route's end before the next point with a choice (at most 500 m):
        along the last road and its successors while there is only one."""
        g = self.sd_map._graph
        r = self.road_ids[-1]
        d = float(g.road_length[r]) - self.end_s_m
        seen = {r}
        while d < 500.0:
            succ = g.successors(r)
            if len(succ) != 1 or succ[0] in seen:
                break
            r = succ[0]
            seen.add(r)
            d += float(g.road_length[r])
        return min(d, 500.0)

    @cached_property
    def maneuvers(self) -> tuple[SDManeuver, ...]:
        """Depart, the maneuvers at the junctions with a choice (see the module docs), and arrive."""
        return _maneuvers(self)

    def next_maneuver(self, pose: Pose, *, near_s_m: Optional[float] = None) -> Optional[tuple[SDManeuver, float]]:
        """The next decision maneuver ahead of the pose (or the one it is in) and the distance to its start
        (negative inside it): ``(maneuver, distance_m)``, None if there is none ahead. Depart, arrive and
        merges are skipped."""
        s0 = self.project(pose, near_s_m=near_s_m)
        for m in self.maneuvers:
            if not m.is_decision:
                continue
            if m.end_s_m >= s0 and m.kind != "depart":
                return m, m.s_m - s0
        return None

    def command(self, pose: Pose, rule: Optional["CommandRule"] = None, *, near_s_m: Optional[float] = None) -> "Command":
        """The driving command at the pose by *rule* (default :data:`kitscenes.navigation.KITSCENES`),
        see :mod:`kitscenes.navigation`."""
        from kitscenes.navigation import KITSCENES, route_command

        return route_command(self, pose, rule or KITSCENES, near_s_m=near_s_m)

    def target_points(self, pose: EgoPose, *, max_distance_m: float = 50.0, spacing_m: Optional[float] = None,
                      at_maneuvers: bool = True, near_s_m: Optional[float] = None) -> np.ndarray:
        """Route points ahead of the pose in its base frame, (K, 2) x, y (x forward, y left).

        CARLA / TransFuser style: the exits of the decision maneuvers within
        *max_distance_m* ahead (``at_maneuvers``) and the point *max_distance_m*
        ahead (or the route end), in order; with *spacing_m* instead a point
        every *spacing_m* up to *max_distance_m*.
        """
        s0 = self.project(pose, near_s_m=near_s_m)
        end = min(s0 + max_distance_m, self.length_m)
        if spacing_m is not None:
            at = list(np.arange(s0 + spacing_m, end + 1e-9, spacing_m))
        else:
            at = [m.end_s_m for m in self.maneuvers if at_maneuvers and m.is_decision and s0 < m.end_s_m < end]
            at.append(end)
        if not at:
            return np.zeros((0, 2))
        xy = self.point_at(np.array(at, float))
        T = T_base_world(pose)
        return xy @ T[:2, :2].T + T[:2, 3]

    def to_lanelets(self, scene_map, *, max_distance: float = 6.0, max_heading_error: float = math.pi / 4,
                    step_m: float = 5.0, candidates: int = 3) -> Optional[list[int]]:
        """The HD-map lanes (:class:`kitscenes.map.SceneMap`, needs ``kitscenes[map]``) along this route.

        The lanes near the route's start and end in its direction (within
        *max_distance*; the OSM centre line lies 1-5 m beside the lanes, the
        best *candidates* of each) are joined with the HD routing graph (lane
        changes included); of those HD routes, the shortest whose centre lines
        stay within *max_distance* + 2 m of the SD route (sampled every
        *step_m*) is returned, None if there is none.
        """
        def lanes_at(s: float) -> list[int]:
            xy = self.point_at(s)
            yaw = self.heading_at(s, 2.0)
            pose = EgoPose(0, np.array([xy[0], xy[1], 0.0]), np.array([0, 0, math.sin(yaw / 2), math.cos(yaw / 2)]))
            return [m.lane_id for m in scene_map.match(pose, max_distance=max_distance,
                                                       max_heading_error=max_heading_error)][:candidates]

        first, last = lanes_at(0.0), lanes_at(self.length_m)
        if not first or not last:
            return None
        samples = self.point_at(np.arange(0.0, self.length_m + 1e-9, step_m))
        best = None
        for a in first:
            for b in last:
                path = [a] if a == b else scene_map.route(a, b)
                if not path:
                    continue
                centre = np.concatenate([scene_map.lane(i).centerline[:, :2] for i in path])
                from scipy.spatial import cKDTree

                from kitscenes.sdmap._geometry import densify

                gap = cKDTree(densify(centre, 1.0)).query(samples)[0].max()
                length = float(sum(arc_length(scene_map.lane(i).centerline)[-1] for i in path))
                if gap <= max_distance + 2.0 and (best is None or length < best[0]):
                    best = (length, path)
        return None if best is None else list(best[1])


def _maneuvers(route: SDRoute) -> tuple[SDManeuver, ...]:
    smap = route.sd_map
    g = smap._graph
    jx = smap._junction_index
    roads = route.road_ids
    starts = route._road_starts
    out = [SDManeuver("depart", Direction.STRAIGHT, 0.0, 0.0, 0.0, route.points[0].copy(), road_out=roads[0])]
    k = 0
    n = len(roads)
    while k < n - 1:
        a = roads[k]
        node = int(g.road_to[a])
        road_a = smap.road(a)
        road_b = smap.road(roads[k + 1])
        if roads[k + 1] == g.reverse(a):  # turning back on the same piece
            s = float(starts[k + 1])
            m = _make(route, "u_turn", s, s, smap.junction_at_node(node), options=len(g.successors(a)),
                      road_in=a, road_out=roads[k + 1])
            out.append(dataclasses.replace(m, direction=Direction.UTURN))
            k += 1
            continue
        # a roundabout: from entering the ring to leaving it
        if road_b.roundabout and not road_a.roundabout:
            j = k + 1
            exits = 0
            while j < n and smap.road(roads[j]).roundabout:
                ring_end = int(g.road_to[roads[j]])
                if any(not smap.road(int(x)).roundabout for x in g.leaving(ring_end)):
                    exits += 1
                j += 1
            entry_s = float(starts[k + 1])
            exit_s = float(starts[j]) if j < n else route.length_m
            out.append(_make(route, "roundabout", entry_s, exit_s, smap.junction_at_node(node), options=exits,
                             road_in=a, road_out=roads[j] if j < n else None, roundabout_exit=exits if j < n else None))
            k = j
            continue
        cluster = jx.cluster_of.get(node)
        if cluster is None:
            kind = _ramp_kind(road_a, road_b)
            if kind is not None and len(g.successors(a)) > 1 or kind == "merge":
                s = float(starts[k + 1])
                out.append(_make(route, kind, s, s, None, options=len(g.successors(a)), road_in=a, road_out=roads[k + 1]))
            k += 1
            continue
        # the passage through the junction: internal roads of the cluster
        j = k + 1
        while j < n and roads[j] in jx.internal_roads and jx.cluster_of.get(int(g.road_to[roads[j]])) == cluster:
            j += 1
        if j >= n:  # the route ends inside the junction
            break
        entry_s, exit_s = float(starts[k + 1]), float(starts[j])
        options, option_angles = _options(route, smap, a, cluster, entry_s)
        b = roads[j]
        kind = _ramp_kind(road_a, smap.road(b))
        if len(options) >= 2 or kind == "merge":
            m = _make(route, kind or "turn", entry_s, exit_s, cluster, options=len(options), road_in=a, road_out=b)
            if b in option_angles and len(option_angles) >= 2:
                others = [abs(x) for r, x in option_angles.items() if r != b]
                m = dataclasses.replace(m, straightest=abs(option_angles[b]) <= min(others) + math.radians(10))
            if kind is None:
                if len(options) == 2 and all(abs(x) < math.radians(40) for x in option_angles.values()) \
                        and b in option_angles:
                    other = [x for r, x in option_angles.items() if r != b][0]
                    left = option_angles[b] > other
                    m = dataclasses.replace(m, kind="fork",
                                            direction=Direction.SLIGHT_LEFT if left else Direction.SLIGHT_RIGHT)
                elif m.direction is Direction.STRAIGHT:
                    m = dataclasses.replace(m, kind="continue")
            out.append(m)
        k = j
    end_road = roads[-1]
    out.append(SDManeuver("arrive", Direction.STRAIGHT, 0.0, route.length_m, route.length_m, route.points[-1].copy(),
                          road_in=end_road))
    return tuple(out)


def _ramp_kind(a, b) -> Optional[str]:
    major = ("motorway", "trunk")
    if b.is_link and not a.is_link and a.road_class.value in major:
        return "off_ramp"
    if b.is_link and not a.is_link and b.road_class.value in major and a.road_class.value not in major:
        return "on_ramp"
    if a.is_link and not b.is_link and b.road_class.value in major:
        return "merge"
    return None


def _options(route: SDRoute, smap, a: int, cluster: int, entry_s: float) -> tuple[list[int], dict[int, float]]:
    """The exits of junction *cluster* reachable from road *a* (through its internal roads, no U-turn
    back onto *a*'s piece) and their turn angles relative to the approach direction."""
    g = smap._graph
    jx = smap._junction_index
    approach = route.heading_at(max(entry_s - APPROACH_M / 2, 0.0), APPROACH_M / 2)
    rev = g.reverse(a)
    seen = {a}
    stack = [a]
    exits: list[int] = []
    while stack:
        r = stack.pop()
        for b in g.successors(r):
            if b in seen or b == rev:
                continue
            seen.add(b)
            if b in jx.internal_roads and jx.cluster_of.get(int(g.road_to[b])) == cluster:
                stack.append(b)
            elif smap._graph.road_public[b] or b in route.road_ids:
                exits.append(b)
    angles = {}
    for b in exits:
        p = g.road_points(b)
        s = arc_length(p)
        d = point_at(p, s, [min(APPROACH_M, s[-1])])[0] - p[0]
        angles[b] = float((math.atan2(d[1], d[0]) - approach + math.pi) % (2 * math.pi) - math.pi)
    return sorted(set(exits)), angles


def _make(route: SDRoute, kind: str, entry_s: float, exit_s: float, junction: Optional[int], *, options: int,
          road_in: Optional[int], road_out: Optional[int], roundabout_exit: Optional[int] = None) -> SDManeuver:
    smap = route.sd_map
    before = route.heading_at(max(entry_s - APPROACH_M / 2, 0.0), APPROACH_M / 2)
    after = route.heading_at(min(exit_s + APPROACH_M / 2, route.length_m), APPROACH_M / 2)
    angle = float((after - before + math.pi) % (2 * math.pi) - math.pi)
    direction = Direction.of(angle)
    lanes_valid = None
    if road_in is not None:
        turn_lanes = smap.road(road_in).turn_lanes
        if turn_lanes:
            wanted = _LANE_INDICATIONS[direction]
            lanes_valid = tuple(bool(lane & wanted) for lane in turn_lanes)
    control = smap.junction(junction).control if junction is not None else "none"
    return SDManeuver(kind, direction, angle, entry_s, exit_s, route.point_at(entry_s), junction, options, control,
                      roundabout_exit, road_in, road_out, lanes_valid)
