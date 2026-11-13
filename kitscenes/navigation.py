"""Driving commands and target points: from the recorded poses alone or from an SD route.

Experimental: the benchmark's navigation input is not final; the test-e2e
scenes will ship their commands.

A command is LEFT, STRAIGHT, RIGHT or UNKNOWN (:class:`Command`), computed by
a :class:`CommandRule`. Three presets:

- :data:`NAVSIM`: the lateral offset of the path point 20 m ahead (the driven path or an SD route), ±2 m.
- :data:`NUSCENES`: the lateral offset of the recorded pose 3 s ahead, ±2 m; it reads the future, so it is a label.
- :data:`KITSCENES`: the next junction maneuver of the SD route within 40 m, a turn from 35°; needs the SD map.

Example::

    from kitscenes.navigation import KITSCENES, NAVSIM, NUSCENES, driving_commands

    commands = driving_commands(scene, NAVSIM)                    # (F,) int8, the driven path
    commands = driving_commands(scene, KITSCENES)                 # from the SD route (needs the SD map)
    nav = frame.navigation(KITSCENES)                             # command + target points of a frame
    nav.command.onehot("navsim"), nav.target_points               # (4,), (K, 2) base frame
"""

from __future__ import annotations

import dataclasses
import math
from enum import IntEnum
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.errors import Availability, MapUnavailable

if TYPE_CHECKING:
    from kitscenes.frame import Frame
    from kitscenes.scene import Scene
    from kitscenes.sdmap.route import SDManeuver, SDRoute

Source = Literal["auto", "path", "sd_route"]
Targets = Literal["goal", "route"]
Goal = Literal["sd_route", "path"]


class Command(IntEnum):
    """A driving command (the NAVSIM order: LEFT, STRAIGHT, RIGHT, UNKNOWN)."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2
    UNKNOWN = 3

    def onehot(self, order: str = "navsim") -> np.ndarray:
        """One-hot float32 vector.

        ``"navsim"``: (left, straight, right, unknown). ``"vad"`` (also UniAD):
        (right, left, straight); UNKNOWN is all zeros there.
        """
        try:
            classes = ONEHOT_ORDERS[order]
        except KeyError:
            raise ValueError(f"order must be one of {sorted(ONEHOT_ORDERS)}, got {order!r}") from None
        return np.array([self is c for c in classes], dtype=np.float32)


#: The class order of the one-hot encodings (:meth:`Command.onehot`).
ONEHOT_ORDERS: dict[str, tuple[Command, ...]] = {
    "navsim": (Command.LEFT, Command.STRAIGHT, Command.RIGHT, Command.UNKNOWN),
    "vad": (Command.RIGHT, Command.LEFT, Command.STRAIGHT),
}


@dataclasses.dataclass(frozen=True)
class CommandRule:
    """How a driving command is computed.

    - ``"lateral"`` (:data:`NAVSIM`): the point *lookahead_m* ahead along a
      path (``source="path"``: the recorded positions; ``"sd_route"``: the
      driven route on the SD map), its lateral offset in the vehicle's
      heading frame: >= *threshold_m* LEFT, <= -*threshold_m* RIGHT, else
      STRAIGHT; UNKNOWN if the path ends sooner.
    - ``"future_pose"`` (:data:`NUSCENES`): the same thresholds on the
      recorded pose *horizon_s* ahead in the vehicle frame; UNKNOWN if the
      scene ends sooner. Needs no map.
    - ``"maneuver"`` (:data:`KITSCENES`): the next maneuver of an SD route
      within *lookahead_m*, or the one the vehicle is in
      (:attr:`kitscenes.sdmap.SDRoute.maneuvers`: junctions with a choice).
      LEFT / RIGHT if it turns by at least *turn_angle_deg*, or by at least
      20° onto a branch while another exit goes straighter, or is a fork to
      that side (a U-turn is LEFT), else STRAIGHT. Without a maneuver ahead
      it is STRAIGHT, or UNKNOWN if the route ends within the window and the
      road beyond its end reaches a junction with a choice there. Bends
      without a choice and roundabout exits straight on are STRAIGHT.

    Attributes:
        name: Its name (stored with any file made with it).
        kind: ``"lateral"``, ``"future_pose"`` or ``"maneuver"``.
        lookahead_m: Distance along the path (lateral) or the maneuver window (maneuver).
        horizon_s: Time ahead (future_pose).
        threshold_m: Lateral offset that makes a turn (lateral, future_pose).
        from_projection: Lateral: measure the offset from the vehicle's
            projection onto the path (NAVSIM; the OSM centre line lies 1.5-5 m
            beside the vehicle, so measuring from the vehicle itself turns
            straight into left), else from the vehicle.
        turn_angle_deg: Maneuver: the turn angle from which it is LEFT / RIGHT.
        unknown_beyond_end: UNKNOWN when the path ends within the lookahead
            (else the end of the path is used, or STRAIGHT for maneuvers).
        version: Bumped when the computation of a preset changes.
    """

    name: str
    kind: Literal["lateral", "future_pose", "maneuver"]
    lookahead_m: Optional[float] = None
    horizon_s: Optional[float] = None
    threshold_m: float = 2.0
    from_projection: bool = True
    turn_angle_deg: float = 35.0
    unknown_beyond_end: bool = True
    version: int = 1

    def __post_init__(self) -> None:
        if self.kind not in ("lateral", "future_pose", "maneuver"):
            raise ValueError(f"kind must be 'lateral', 'future_pose' or 'maneuver', got {self.kind!r}")
        if self.kind in ("lateral", "maneuver") and not (self.lookahead_m and self.lookahead_m > 0):
            raise ValueError(f"a {self.kind} rule needs lookahead_m > 0")
        if self.kind == "future_pose" and not (self.horizon_s and self.horizon_s > 0):
            raise ValueError("a future_pose rule needs horizon_s > 0")

    @property
    def needs_route(self) -> bool:
        """Whether it needs an SD route (maneuver rules)."""
        return self.kind == "maneuver"


#: NAVSIM / OpenScene: the path point 20 m ahead, its lateral offset from the projection, ±2 m.
NAVSIM = CommandRule("navsim", "lateral", lookahead_m=20.0, threshold_m=2.0, from_projection=True)
#: nuScenes planners (VAD, UniAD, ST-P3): the recorded pose 3 s ahead, ±2 m lateral.
NUSCENES = CommandRule("nuscenes", "future_pose", horizon_s=3.0, threshold_m=2.0)
#: Ours, junction-aware: the next SD-route maneuver within 40 m, turns from 35°.
KITSCENES = CommandRule("kitscenes", "maneuver", lookahead_m=40.0, turn_angle_deg=35.0)
#: The presets by name.
PRESETS: dict[str, CommandRule] = {r.name: r for r in (NAVSIM, NUSCENES, KITSCENES)}


def _lateral(dxy: np.ndarray, yaw: float) -> float:
    return float(-math.sin(yaw) * dxy[0] + math.cos(yaw) * dxy[1])


def _from_offset(offset: float, threshold: float) -> Command:
    if offset >= threshold:
        return Command.LEFT
    if offset <= -threshold:
        return Command.RIGHT
    return Command.STRAIGHT


def path_command(points: np.ndarray, s: np.ndarray, s0: float, xy: np.ndarray, yaw: float,
                 rule: CommandRule) -> Command:
    """A lateral rule on a polyline *points* (M, 2) with arc lengths *s*, for a vehicle at *xy*, *yaw*
    whose projection onto it is at arc length *s0*."""
    from kitscenes.sdmap._geometry import point_at

    if rule.kind != "lateral":
        raise ValueError(f"path_command takes a lateral rule, got {rule.kind!r}")
    target_s = s0 + rule.lookahead_m
    if target_s > s[-1] + 1e-6 and rule.unknown_beyond_end:
        return Command.UNKNOWN
    target, origin = point_at(points, s, [min(target_s, s[-1]), s0])
    if not rule.from_projection:
        origin = np.asarray(xy, float)[:2]
    return _from_offset(_lateral(target - origin, yaw), rule.threshold_m)


def route_command(route: "SDRoute", pose, rule: CommandRule, *, near_s_m: Optional[float] = None) -> Command:
    """The command at a pose (EgoPose or (x, y, yaw)) on an SD route by a lateral or maneuver rule."""
    from kitscenes.sdmap.route import _xy_yaw

    xy, yaw = _xy_yaw(pose)
    if yaw is None:
        raise ValueError("a command needs the vehicle's heading: pass an EgoPose or (x, y, yaw)")
    s0 = route.project(xy, near_s_m=near_s_m)
    if rule.kind == "lateral":
        return path_command(route.points, route.s_m, s0, xy, yaw, rule)
    if rule.kind != "maneuver":
        raise ValueError(f"a route gives lateral and maneuver commands, not {rule.kind!r} (use driving_commands)")
    nxt = route.next_maneuver(xy, near_s_m=s0)
    if nxt is not None and nxt[1] <= rule.lookahead_m:
        return maneuver_command(nxt[0], rule)
    if s0 + rule.lookahead_m > route.length_m + route.free_run_m and rule.unknown_beyond_end:
        return Command.UNKNOWN
    return Command.STRAIGHT


def maneuver_command(maneuver: "SDManeuver", rule: CommandRule) -> Command:
    """The command of a maneuver by a maneuver rule: its side if it turns by ``turn_angle_deg`` or is a fork."""
    if maneuver.kind == "fork":
        return Command.LEFT if maneuver.direction.is_left else Command.RIGHT
    if maneuver.kind == "u_turn" or maneuver.direction.value == "uturn":
        return Command.LEFT  # right-hand traffic
    limit = math.radians(rule.turn_angle_deg)
    if maneuver.straightest is False:  # onto a branch while another one goes straighter: "keep left / right"
        limit = min(limit, math.radians(20.0))
    if maneuver.angle_rad >= limit:
        return Command.LEFT
    if maneuver.angle_rad <= -limit:
        return Command.RIGHT
    return Command.STRAIGHT


# ----------------------------------------------------------------------
# Scenes and frames
# ----------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FramePoses:
    """The pose of every frame of a scene (interpolated at the frame times).

    Attributes:
        valid: (F,) frames with a pose.
        xyz: (F, 3) positions (NaN without a pose), in the scene's world frame.
        R: (F, 3, 3) rotations ``R_world_base`` (NaN without).
        yaw: (F,) headings.
        s_m: (F,) arc length of the recorded path at each frame (NaN without).
    """

    __repr__ = compact_repr

    valid: np.ndarray
    xyz: np.ndarray
    R: np.ndarray
    yaw: np.ndarray
    s_m: np.ndarray


def frame_poses(scene: "Scene") -> FramePoses:
    """:class:`FramePoses` of *scene*."""
    n = scene.num_frames
    valid = np.array([scene.availability("pose", i) is Availability.PRESENT for i in range(n)], dtype=bool)
    xyz = np.full((n, 3), np.nan)
    R = np.full((n, 3, 3), np.nan)
    if valid.any():
        rot, trans = scene._trajectory[0].interpolate_matrices(scene.timestamps_ns[valid] / 1e9)
        xyz[valid], R[valid] = trans, rot
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    s = np.full(n, np.nan)
    if valid.any():
        p = xyz[valid, :2]
        s[valid] = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    return FramePoses(valid, xyz, R, yaw, s)


def _resolve_source(rule: CommandRule, source: Source) -> str:
    if source not in ("auto", "path", "sd_route"):
        raise ValueError(f"source must be 'auto', 'path' or 'sd_route', got {source!r}")
    if source == "auto":
        return "sd_route" if rule.needs_route else "path"
    if rule.needs_route and source != "sd_route":
        raise ValueError(f"rule {rule.name!r} needs an SD route: source='sd_route'")
    return source


def driving_commands(scene: "Scene", rule: CommandRule, *, source: Source = "auto",
                     route: Optional["SDRoute"] = None) -> np.ndarray:
    """The command of every frame of *scene* by *rule*: (F,) int8 :class:`Command` values.

    Frames without a pose, and with ``source="sd_route"`` frames more than
    15 m from the route, are UNKNOWN.

    Args:
        rule: A :class:`CommandRule` (:data:`NAVSIM`, :data:`NUSCENES`, :data:`KITSCENES`, or your own).
        source: ``"path"`` (the recorded path), ``"sd_route"`` (*route*, by
            default the driven SD route :attr:`Scene.sd_route`) or ``"auto"``
            (the SD route for maneuver rules, else the recorded path).
            Ignored by future-pose rules.
        route: The SD route for ``source="sd_route"``.

    Raises:
        MapUnavailable: For an SD route in a scene that cannot be placed on
            the SD map (test-e2e), see :attr:`Scene.sd_map`.
        ValueError: A maneuver rule with ``source="path"``.
    """
    fp = frame_poses(scene)
    n = len(fp.valid)
    out = np.full(n, int(Command.UNKNOWN), dtype=np.int8)
    if rule.kind == "future_pose":
        trajectory = scene._trajectory[0]
        for i in np.flatnonzero(fp.valid):
            t = scene.timestamps_ns[i] / 1e9 + rule.horizon_s
            if t > trajectory.times[-1] + 1e-6:
                continue
            _, trans = trajectory.interpolate_matrices(np.array([t]))
            y = (fp.R[i].T @ (trans[0] - fp.xyz[i]))[1]
            out[i] = _from_offset(float(y), rule.threshold_m)
        return out
    src = _resolve_source(rule, source)
    if src == "path":
        p = fp.xyz[fp.valid, :2]
        s = fp.s_m[fp.valid]
        for k, i in enumerate(np.flatnonzero(fp.valid)):
            out[i] = path_command(p, s, float(s[k]), p[k], float(fp.yaw[i]), rule)
        return out
    route = route if route is not None else scene.sd_route
    if route is None:
        return out
    near = None
    for i in np.flatnonzero(fp.valid):
        xy = fp.xyz[i, :2]
        s0 = route.project(xy, near_s_m=near)
        if np.hypot(*(route.point_at(s0) - xy)) > 15.0:
            continue
        near = s0
        out[i] = route_command(route, (xy[0], xy[1], float(fp.yaw[i])), rule, near_s_m=s0)
    return out


@dataclasses.dataclass(frozen=True)
class Navigation:
    """The navigation input of one frame (:meth:`kitscenes.Frame.navigation`).

    Attributes:
        command: The :class:`Command`.
        rule: Name of the :class:`CommandRule`.
        source: Where the command comes from, ``"path"`` or ``"sd_route"``.
        targets: ``"goal"`` or ``"route"`` (see :meth:`kitscenes.Frame.navigation`).
        target_points: (K, 2) x, y in the base frame (x forward, y left).
            ``"goal"``: one point, the end of the scene's route, the same for
            every frame of the scene (:attr:`goal`: ``"sd_route"`` the route's
            end on the OSM road centre line, ``"path"`` the last recorded
            position); empty where it is unknown.
            ``"route"``: route points ahead, see
            :meth:`kitscenes.sdmap.SDRoute.target_points` (``"path"``: the
            recorded path point at the lookahead, or every *spacing_m*).
        route_ego: (M, 2) the route (or path) ahead, up to the last target point, base frame;
            only with ``targets="route"``, else None.
        maneuver: The next decision maneuver of the SD route (only with ``targets="route"`` and ``"sd_route"``).
        distance_to_maneuver_m: Distance to it along the route (negative inside it).
        goal: Where the goal point comes from, ``"sd_route"`` or ``"path"`` (``targets="goal"``), else None.
    """

    __repr__ = compact_repr

    command: Command
    rule: str
    source: str
    targets: str
    target_points: np.ndarray
    route_ego: Optional[np.ndarray] = None
    maneuver: Optional["SDManeuver"] = None
    distance_to_maneuver_m: Optional[float] = None
    goal: Optional[str] = None


def frame_navigation(frame: "Frame", rule: CommandRule, *, source: Source = "auto", targets: Targets = "goal",
                     goal: Goal = "sd_route", max_distance_m: float = 50.0,
                     spacing_m: Optional[float] = None) -> Navigation:
    """:meth:`kitscenes.Frame.navigation`."""
    if targets not in ("goal", "route"):
        raise ValueError(f"targets must be 'goal' or 'route', not {targets!r}")
    if goal not in ("sd_route", "path"):
        raise ValueError(f"goal must be 'sd_route' or 'path', not {goal!r}")
    from kitscenes.sdmap._geometry import arc_length, point_at
    from kitscenes.sdmap.elements import T_base_world

    scene = frame.scene
    pose = frame.pose  # PoseUnavailable without one
    cmd = Command(int(driving_commands_at(scene, frame.index, rule, source)))
    src = _resolve_source(rule, source) if rule.kind != "future_pose" else (
        "path" if source in ("auto", "path") else "sd_route")
    T = T_base_world(pose)
    if targets == "goal":
        # The goal is the same for every rule: by default the end of the driven SD route (route level,
        # like a navigation destination), with goal="path" the exact last recorded position.
        if goal == "sd_route":
            try:
                route = scene.sd_route
            except MapUnavailable as err:  # a split that withholds the map frame (test-e2e): no goal, as without a route
                if err.state != Availability.WITHHELD_BY_SPLIT:
                    raise
                route = None
            end = np.zeros((0, 2)) if route is None else route.points[-1:, :2]
        else:
            fp = frame_poses(scene)
            end = fp.xyz[fp.valid, :2][-1:]
        return Navigation(cmd, rule.name, src, targets, end @ T[:2, :2].T + T[:2, 3] if len(end) else end,
                          goal=goal)
    if src == "sd_route":
        route = scene.sd_route
        if route is None:
            return Navigation(cmd, rule.name, src, targets, np.zeros((0, 2)))
        s0 = route.project(pose)
        points = route.target_points(pose, max_distance_m=max_distance_m, spacing_m=spacing_m, near_s_m=s0)
        ahead = route.points[(route.s_m > s0) & (route.s_m < s0 + max_distance_m)]
        pts = np.concatenate([route.point_at([s0]), ahead, route.point_at([min(s0 + max_distance_m, route.length_m)])])
        nxt = route.next_maneuver(pose, near_s_m=s0)
        return Navigation(cmd, rule.name, src, "route", points, pts @ T[:2, :2].T + T[:2, 3],
                          None if nxt is None else nxt[0], None if nxt is None else float(nxt[1]))
    fp = frame_poses(scene)
    p = fp.xyz[fp.valid, :2]
    s = arc_length(p)
    k = int(np.flatnonzero(np.flatnonzero(fp.valid) == frame.index)[0])
    end = min(s[k] + max_distance_m, s[-1])
    at = np.arange(s[k] + spacing_m, end + 1e-9, spacing_m) if spacing_m else np.array([end])
    points = point_at(p, s, at) if end > s[k] else np.zeros((0, 2))
    inside = p[(s > s[k]) & (s < end)]
    pts = np.concatenate([p[k:k + 1], inside, point_at(p, s, [end])])
    return Navigation(cmd, rule.name, src, "route", points @ T[:2, :2].T + T[:2, 3] if len(points) else points,
                      pts @ T[:2, :2].T + T[:2, 3])


def driving_commands_at(scene: "Scene", index: int, rule: CommandRule, source: Source = "auto") -> int:
    """The command of one frame (cached per scene and rule: the whole scene is computed once)."""
    key = ("_driving_commands", rule, source)
    cache = scene.__dict__.setdefault("_navigation_cache", {})
    if key not in cache:
        cache[key] = driving_commands(scene, rule, source=source)
    return int(cache[key][index])


__all__ = [
    "Command", "CommandRule", "NAVSIM", "NUSCENES", "KITSCENES", "PRESETS", "driving_commands", "Navigation",
]
