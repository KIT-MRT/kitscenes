"""SD-map elements as plain numpy data, so they pickle and outlive the map.

Frames: points are in the **map frame** of the :class:`~kitscenes.sdmap.SDMap`
(the local UTM frame around its origin, the frame of ``poses.txt``) unless a
``frame`` attribute or an ``in_ego`` call says otherwise. OpenStreetMap has no
heights: roads and routes are 2D (N, 2); raw elements (:class:`SDElements`)
are (N, 3) with z NaN in the map frame and at road height in the ego frame.
"""

from __future__ import annotations

import dataclasses
from typing import Mapping, Optional

import numpy as np

from kitscenes._repr import compact_repr
from kitscenes.constants import DEFAULT_ROAD_HEIGHT_M
from kitscenes.sdmap.attributes import RoadClass
from kitscenes.schema import EgoPose


def T_base_world(pose: EgoPose) -> np.ndarray:
    """(4, 4) transform from the world (map) frame into the base frame at *pose*."""
    T = pose.T_world_base
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def to_ego(xy: np.ndarray, pose: EgoPose, road_height_m: float = DEFAULT_ROAD_HEIGHT_M) -> np.ndarray:
    """Map-frame points (N, 2), or (N, 3) with NaN z, into the base frame at *pose*: (N, 3).

    OpenStreetMap has no heights: the points are put on the road, *road_height_m*
    below the base origin at the pose (z of the pose minus the height), then
    transformed with the full pose (pitch and roll included). In the base frame
    the road under the vehicle is then at about ``z = -road_height_m``.
    """
    xy = np.asarray(xy, dtype=np.float64).reshape(len(xy), -1)
    p = np.empty((len(xy), 3))
    p[:, :2] = xy[:, :2]
    z = xy[:, 2] if xy.shape[1] > 2 else np.full(len(xy), np.nan)
    p[:, 2] = np.where(np.isfinite(z), z, float(pose.xyz[2]) - road_height_m)
    T = T_base_world(pose)
    return p @ T[:3, :3].T + T[:3, 3]


@dataclasses.dataclass(frozen=True, eq=False)
class SDRelation:
    """An OpenStreetMap relation (turn restriction, route, multipolygon, ...) of an :class:`SDElements`.

    Attributes:
        id: OSM relation ID.
        tags: All tags.
        member_types: ``"node"``, ``"way"`` or ``"relation"`` per member, in relation order.
        member_ids: OSM IDs of the members.
        member_roles: Their roles (``"from"``, ``"via"``, ``"outer"``, ...; ``""`` if none).
        member_index: Position of each member in the element lists of the
            :class:`SDElements` (``node_ids``, the ways' ``way_ids``, ``relations``),
            -1 if it is not in them. A way clipped into several pieces is listed by its first piece.
    """

    __repr__ = compact_repr

    id: int
    tags: Mapping[str, str]
    member_types: tuple[str, ...]
    member_ids: tuple[int, ...]
    member_roles: tuple[str, ...]
    member_index: tuple[int, ...]


@dataclasses.dataclass(frozen=True, eq=False)
class SDElements:
    """The raw OpenStreetMap elements in a region, with all tags (what SD-prior models such as SDTagNet read).

    Attributes:
        node_ids: (n,) OSM IDs of the nodes with tags in the region (traffic
            signals, crossings, stop signs, ...); untagged nodes are only geometry.
        node_points: (n, 3) their positions; z is NaN in the map frame.
        node_tags: Their tags.
        way_ids: (m,) OSM IDs of the ways in the region, one entry per piece
            (a way cut by the region's border can give several pieces).
        way_points: m (k, 3) polylines, clipped to the region (``clip=True``).
        way_tags: Their tags.
        relations: The relations with a member among these nodes and ways.
        frame: ``"map"`` or ``"ego"`` (the base frame at a pose).
    """

    __repr__ = compact_repr

    node_ids: np.ndarray
    node_points: np.ndarray
    node_tags: list[Mapping[str, str]]
    way_ids: np.ndarray
    way_points: list[np.ndarray]
    way_tags: list[Mapping[str, str]]
    relations: list[SDRelation]
    frame: str = "map"

    def __len__(self) -> int:
        return len(self.node_ids) + len(self.way_ids)

    @property
    def empty(self) -> bool:
        """True if the region holds no node, way or relation (e.g. outside the snapshot's area)."""
        return len(self.node_ids) == 0 and len(self.way_ids) == 0 and not self.relations

    def transform(self, T: np.ndarray, frame: str = "custom", *, z: float = 0.0) -> "SDElements":
        """All points mapped by the (4, 4) transform *T*; points without height get *z* first."""

        def tf(p: np.ndarray) -> np.ndarray:
            q = np.array(p, dtype=np.float64, copy=True).reshape(-1, 3)
            q[~np.isfinite(q[:, 2]), 2] = z
            return q @ T[:3, :3].T + T[:3, 3]

        return dataclasses.replace(self, node_points=tf(self.node_points),
                                   way_points=[tf(p) for p in self.way_points], frame=frame)

    def in_ego(self, pose: EgoPose, *, road_height_m: float = DEFAULT_ROAD_HEIGHT_M) -> "SDElements":
        """The elements in the vehicle's base frame at *pose*, put on the road (see :func:`to_ego`).

        Pass the scene's ``scene.road_height_m`` for the measured height of its vehicle.
        """
        return self.transform(T_base_world(pose), "ego", z=float(pose.xyz[2]) - road_height_m)

    def resampled(self, points_per_way: int) -> "SDElements":
        """Every way piece resampled to *points_per_way* points, evenly spaced along it."""
        from kitscenes.sdmap._geometry import resample

        return dataclasses.replace(self, way_points=[resample(p, points_per_way) for p in self.way_points])


@dataclasses.dataclass(frozen=True, eq=False)
class SDRoad:
    """One drivable direction of a piece of an OpenStreetMap way, between two graph nodes.

    A way is cut where it meets another drivable way; a two-way piece gives
    two roads (``forward`` along the way's node order, and against it).

    Attributes:
        id: Road ID in its map (stable for one snapshot and kitscenes version).
        way_id: OSM way ID.
        forward: Whether the road runs along the way's node order.
        from_node, to_node: OSM node IDs of its ends.
        points: (N, 2) map-frame polyline in driving direction.
        length_m: Its length.
        road_class: The ``highway`` class (links have their parent's class).
        is_link: A ``*_link`` (ramp, slip road).
        one_way: Whether the way can only be driven in one direction.
        public: Whether any car may use it without a reason (``access == "public"``).
        access: ``"public"``, ``"restricted"`` (``private``, ``destination``,
            ``delivery``, ``customers``, ...: routable, not for through
            traffic) or ``"no"`` (closed to cars: ``access=no``, a busway, ...),
            from ``motorcar`` / ``motor_vehicle`` / ``vehicle`` / ``access`` as in OSRM's car profile.
        speed_limit_mps: Speed limit of this direction in m/s (``inf``: none,
            e.g. ``maxspeed=none`` on an Autobahn), None if unknown.
        speed_limit_source: ``"tag"``, ``"implicit"`` (``DE:urban``, ...),
            ``"zone"``, ``"default"`` (class default) or None; see
            :mod:`kitscenes.sdmap.attributes`.
        lanes: Lanes in this direction.
        lanes_source: ``"tag"``, ``"tag_split"`` (half the lanes of a two-way way) or ``"default"``.
        turn_lanes: ``turn:lanes`` of this direction, left to right; None without the tag.
        junction: The ``junction`` tag (``"roundabout"``, ...), None without.
        bridge, tunnel: From the tags.
        layer: The ``layer`` tag (0 without).
        name, ref: Street name and road number, None without.
        successors: Roads a vehicle can continue to at ``to_node``,
            ignoring turn restrictions; no U-turn onto the same piece except at a dead end.
        predecessors: Roads that lead into ``from_node`` the same way.
        reverse_id: The road of the same piece in the other direction, None if one-way.
        tags: All tags of the way.
    """

    __repr__ = compact_repr

    id: int
    way_id: int
    forward: bool
    from_node: int
    to_node: int
    points: np.ndarray
    length_m: float
    road_class: RoadClass
    is_link: bool
    one_way: bool
    public: bool
    access: str
    speed_limit_mps: Optional[float]
    speed_limit_source: Optional[str]
    lanes: int
    lanes_source: str
    turn_lanes: Optional[tuple[frozenset, ...]]
    junction: Optional[str]
    bridge: bool
    tunnel: bool
    layer: int
    name: Optional[str]
    ref: Optional[str]
    successors: tuple[int, ...] = ()
    predecessors: tuple[int, ...] = ()
    reverse_id: Optional[int] = None
    tags: Mapping[str, str] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def speed_limit_kph(self) -> Optional[float]:
        """:attr:`speed_limit_mps` in km/h, as on the signs (None and ``inf`` stay as they are)."""
        return None if self.speed_limit_mps is None else self.speed_limit_mps * 3.6

    @property
    def roundabout(self) -> bool:
        return self.junction in ("roundabout", "circular")

    def in_ego(self, pose: EgoPose, *, road_height_m: float = DEFAULT_ROAD_HEIGHT_M) -> np.ndarray:
        """(N, 3) polyline in the base frame at *pose*, on the road (see :func:`to_ego`)."""
        return to_ego(self.points, pose, road_height_m)


@dataclasses.dataclass(frozen=True, eq=False)
class SDJunction:
    """A junction: graph nodes where three or more road pieces meet, merged across internal pieces shorter
    than the map's ``junction_merge_m`` (one dual-carriageway crossing is one junction).

    Attributes:
        id: Junction ID in its map.
        node_ids: OSM node IDs of its graph nodes.
        xy: (2,) map-frame centre (mean of the nodes).
        control: ``"traffic_signals"``, ``"stop"``, ``"give_way"``,
            ``"roundabout"``, ``"mini_roundabout"`` or ``"none"``, from the
            tags of its nodes and of the roads that enter it (``highway=*``
            nodes on the incoming roads within 30 m count).
        incoming: Roads that end in it from outside.
        outgoing: Roads that leave it to the outside.
        internal: Roads between its nodes.
    """

    __repr__ = compact_repr

    id: int
    node_ids: tuple[int, ...]
    xy: np.ndarray
    control: str
    incoming: tuple[int, ...]
    outgoing: tuple[int, ...]
    internal: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True, eq=False)
class SDTurnRestriction:
    """A turn restriction (``type=restriction`` relation) that applies to cars.

    Attributes:
        id: OSM relation ID.
        kind: The ``restriction`` value: ``"no_left_turn"``, ``"only_straight_on"``, ...
        from_way, to_way: OSM way IDs.
        via_nodes, via_ways: The via member (a node, or one or more ways).
        road_sequences: The road sequences it forbids (``no_*``) or, after all
            but the last road, makes the only way on (``only_*``).
    """

    __repr__ = compact_repr

    id: int
    kind: str
    from_way: int
    to_way: int
    via_nodes: tuple[int, ...]
    via_ways: tuple[int, ...]
    road_sequences: tuple[tuple[int, ...], ...]

    @property
    def only(self) -> bool:
        return self.kind.startswith("only_")


@dataclasses.dataclass(frozen=True)
class SDMatch:
    """A road a vehicle at a pose may be on, see :meth:`SDMap.match`.

    Attributes:
        road_id: The road.
        s_m: Arc length of the closest point on it, from its start.
        distance_m: Distance of the pose to the road (the OSM centre line; a
            vehicle in its lane is typically 1-5 m beside it).
        heading_error: Angle between the heading and the road direction there, radians.
        xy: (2,) the closest point, map frame.
    """

    __repr__ = compact_repr

    road_id: int
    s_m: float
    distance_m: float
    heading_error: float
    xy: np.ndarray


@dataclasses.dataclass(frozen=True, eq=False)
class SDTrack:
    """A trajectory matched to the road graph, see :meth:`SDMap.match_trajectory`.

    Attributes:
        road_ids: (P,) road of each pose, -1 where it is not matched.
        s_m: (P,) arc length of each pose along its route piece (NaN unmatched).
        distance_m: (P,) distance of each pose to the route (NaN unmatched).
        matched: (P,) bool.
        routes: The matched route pieces (:class:`~kitscenes.sdmap.SDRoute`);
            one unless the trajectory leaves the road graph in between.
        route_index: (P,) piece of each pose, -1 unmatched.
    """

    __repr__ = compact_repr

    road_ids: np.ndarray
    s_m: np.ndarray
    distance_m: np.ndarray
    matched: np.ndarray
    routes: tuple
    route_index: np.ndarray

    @property
    def matched_share(self) -> float:
        return float(self.matched.mean()) if len(self.matched) else 0.0
