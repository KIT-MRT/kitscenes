"""An OpenStreetMap SD map in a scene's map frame: raw elements, the road graph, matching and routing.

The file is parsed once per process (and cached on disk, see
:mod:`kitscenes.sdmap._cache`), projected into the frame of the map's origin,
and the road graph is built on first use. An SDMap pickles to its path,
origin and options.
"""

from __future__ import annotations

import logging
import math
import threading
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Union

import numpy as np

from kitscenes.map.origin import MapOrigin
from kitscenes.map.roi import Roi
from kitscenes.schema import EgoPose
from kitscenes.sdmap import _geometry as geo
from kitscenes.sdmap import attributes as attrs
from kitscenes.sdmap._cache import load_osm
from kitscenes.sdmap._graph import JunctionIndex, RoadGraph
from kitscenes.sdmap._parse import MEMBER_TYPES, OSMData, looks_double_encoded, with_repaired_strings
from kitscenes.sdmap.attributes import RoadClass
from kitscenes.sdmap.elements import (
    SDElements, SDJunction, SDMatch, SDRelation, SDRoad, SDTrack, SDTurnRestriction,
)

if TYPE_CHECKING:
    from kitscenes.sdmap.route import SDRoute

logger = logging.getLogger(__name__)

#: The attribution that goes with the data and anything shown or published from it.
ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0"

#: Junction graph nodes closer than this (metres, along a road piece between them) form one junction.
JUNCTION_MERGE_M = 20.0

Position = Union[EgoPose, SDMatch, np.ndarray, Sequence[float]]

# SDMap.route: road candidates tried at the start and the goal, and the costs (seconds) of a U-turn at a graph
# node and of entering a road with restricted access.
_ROUTE_CANDIDATES = 3
_UTURN_PENALTY_S = 20.0
_RESTRICTED_PENALTY_S = 3600.0


class SDMap:
    """An OpenStreetMap SD map (one file of ``openstreetmap_sd_maps/``) in a local UTM frame.

    Use :meth:`kitscenes.Dataset.sd_map` (the city frame, shared by every
    scene of the city that uses the file) or :attr:`kitscenes.Scene.sd_map`. Frames: the **map
    frame** of *origin* (the frame of ``poses.txt``; metres, x east, y
    north), 2D (OpenStreetMap has no heights); ``in_ego`` methods move
    elements into the vehicle's base frame at a pose, on the road.

    Args:
        osm_path: OpenStreetMap PBF or XML.
        origin: The frame origin (a scene's ``maps/origin.json`` or the city's,
            :data:`kitscenes.map.origin.CITY_ORIGINS`).
        city: The city, for messages.
        cache_dir: Where the parsed file is cached (default
            ``$KITSCENES_CACHE/sdmap`` or ``~/.cache/kitscenes/sdmap``).
        junction_merge_m: See :data:`JUNCTION_MERGE_M`.
        repair_encoding: Repair double-encoded text (``"StraÃŸe"`` ->
            ``"Straße"``, see :func:`kitscenes.sdmap._parse.repair_mojibake`):
            ``"auto"`` (default) only in a file whose non-ASCII text is mostly
            that way (the 2026-05-03 snapshots; the weekly snapshots are correct
            UTF-8 and stay as they are), True always, False never (the strings
            as in the file).

    Raises:
        FileNotFoundError: If the file does not exist.
    """

    def __init__(self, osm_path: Path | str, origin: MapOrigin, *, city: Optional[str] = None,
                 cache_dir: Optional[Path | str] = None, junction_merge_m: float = JUNCTION_MERGE_M,
                 repair_encoding: Union[bool, str] = "auto") -> None:
        self.path = Path(osm_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"SD map file not found: {self.path}")
        self.origin = origin
        self.city = city
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.junction_merge_m = float(junction_merge_m)
        if repair_encoding not in (True, False, "auto"):
            raise ValueError(f"repair_encoding must be True, False or 'auto', got {repair_encoding!r}")
        self.repair_encoding = repair_encoding
        self._lock = threading.RLock()

    def __repr__(self) -> str:
        state = f"{self._graph.num_roads} roads" if "_graph" in self.__dict__ else "not loaded"
        return (f"SDMap({str(self.path)!r}, origin=({self.origin.lat:.5f}, {self.origin.lon:.5f}), "
                f"city={self.city!r}, {state})")

    def __getstate__(self) -> dict:
        return {"path": self.path, "origin": self.origin, "city": self.city, "cache_dir": self.cache_dir,
                "junction_merge_m": self.junction_merge_m, "repair_encoding": self.repair_encoding}

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    #: See :data:`ATTRIBUTION`.
    attribution = ATTRIBUTION

    # ------------------------------------------------------------------
    # Loading (once)
    # ------------------------------------------------------------------

    @cached_property
    def _data(self) -> OSMData:
        with self._lock:
            if "_data" in self.__dict__:
                return self.__dict__["_data"]
            data = load_osm(self.path, cache_dir=self.cache_dir)
            repair = self.repair_encoding
            if repair == "auto":
                repair = looks_double_encoded(data.strings)
            self.__dict__["_encoding_repaired"] = bool(repair)
            return with_repaired_strings(data) if repair else data

    @property
    def encoding_repaired(self) -> bool:
        """Whether the text was repaired on load (see *repair_encoding*); loads the file."""
        self._data
        return self.__dict__["_encoding_repaired"]

    @cached_property
    def _node_xy(self) -> np.ndarray:
        d = self._data
        xy = self.origin.from_latlon(d.node_lat, d.node_lon) if d.num_nodes else np.zeros((0, 2))
        xy.flags.writeable = False
        return xy

    @cached_property
    def _graph(self) -> RoadGraph:
        with self._lock:
            if "_graph" in self.__dict__:
                return self.__dict__["_graph"]
            return RoadGraph.build(self._data, self._node_xy)

    @cached_property
    def _junction_index(self) -> JunctionIndex:
        return JunctionIndex.build(self._graph, self.junction_merge_m)

    @cached_property
    def _way_boxes(self) -> np.ndarray:
        d = self._data
        xy = self._node_xy
        nodes = d.way_nodes
        valid = nodes >= 0
        pts = np.where(valid[:, None], xy[np.maximum(nodes, 0)], np.nan)
        out = np.full((d.num_ways, 4), np.nan)
        nonempty = d.way_node_ptr[1:] > d.way_node_ptr[:-1]
        if nonempty.any():
            idx = d.way_node_ptr[:-1][nonempty]
            with np.errstate(invalid="ignore"):
                out[nonempty, :2] = np.fmin.reduceat(pts, idx, axis=0)
                out[nonempty, 2:] = np.fmax.reduceat(pts, idx, axis=0)
        return out

    @cached_property
    def _tagged_nodes(self) -> np.ndarray:
        d = self._data
        return np.flatnonzero(d.node_tag_ptr[1:] > d.node_tag_ptr[:-1])

    @cached_property
    def _relations_of(self) -> dict[tuple[int, int], list[int]]:
        """``(member type, member OSM id) -> relation positions``."""
        d = self._data
        out: dict[tuple[int, int], list[int]] = {}
        owner = np.searchsorted(d.rel_member_ptr, np.arange(len(d.member_ref)), side="right") - 1
        for r, t, ref in zip(owner.tolist(), d.member_type.tolist(), d.member_ref.tolist()):
            lst = out.setdefault((t, ref), [])
            if not lst or lst[-1] != r:
                lst.append(r)
        return out

    @cached_property
    def _roads(self) -> dict[int, SDRoad]:
        return {}

    @property
    def snapshot(self) -> Optional[str]:
        """The OpenStreetMap data state of the file, ISO 8601 UTC: the PBF header's ``osmosis_replication_timestamp``
        of the weekly snapshots (``"2026-02-02T00:00:00Z"``, Monday 00:00 of the recording week) or the Overpass
        ``osm_base`` of the 2026-05-03 files (``"2026-05-03T15:48:06Z"``); None if the file says neither."""
        return self._data.osm_base

    # ------------------------------------------------------------------
    # Raw elements (SD priors)
    # ------------------------------------------------------------------

    def elements(self, roi: Roi, *, ways: str = "all", nodes: str = "tagged", clip: bool = True) -> SDElements:
        """The OpenStreetMap elements in *roi* (map frame), with all tags.

        An empty result (no node, way or relation) is normal outside the
        snapshot's area (:meth:`covers`).

        Args:
            roi: A :class:`~kitscenes.map.Box` (``Region(...).around(pose)``) or
                :class:`~kitscenes.map.Circle` in the map frame.
            ways: ``"all"`` (every way: roads, buildings, land use, ...),
                ``"highway"`` (ways with a ``highway`` tag: roads, paths,
                cycleways) or ``"roads"`` (the drivable classes of
                :data:`kitscenes.sdmap.attributes.DRIVABLE_HIGHWAYS`) or ``"none"``.
            nodes: ``"tagged"`` (nodes with tags: signals, crossings, ...), ``"all"`` or ``"none"``.
            clip: Cut the ways at the region's border (pieces inside); False
                keeps every way that touches it whole.
        """
        if ways not in ("all", "highway", "roads", "none"):
            raise ValueError(f"ways must be 'all', 'highway', 'roads' or 'none', got {ways!r}")
        if nodes not in ("tagged", "all", "none"):
            raise ValueError(f"nodes must be 'tagged', 'all' or 'none', got {nodes!r}")
        d = self._data
        xy = self._node_xy
        way_pos = self._ways_in(roi, ways)
        way_ids, way_points, way_tags, piece_of_way = [], [], [], {}
        for w in way_pos:
            pts = self._way_points(w)
            if len(pts) == 0 or not roi.hits(pts):
                continue
            pieces = geo.clip(pts, roi) if clip else [pts]
            if not pieces:
                continue
            tags = d.tags("way", w)
            piece_of_way.setdefault(int(d.way_ids[w]), len(way_ids))
            for p in pieces:
                way_ids.append(int(d.way_ids[w]))
                way_points.append(_xyz(p))
                way_tags.append(tags)
        node_pos = self._nodes_in(roi, nodes)
        node_index = {int(d.node_ids[n]): k for k, n in enumerate(node_pos)}
        of = self._relations_of
        rel_pos = sorted({r for i in node_index for r in of.get((0, i), ())}
                         | {r for i in piece_of_way for r in of.get((1, i), ())})
        rel_index = {int(d.rel_ids[r]): k for k, r in enumerate(rel_pos)}
        index_of = {"node": node_index, "way": piece_of_way, "relation": rel_index}
        relations = [self._relation(r, index_of) for r in rel_pos]
        return SDElements(
            node_ids=np.array([int(d.node_ids[n]) for n in node_pos], dtype=np.int64),
            node_points=_xyz(xy[np.asarray(node_pos, dtype=np.int64)]) if len(node_pos) else np.zeros((0, 3)),
            node_tags=[d.tags("node", n) for n in node_pos],
            way_ids=np.array(way_ids, dtype=np.int64),
            way_points=way_points,
            way_tags=way_tags,
            relations=relations,
        )

    def _relation(self, r: int, index_of: Optional[dict[str, dict[int, int]]] = None) -> SDRelation:
        d = self._data
        a, b = d.rel_member_ptr[r], d.rel_member_ptr[r + 1]
        types = tuple(MEMBER_TYPES[t] for t in d.member_type[a:b])
        ids = tuple(int(x) for x in d.member_ref[a:b])
        index = (-1,) * len(ids) if index_of is None else tuple(index_of[t].get(i, -1) for t, i in zip(types, ids))
        return SDRelation(int(d.rel_ids[r]), d.tags("relation", r), types, ids,
                          tuple(d.strings[k] for k in d.member_role[a:b]), index)

    def relations(self) -> list[SDRelation]:
        """Every relation of the file with all tags and members, in file order.

        ``member_index`` is -1 for every member: there are no element lists
        to point into (:meth:`elements` gives the relations of a region with
        the positions of their members).
        """
        return list(self._all_relations)

    @cached_property
    def _all_relations(self) -> tuple[SDRelation, ...]:
        return tuple(self._relation(r) for r in range(self._data.num_relations))

    def _way_points(self, w: int) -> np.ndarray:
        d = self._data
        nodes = d.way_nodes[d.way_node_ptr[w]:d.way_node_ptr[w + 1]]
        return self._node_xy[nodes[nodes >= 0]]

    def _ways_in(self, roi: Roi, which: str = "all") -> list[int]:
        if which == "none":
            return []
        boxes = self._way_boxes
        ok = np.isfinite(boxes[:, 0]) & geo.roi_contains_box(roi, boxes[:, :2], boxes[:, 2:])
        cand = np.flatnonzero(ok)
        if which != "all" and len(cand):
            hw = self._highway_of_way
            if which == "highway":
                cand = cand[hw[cand] >= 0]
            else:
                cand = cand[self._drivable_way[cand]]
        return [int(w) for w in cand]

    @cached_property
    def _highway_of_way(self) -> np.ndarray:
        return self._data.tag_value_index("way", "highway")

    @cached_property
    def _drivable_way(self) -> np.ndarray:
        hw = self._highway_of_way
        good = np.array([s in attrs.DRIVABLE_HIGHWAYS for s in self._data.strings] + [False], dtype=bool)
        return good[np.where(hw >= 0, hw, len(self._data.strings))]

    def _nodes_in(self, roi: Roi, which: str) -> list[int]:
        if which == "none":
            return []
        xy = self._node_xy
        lo, hi = geo.bounding_box(roi)
        cand = self._tagged_nodes if which == "tagged" else np.arange(len(xy))
        p = xy[cand]
        box = (p[:, 0] >= lo[0]) & (p[:, 0] <= hi[0]) & (p[:, 1] >= lo[1]) & (p[:, 1] <= hi[1])
        cand = cand[box]
        return [int(n) for n in cand[geo.contains(roi, xy[cand])]] if len(cand) else []

    def covers(self, xy: np.ndarray, margin_m: float = 0.0) -> np.ndarray:
        """(N,) whether map-frame points lie inside the snapshot's area, at least *margin_m* from its border.

        The KITScenes snapshots are bounding-box cuts (Overpass for the
        2026-05-03 files, the same selection with osmium for the weekly ones):
        complete inside the box, with only the ways that cross its border
        reaching out. The box is estimated as the extent of the nodes that
        belong to no way and no relation (the "free" nodes, which the
        bounding-box selection keeps only inside the box). Outside it, the
        map is empty or partial.
        """
        lat_lo, lat_hi, lon_lo, lon_hi = self.coverage_latlon
        ll = self.origin.to_latlon(np.atleast_2d(np.asarray(xy, float))[:, :2])
        # margin in degrees (metres -> degrees, at the points' latitude)
        dlat = margin_m / 111_320.0
        dlon = margin_m / (111_320.0 * np.cos(np.radians(ll[:, 0])))
        return ((ll[:, 0] >= lat_lo + dlat) & (ll[:, 0] <= lat_hi - dlat)
                & (ll[:, 1] >= lon_lo + dlon) & (ll[:, 1] <= lon_hi - dlon))

    @cached_property
    def coverage_latlon(self) -> tuple[float, float, float, float]:
        """``(lat_min, lat_max, lon_min, lon_max)`` of the snapshot's area, see :meth:`covers`."""
        d = self._data
        in_way = np.zeros(d.num_nodes, dtype=bool)
        in_way[d.way_nodes[d.way_nodes >= 0]] = True
        node_members = d.member_ref[d.member_type == 0]
        free = ~in_way & ~np.isin(d.node_ids, node_members)
        if not free.any():
            free = np.ones(d.num_nodes, dtype=bool)
        lat, lon = d.node_lat[free], d.node_lon[free]
        return float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())

    # ------------------------------------------------------------------
    # The road graph
    # ------------------------------------------------------------------

    def road(self, road_id: int) -> SDRoad:
        """The road with this ID."""
        road_id = int(road_id)
        cached = self._roads.get(road_id)
        if cached is not None:
            return cached
        g = self._graph
        if not 0 <= road_id < g.num_roads:
            raise KeyError(f"No road {road_id} in {self.path.name} ({g.num_roads} roads)")
        d = self._data
        w = int(g.way_index[g.road_edge[road_id]])
        tags = d.tags("way", w)
        forward = bool(g.road_forward[road_id])
        cls, is_link = attrs.road_class(tags["highway"])
        fwd, bwd = attrs.directions(tags)
        speed, speed_source = attrs.speed_limit(tags, forward)
        lanes, lanes_source = attrs.lanes(tags, forward)
        rev = g.reverse(road_id)
        try:
            layer = int(tags.get("layer", "0"))
        except ValueError:
            layer = 0
        points = g.road_points(road_id)
        points.flags.writeable = False
        road = SDRoad(
            id=road_id, way_id=int(d.way_ids[w]), forward=forward,
            from_node=int(d.node_ids[g.road_from[road_id]]), to_node=int(d.node_ids[g.road_to[road_id]]),
            points=points, length_m=float(g.road_length[road_id]), road_class=cls, is_link=is_link,
            one_way=fwd != bwd, public=bool(g.road_public[road_id]),
            access=attrs.ACCESS_LEVELS[int(g.road_access[road_id])],
            speed_limit_mps=None if speed is None else speed / 3.6, speed_limit_source=speed_source, lanes=lanes, lanes_source=lanes_source,
            turn_lanes=attrs.turn_lanes(tags, forward), junction=tags.get("junction"),
            bridge=tags.get("bridge", "no") not in ("no",), tunnel=tags.get("tunnel", "no") not in ("no",),
            layer=layer, name=tags.get("name"), ref=tags.get("ref"),
            successors=tuple(g.successors(road_id)), predecessors=tuple(g.predecessors(road_id)),
            reverse_id=None if rev < 0 else rev, tags=tags,
        )
        self._roads[road_id] = road
        return road

    @property
    def num_roads(self) -> int:
        return self._graph.num_roads

    def roads(self, roi: Optional[Roi] = None, classes: Optional[Iterable[Union[RoadClass, str]]] = None, *,
              public_only: bool = False, include_links: bool = True) -> list[SDRoad]:
        """Roads touching *roi* (all if None), optionally of the given classes, sorted by ID.

        An empty list is normal outside the snapshot's area (:meth:`covers`).
        """
        wanted = None if classes is None else {RoadClass(c) for c in classes}
        g = self._graph
        if roi is None:
            cand = range(g.num_roads)
        else:
            boxes = g.edge_boxes()
            edges = np.flatnonzero(geo.roi_contains_box(roi, boxes[:, :2], boxes[:, 2:]))
            cand = sorted(int(r) for e in edges for r in g.edge_road[e] if r >= 0)
        out = []
        for r in cand:
            if public_only and not g.road_public[r]:
                continue
            road = self.road(r)
            if wanted is not None and road.road_class not in wanted:
                continue
            if not include_links and road.is_link:
                continue
            if roi is not None and not roi.hits(road.points):
                continue
            out.append(road)
        return out

    def restrictions(self, roi: Optional[Roi] = None) -> list[SDTurnRestriction]:
        """The turn restrictions for cars (with their road sequences) whose roads touch *roi* (all if None)."""
        out = []
        for res in self._graph.restrictions:
            if roi is not None and not any(roi.hits(self._graph.road_points(r)) for seq in res.sequences for r in seq):
                continue
            out.append(SDTurnRestriction(res.id, res.kind, res.from_way, res.to_way, res.via_nodes, res.via_ways,
                                         res.sequences))
        return out

    def junction(self, junction_id: int) -> SDJunction:
        """The junction with this ID."""
        return self._junctions[int(junction_id)]

    def junction_at_node(self, node: int) -> Optional[int]:
        """The junction of a graph node (a position in the parsed nodes), None if it is none."""
        return self._junction_index.cluster_of.get(int(node))

    def junctions(self, roi: Optional[Roi] = None) -> list[SDJunction]:
        """Junctions whose centre is in *roi* (all if None)."""
        js = self._junctions
        if roi is None:
            return list(js)
        if not js:
            return []
        centres = np.array([j.xy for j in js])
        return [js[i] for i in np.flatnonzero(geo.contains(roi, centres))]

    @cached_property
    def _junctions(self) -> list[SDJunction]:
        g, jx, d = self._graph, self._junction_index, self._data
        control_idx = d.tag_value_index("node", "highway")
        strings = d.strings
        out = []
        for cid, members in enumerate(jx.members):
            ms = set(members)
            incoming = sorted(int(r) for n in members for r in g.entering(n) if int(g.road_from[r]) not in ms)
            outgoing = sorted(int(r) for n in members for r in g.leaving(n) if int(g.road_to[r]) not in ms)
            internal = sorted(r for n in members for r in g.leaving(n) if int(r) in jx.internal_roads)
            control = self._control(members, incoming, control_idx, strings)
            out.append(SDJunction(cid, tuple(int(d.node_ids[n]) for n in members),
                                  self._node_xy[members].mean(axis=0), control,
                                  tuple(incoming), tuple(outgoing), tuple(int(r) for r in internal)))
        return out

    def _control(self, members, incoming, control_idx, strings) -> str:
        g = self._graph
        values = set()
        for n in members:
            if control_idx[n] >= 0:
                values.add(strings[control_idx[n]])
        for r in incoming:
            nodes = g.road_nodes(r)
            p = self._node_xy[nodes]
            s = geo.arc_length(p)
            near = nodes[s >= s[-1] - 30.0]
            for n in near:
                if control_idx[n] >= 0:
                    values.add(strings[control_idx[n]])
        if any(self.road(r).roundabout for r in incoming):
            return "roundabout"
        for value in ("traffic_signals", "stop", "give_way", "mini_roundabout"):
            if value in values:
                return value
        return "none"

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, pose: Position, *, max_distance: float = 10.0, max_heading_error: float = math.pi / 4,
              public_only: bool = False) -> list[SDMatch]:
        """Roads a vehicle at the pose may be driving on, best first.

        All roads within *max_distance* (to the OSM centre line; a vehicle in
        its lane is typically 1-5 m beside it) whose direction differs from
        the heading by at most *max_heading_error* (one-way roads only in their
        direction). Sorted by distance plus 5 m per radian of heading error.

        Args:
            pose: An :class:`~kitscenes.EgoPose`, or ``(x, y)`` / ``(x, y, yaw)`` in the map frame.
        """
        xy, yaw = _xy_yaw(pose)
        out = []
        for r, s, dist, err in self._graph.snap(xy, yaw, max_distance, max_heading_error, public_only):
            out.append(SDMatch(r, s, dist, err, geo.point_at(self._graph.road_points(r),
                                                             geo.arc_length(self._graph.road_points(r)), [s])[0]))
        return out

    def attributes_at(self, pose: Position, **kwargs) -> Optional[SDRoad]:
        """The road of the best :meth:`match` (its speed limit, lanes, class, ...), None without a match."""
        matches = self.match(pose, **kwargs)
        return self.road(matches[0].road_id) if matches else None

    def match_trajectory(self, xy: np.ndarray, yaw: Optional[np.ndarray] = None, *, sigma_m: float = 4.0,
                         max_distance: float = 20.0, max_heading_error: float = math.pi / 2, step_m: float = 5.0,
                         beta_m: float = 5.0, public_only: bool = False) -> SDTrack:
        """A trajectory matched to the road graph with a hidden Markov model (Viterbi).

        The poses at least *step_m* apart are matched: candidates are the
        roads within *max_distance* whose direction is within
        *max_heading_error* of *yaw* (if given); the emission is Gaussian in
        the distance (*sigma_m*) and the heading error; the transition prefers
        driving distances close to the straight distance between the poses
        (*beta_m*, Newson & Krumm). Turn restrictions are not applied (the
        recorded drive is what happened). Every pose is then projected onto
        the matched route.

        Args:
            xy: (P, 2+) map-frame positions in time order.
            yaw: (P,) headings in radians, or None.

        Returns:
            The :class:`SDTrack`; poses outside the road graph (parking lots,
            private ground, outside the snapshot) are not matched, and the
            trajectory is split into route pieces there.
        """
        from kitscenes.sdmap import _match
        from kitscenes.sdmap.route import SDRoute

        xy = np.asarray(xy, dtype=np.float64)[:, :2]
        yaw = None if yaw is None else np.asarray(yaw, dtype=np.float64)
        P = len(xy)
        road_ids = np.full(P, -1, dtype=np.int64)
        s_m = np.full(P, np.nan)
        dist = np.full(P, np.nan)
        route_index = np.full(P, -1, dtype=np.int64)
        routes = []
        if P:
            keys = _match.key_poses(xy, step_m)
            pieces = _match.viterbi(self._graph, xy[keys], None if yaw is None else yaw[keys], sigma_m=sigma_m,
                                    max_distance=max_distance, max_heading_error=max_heading_error, beta_m=beta_m,
                                    public_only=public_only)
            for pose_idx, positions in pieces:
                joined = _match.connect(self._graph, positions)
                if joined is None:
                    continue
                roads, s0, s1 = joined
                if len(roads) == 1 and s1 < s0:
                    s0, s1 = s1, s0
                route = SDRoute.from_roads(self, roads, s0, s1, engine="matched")
                first, last = int(keys[pose_idx[0]]), int(keys[pose_idx[-1]])
                k = len(routes)
                routes.append(route)
                near = 0.0
                for i in range(first, last + 1):
                    s = route.project(xy[i], near_s_m=near, window_m=60.0)
                    near = s
                    s_m[i] = s
                    dist[i] = float(np.hypot(*(route.point_at(s) - xy[i])))
                    road_ids[i] = route.road_at(s)
                    route_index[i] = k
        matched = route_index >= 0
        return SDTrack(road_ids, s_m, dist, matched, tuple(routes), route_index)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, start: Position, goal: Position, *, via: Sequence[Position] = (), restrictions: bool = True,
              weight: str = "length", max_distance: float = 15.0,
              max_heading_error: float = math.pi / 4) -> Optional["SDRoute"]:
        """The shortest route from *start* over *via* to *goal* (A* on the road graph), None if there is none.

        Positions are :class:`~kitscenes.EgoPose`, :class:`SDMatch`, or
        ``(x, y)`` / ``(x, y, yaw)`` in the map frame; a heading picks the
        roads in its direction (within *max_heading_error*, at most
        *max_distance* away). The best three road candidates are tried at
        the start and at the goal; the cheapest route wins, each candidate's
        distance to its position counted twice on top. Turn restrictions
        (``type=restriction`` relations for cars) are applied by default.

        Costs follow OSRM's car profile where it matters for short urban
        routes: a U-turn at a junction costs 20 s; roads with restricted
        access (``private``, ``destination``, ``delivery``, ...) cost an hour
        to enter from a public road, so they are used only where the route
        starts, ends or has no other way; roads closed to cars
        (``access=no``, busways) are used only to end on. With
        ``weight="length"`` the penalties count at 13.9 m/s (50 km/h).

        Args:
            weight: ``"length"`` (metres) or ``"time"`` (at the speed limits).
        """
        from kitscenes.sdmap.route import SDRoute

        g = self._graph
        scale = 13.9 if weight == "length" else 1.0  # seconds -> metres at 50 km/h
        positions = [start, *via, goal]
        snapped = []
        for k, p in enumerate(positions):
            if isinstance(p, SDMatch):
                snapped.append([p])
                continue
            m = self.match(p, max_distance=max_distance, max_heading_error=max_heading_error)
            if not m:
                return None
            snapped.append(m[:_ROUTE_CANDIDATES if k in (0, len(positions) - 1) else 1])
        roads: list[int] = []
        start_s = end_s = 0.0
        for leg in range(len(snapped) - 1):
            best = None
            starts = snapped[leg] if leg == 0 else [snapped[leg][0]]
            for a in starts:
                for b in snapped[leg + 1]:
                    found = g.shortest_path(a.road_id, a.s_m, b.road_id, b.s_m, weight=weight,
                                            restrictions=restrictions, uturn_penalty=_UTURN_PENALTY_S * scale,
                                            access_penalty=_RESTRICTED_PENALTY_S * scale)
                    if found is None:
                        continue
                    penalty = 2.0 * (a.distance_m + b.distance_m)
                    if weight == "time":
                        penalty /= 13.9
                    score = found[0] + penalty
                    if best is None or score < best[0]:
                        best = (score, found[1], a, b)
            if best is None:
                return None
            _, path, a, b = best
            if leg == 0:
                start_s = a.s_m
                roads.extend(path)
            else:
                roads.extend(path[1:] if roads and path[0] == roads[-1] else path)
            snapped[leg + 1] = [b]
            end_s = b.s_m
        return SDRoute.from_roads(self, roads, start_s, end_s)

    def route_along(self, xy: np.ndarray, yaw: Optional[np.ndarray] = None, **kwargs) -> Optional["SDRoute"]:
        """The route a trajectory drove: the longest piece of :meth:`match_trajectory`, None if nothing matches."""
        track = self.match_trajectory(xy, yaw, **kwargs)
        if not track.routes:
            return None
        counts = np.bincount(track.route_index[track.matched], minlength=len(track.routes))
        return track.routes[int(np.argmax(counts))]

    # ------------------------------------------------------------------
    # Interop
    # ------------------------------------------------------------------

    def to_networkx(self, roi: Optional[Roi] = None):
        """The road graph as a ``networkx.DiGraph`` (needs networkx): nodes are OSM node IDs with ``xy``,
        edges are roads with ``road_id``, ``way_id`` and ``length_m``."""
        from kitscenes._optional import require

        nx = require("networkx", "sdmap", purpose="SDMap.to_networkx")
        G = nx.DiGraph()
        for road in self.roads(roi):
            G.add_node(road.from_node, xy=road.points[0])
            G.add_node(road.to_node, xy=road.points[-1])
            G.add_edge(road.from_node, road.to_node, road_id=road.id, way_id=road.way_id, length_m=road.length_m)
        return G


def _xyz(p: np.ndarray) -> np.ndarray:
    out = np.full((len(p), 3), np.nan)
    out[:, :2] = p[:, :2]
    return out


def _xy_yaw(pose: Position) -> tuple[np.ndarray, Optional[float]]:
    if isinstance(pose, EgoPose):
        return np.asarray(pose.xyz[:2], float), pose.yaw
    if isinstance(pose, SDMatch):
        return np.asarray(pose.xy, float), None
    p = np.asarray(pose, dtype=np.float64).reshape(-1)
    return p[:2], (float(p[2]) if len(p) > 2 else None)


_shared_lock = threading.Lock()
_shared: dict[tuple, SDMap] = {}


def shared_sd_map(osm_path: Path | str, origin: MapOrigin, city: Optional[str] = None) -> SDMap:
    """The process-wide :class:`SDMap` of a file in a frame (what ``ds.sd_map`` and ``scene.sd_map`` return),
    so every scene of a city shares one parsed map and graph."""
    path = Path(osm_path)
    key = (str(path.resolve()), float(origin.lat), float(origin.lon))
    with _shared_lock:
        smap = _shared.get(key)
        if smap is None:
            smap = _shared[key] = SDMap(path, origin, city=city)
    return smap


def city_of_origin(origin: MapOrigin) -> Optional[str]:
    """The city whose origin (:data:`kitscenes.map.origin.CITY_ORIGINS`) this is, None for another one."""
    from kitscenes.map.origin import CITY_ORIGINS

    for city, o in CITY_ORIGINS.items():
        if abs(o.lat - origin.lat) < 1e-9 and abs(o.lon - origin.lon) < 1e-9:
            return city
    return None
