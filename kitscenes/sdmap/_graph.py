"""The directed road graph of an SD map, with turn restrictions, junctions, routing and snapping.

Internal: :class:`kitscenes.sdmap.SDMap` builds one :class:`RoadGraph` per map
and frame. Terms:

- **graph node**: an OSM node where drivable ways meet or end;
- **edge**: an undirected piece of a way between two graph nodes;
- **road** (:class:`kitscenes.sdmap.SDRoad`): one drivable direction of an
  edge. Road IDs are positions in the road arrays: stable for one file and
  parser version, not across snapshots (the OSM way ID is in ``way_id``).
"""

from __future__ import annotations

import heapq
import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from kitscenes.sdmap import attributes as attrs
from kitscenes.sdmap._parse import MEMBER_TYPES, OSMData

logger = logging.getLogger(__name__)

#: Index entries of the snapping tree are pieces of at most this length (metres).
_INDEX_PIECE_M = 10.0
#: Restriction values the router applies (``restriction=`` or ``restriction:motorcar=``).
_RESTRICTION_KINDS = ("no_left_turn", "no_right_turn", "no_straight_on", "no_u_turn", "no_entry", "no_exit",
                      "only_left_turn", "only_right_turn", "only_straight_on", "only_u_turn")
_CAR_EXCEPTIONS = ("motorcar", "motor_vehicle", "vehicle")


@dataclass
class Restriction:
    """A turn restriction resolved to road sequences (see :class:`kitscenes.sdmap.SDTurnRestriction`)."""

    id: int
    kind: str
    from_way: int
    via_nodes: tuple[int, ...]
    via_ways: tuple[int, ...]
    to_way: int
    sequences: tuple[tuple[int, ...], ...]

    @property
    def only(self) -> bool:
        return self.kind.startswith("only_")


@dataclass
class RoadGraph:
    """Arrays of the graph (all in the map frame of one origin)."""

    data: OSMData
    node_xy: np.ndarray            # (N, 2) every OSM node, map frame
    way_index: np.ndarray          # (E,) way position (into data.way_ids) of each edge
    edge_ptr: np.ndarray           # (E + 1,) into edge_nodes
    edge_nodes: np.ndarray         # node indices along the edges (way order)
    edge_road: np.ndarray          # (E, 2) road id forward / backward, -1 if not drivable that way
    road_edge: np.ndarray          # (R,) edge of each road
    road_forward: np.ndarray       # (R,) bool: along the way's node order
    road_from: np.ndarray          # (R,) node index
    road_to: np.ndarray            # (R,) node index
    road_length: np.ndarray        # (R,) metres
    road_public: np.ndarray        # (R,) bool
    road_access: np.ndarray        # (R,) uint8: 0 public, 1 restricted, 2 no (attributes.ACCESS_LEVELS)
    road_speed: np.ndarray         # (R,) m/s for time weights (speed limit, inf -> 130 km/h, unknown -> 50 km/h)
    out_ptr: np.ndarray            # (N + 1,) roads leaving each node (CSR)
    out_roads: np.ndarray
    in_ptr: np.ndarray
    in_roads: np.ndarray
    restrictions: list[Restriction] = field(default_factory=list)
    _seq_start: dict = field(default_factory=dict, repr=False)
    _tree: Optional[cKDTree] = field(default=None, repr=False)
    _tree_edge: Optional[np.ndarray] = field(default=None, repr=False)
    _edge_boxes: Optional[np.ndarray] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, data: OSMData, node_xy: np.ndarray) -> "RoadGraph":
        hw = data.tag_value_index("way", "highway")
        drivable_strings = {i for i, s in enumerate(data.strings) if s in attrs.DRIVABLE_HIGHWAYS}
        ways = [w for w in np.flatnonzero(hw >= 0) if int(hw[w]) in drivable_strings]

        node_lists: list[np.ndarray] = []
        kept: list[int] = []
        for w in ways:
            nodes = data.way_nodes[data.way_node_ptr[w]:data.way_node_ptr[w + 1]]
            nodes = nodes[nodes >= 0]
            if len(nodes) >= 2:
                node_lists.append(nodes)
                kept.append(int(w))
        n_nodes = data.num_nodes
        if node_lists:
            used = np.bincount(np.concatenate(node_lists), minlength=n_nodes)
        else:
            used = np.zeros(n_nodes, np.int64)
        is_graph_node = used >= 2
        for nodes in node_lists:
            is_graph_node[nodes[0]] = is_graph_node[nodes[-1]] = True

        way_index, edge_ptr, edge_nodes = [], [0], []
        road_edge, road_forward, road_access, road_speed = [], [], [], []
        edge_road = []
        for w, nodes in zip(kept, node_lists):
            tags = data.tags("way", w)
            fwd, bwd = attrs.directions(tags)
            level = attrs.ACCESS_LEVELS.index(attrs.access(tags))
            speeds = [_routing_speed(tags, True), _routing_speed(tags, False)]
            cut = np.flatnonzero(is_graph_node[nodes[1:-1]]) + 1
            bounds = [0, *cut.tolist(), len(nodes) - 1]
            for a, b in zip(bounds[:-1], bounds[1:]):
                piece = nodes[a:b + 1]
                if len(piece) < 2 or (len(piece) == 2 and piece[0] == piece[1]):
                    continue
                e = len(way_index)
                way_index.append(w)
                edge_nodes.extend(piece.tolist())
                edge_ptr.append(len(edge_nodes))
                ids = [-1, -1]
                for k, allowed in enumerate((fwd, bwd)):
                    if allowed:
                        ids[k] = len(road_edge)
                        road_edge.append(e)
                        road_forward.append(k == 0)
                        road_access.append(level)
                        road_speed.append(speeds[k])
                edge_road.append(ids)

        edge_ptr_a = np.array(edge_ptr, dtype=np.int64)
        edge_nodes_a = np.array(edge_nodes, dtype=np.int64)
        road_edge_a = np.array(road_edge, dtype=np.int64)
        road_forward_a = np.array(road_forward, dtype=bool)
        first = edge_nodes_a[edge_ptr_a[:-1]] if len(edge_ptr_a) > 1 else np.zeros(0, np.int64)
        last = edge_nodes_a[edge_ptr_a[1:] - 1] if len(edge_ptr_a) > 1 else np.zeros(0, np.int64)
        road_from = np.where(road_forward_a, first[road_edge_a], last[road_edge_a]) if len(road_edge_a) else road_edge_a
        road_to = np.where(road_forward_a, last[road_edge_a], first[road_edge_a]) if len(road_edge_a) else road_edge_a

        seg = np.diff(node_xy[edge_nodes_a], axis=0) if len(edge_nodes_a) > 1 else np.zeros((0, 2))
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        edge_length = cum[edge_ptr_a[1:] - 1] - cum[edge_ptr_a[:-1]] if len(edge_ptr_a) > 1 else np.zeros(0)
        road_length = edge_length[road_edge_a] if len(road_edge_a) else np.zeros(0)

        out_ptr, out_roads = _csr(road_from, n_nodes)
        in_ptr, in_roads = _csr(road_to, n_nodes)
        g = cls(
            data=data, node_xy=node_xy, way_index=np.array(way_index, dtype=np.int64), edge_ptr=edge_ptr_a,
            edge_nodes=edge_nodes_a, edge_road=np.array(edge_road, dtype=np.int64).reshape(-1, 2),
            road_edge=road_edge_a, road_forward=road_forward_a, road_from=road_from, road_to=road_to,
            road_length=road_length, road_public=np.array(road_access, dtype=np.uint8) == 0,
            road_access=np.array(road_access, dtype=np.uint8),
            road_speed=np.array(road_speed, dtype=np.float64), out_ptr=out_ptr, out_roads=out_roads,
            in_ptr=in_ptr, in_roads=in_roads,
        )
        g._build_restrictions()
        return g

    @property
    def num_roads(self) -> int:
        return len(self.road_edge)

    def road_way(self, r: int) -> int:
        """OSM way ID of road *r*."""
        return int(self.data.way_ids[self.way_index[self.road_edge[r]]])

    def road_nodes(self, r: int) -> np.ndarray:
        """Node indices of road *r* in driving order."""
        e = self.road_edge[r]
        nodes = self.edge_nodes[self.edge_ptr[e]:self.edge_ptr[e + 1]]
        return nodes if self.road_forward[r] else nodes[::-1]

    def road_points(self, r: int) -> np.ndarray:
        return self.node_xy[self.road_nodes(r)]

    def reverse(self, r: int) -> int:
        """The road of the same edge in the other direction, -1 if one-way."""
        return int(self.edge_road[self.road_edge[r], 1 if self.road_forward[r] else 0])

    def leaving(self, node: int) -> np.ndarray:
        return self.out_roads[self.out_ptr[node]:self.out_ptr[node + 1]]

    def entering(self, node: int) -> np.ndarray:
        return self.in_roads[self.in_ptr[node]:self.in_ptr[node + 1]]

    def successors(self, r: int) -> list[int]:
        """Roads a vehicle can continue to from *r* (no U-turn onto its own edge, except at a dead end)."""
        rev = self.reverse(r)
        out = [int(x) for x in self.leaving(int(self.road_to[r])) if x != rev]
        return out if out or rev < 0 else [rev]

    def predecessors(self, r: int) -> list[int]:
        rev = self.reverse(r)
        out = [int(x) for x in self.entering(int(self.road_from[r])) if x != rev]
        return out if out or rev < 0 else [rev]

    # ------------------------------------------------------------------
    # Turn restrictions
    # ------------------------------------------------------------------

    def _build_restrictions(self) -> None:
        d = self.data
        type_idx = d.tag_value_index("relation", "type")
        strings = d.strings
        way_pos = {int(w): i for i, w in enumerate(d.way_ids)}
        node_pos = None
        roads_of_way: dict[int, list[int]] = {}
        for r in range(self.num_roads):
            roads_of_way.setdefault(int(self.way_index[self.road_edge[r]]), []).append(r)
        out: list[Restriction] = []
        for i in np.flatnonzero(type_idx >= 0):
            if strings[type_idx[i]] != "restriction":
                continue
            tags = d.tags("relation", int(i))
            kind = tags.get("restriction:motorcar") or tags.get("restriction:motor_vehicle") or tags.get("restriction")
            if kind not in _RESTRICTION_KINDS:
                continue
            if any(x.strip() in _CAR_EXCEPTIONS for x in tags.get("except", "").split(";")):
                continue
            a, b = d.rel_member_ptr[i], d.rel_member_ptr[i + 1]
            froms, tos, via_n, via_w = [], [], [], []
            for t, ref, role in zip(d.member_type[a:b], d.member_ref[a:b], d.member_role[a:b]):
                role_s, kind_s = strings[role], MEMBER_TYPES[t]
                if role_s == "from" and kind_s == "way":
                    froms.append(int(ref))
                elif role_s == "to" and kind_s == "way":
                    tos.append(int(ref))
                elif role_s == "via" and kind_s == "node":
                    via_n.append(int(ref))
                elif role_s == "via" and kind_s == "way":
                    via_w.append(int(ref))
            if not froms or not tos or (bool(via_n) == bool(via_w)):
                continue
            if node_pos is None:
                order = np.argsort(d.node_ids)
                node_pos = (d.node_ids[order], order)
            for fw in froms:
                for tw in tos:
                    fr = roads_of_way.get(way_pos.get(fw, -1), [])
                    to = roads_of_way.get(way_pos.get(tw, -1), [])
                    if not fr or not to:
                        continue
                    if via_n:
                        sids, sorder = node_pos
                        k = int(np.searchsorted(sids, via_n[0]))
                        if k >= len(sids) or sids[k] != via_n[0]:
                            continue
                        v = int(sorder[k])
                        seqs = [(x, y) for x in fr if self.road_to[x] == v for y in to if self.road_from[y] == v
                                and (fw != tw or self.road_forward[x] != self.road_forward[y])]
                        if len(seqs) > 1:  # a from or to way through the via node: keep the turns of the kind
                            seqs = [p for p in seqs if _turn_fits(kind, self._turn_angle(*p))] or seqs
                    else:
                        via_set = {way_pos.get(w, -1) for w in via_w}
                        seqs = self._via_way_sequences(fr, set(to), via_set)
                    if seqs:
                        out.append(Restriction(int(d.rel_ids[i]), kind, fw, tuple(via_n), tuple(via_w), tw,
                                               tuple(sorted(set(seqs)))))
        self.restrictions = out
        start: dict[int, list[tuple[int, int]]] = {}
        for ri, res in enumerate(out):
            for si, seq in enumerate(res.sequences):
                start.setdefault(seq[0], []).append((ri, si))
        self._seq_start = start

    def _turn_angle(self, a: int, b: int) -> float:
        """Angle (left positive) between the last segment of road *a* and the first of road *b*."""
        pa, pb = self.road_points(a), self.road_points(b)
        da, db = pa[-1] - pa[-2], pb[1] - pb[0]
        return _wrap(math.atan2(db[1], db[0]) - math.atan2(da[1], da[0]))

    def _via_way_sequences(self, froms: list[int], tos: set[int], via_ways: set[int], max_depth: int = 12):
        seqs = []
        for a in froms:
            stack = [(a, (a,))]
            while stack:
                r, path = stack.pop()
                if len(path) > max_depth:
                    continue
                for nxt in self.successors(r):
                    if nxt in path:
                        continue
                    if nxt in tos and len(path) >= 2:
                        seqs.append(path + (nxt,))
                    elif int(self.way_index[self.road_edge[nxt]]) in via_ways:
                        stack.append((nxt, path + (nxt,)))
        return seqs

    def _advance(self, active: tuple, b: int) -> Optional[tuple]:
        """The restriction state after moving onto road *b*, or None if the move is forbidden."""
        nxt_active = []
        for ri, si, k in active:
            res = self.restrictions[ri]
            seq = res.sequences[si]
            last = len(seq) - 1
            if res.only and k + 1 == last and seq[last] != b:
                # an "only" sequence matched up to its last road: b must be its target, unless b continues
                # another sequence of the same restriction (several from/to roads)
                if not any(rj == ri and sj != si and k2 + 1 <= len(res.sequences[sj]) - 1
                           and res.sequences[sj][k2 + 1] == b for rj, sj, k2 in active):
                    return None
                continue
            if seq[k + 1] == b:
                if k + 1 == last:
                    if not res.only:
                        return None
                else:
                    nxt_active.append((ri, si, k + 1))
        for ri, si in self._seq_start.get(b, ()):
            nxt_active.append((ri, si, 0))
        return tuple(sorted(set(nxt_active)))

    def initial_state(self, r: int) -> tuple:
        return tuple(sorted((ri, si, 0) for ri, si in self._seq_start.get(r, ())))

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def road_cost(self, r: int, weight: str) -> float:
        length = float(self.road_length[r])
        return length if weight == "length" else length / float(self.road_speed[r])

    def _moves(self, r: int, uturn_penalty: Optional[float], access_penalty: Optional[float] = None,
               goal: int = -1) -> list[tuple[int, float]]:
        """``(next road, extra cost)``: the successors, plus the U-turn onto *r*'s own piece at its end
        node for *uturn_penalty* (when it is not None and that U-turn is not already a dead-end successor).

        With *access_penalty* (not None) access applies as in OSRM's car profile: roads closed to cars
        are left out (but for *goal*), and entering a restricted road from an unrestricted one costs
        *access_penalty*.
        """
        succ = self.successors(r)
        if access_penalty is not None:
            acc = self.road_access
            out = []
            for b in succ:
                if acc[b] == 2 and b != goal:
                    continue
                out.append((b, access_penalty if acc[b] == 1 and acc[r] == 0 else 0.0))
        else:
            out = [(b, 0.0) for b in succ]
        if uturn_penalty is not None:
            rev = self.reverse(r)
            if rev >= 0 and rev not in succ:
                out.append((rev, float(uturn_penalty)))
        return out

    def shortest_path(
        self, start: int, start_s: float, goal: int, goal_s: float, *, weight: str = "length",
        restrictions: bool = True, access_penalty: Optional[float] = None, max_cost: float = math.inf,
        uturn_penalty: Optional[float] = None,
    ) -> Optional[tuple[float, list[int]]]:
        """A* from position *start_s* on road *start* to *goal_s* on road *goal*: ``(cost, roads)`` or None.

        The cost is metres (``weight="length"``) or seconds (``"time"``, at the
        roads' speed limits). Ties are broken by insertion order, so the result
        is deterministic. With *access_penalty* (in units of the cost) access
        applies (see :meth:`_moves`). U-turns are made only at dead ends, or
        at any graph node for *uturn_penalty* (in units of the cost) if it is given.
        """
        if weight not in ("length", "time"):
            raise ValueError(f"weight must be 'length' or 'time', got {weight!r}")
        goal_xy = self.node_xy[self.road_from[goal]]
        vmax = 1.0 if weight == "length" else max(float(self.road_speed.max(initial=1.0)), 1.0)

        def frac_cost(r: int, s: float) -> float:
            length = float(self.road_length[r])
            return self.road_cost(r, weight) * (s / length if length > 0 else 0.0)

        def h(r: int) -> float:
            return float(np.hypot(*(self.node_xy[self.road_to[r]] - goal_xy))) / vmax

        first_cost = self.road_cost(start, weight) - frac_cost(start, start_s)
        if start == goal and goal_s >= start_s:
            return frac_cost(goal, goal_s) - frac_cost(start, start_s), [start]
        state0 = (start, self.initial_state(start) if restrictions else ())
        best = {state0: first_cost}
        parent: dict = {state0: None}
        counter = 0
        heap = [(first_cost + h(start), counter, first_cost, state0, False)]
        while heap:
            _, _, cost, state, final = heapq.heappop(heap)
            if final:
                path = []
                s = state
                while s is not None:
                    path.append(s[0])
                    s = parent[s]
                return cost, path[::-1]
            if cost > best.get(state, math.inf) or cost > max_cost:
                continue
            r, active = state
            for b, extra in self._moves(r, uturn_penalty, access_penalty, goal):
                if restrictions:
                    nxt = self._advance(active, b)
                    if nxt is None:
                        continue
                else:
                    nxt = ()
                ns = (b, nxt)
                counter += 1
                if b == goal:
                    total = cost + extra + frac_cost(b, goal_s)
                    end = (b, nxt + ("goal",))
                    if total < best.get(end, math.inf):
                        best[end] = total
                        parent[end] = state
                        heapq.heappush(heap, (total, counter, total, end, True))
                c = cost + extra + self.road_cost(b, weight)
                if c < best.get(ns, math.inf):
                    best[ns] = c
                    parent[ns] = state
                    heapq.heappush(heap, (c + h(b), counter, c, ns, False))
        return None

    def distances_from(self, start: int, start_s: float, max_m: float, *, public_only: bool = False,
                       uturn_penalty: Optional[float] = None) -> dict[int, float]:
        """Driving distance (m) from *start_s* on *start* to the start of every road within *max_m*
        (Dijkstra, no restrictions; U-turns at graph nodes for *uturn_penalty* metres if given)."""
        out: dict[int, float] = {}
        first = float(self.road_length[start]) - start_s
        heap = [(first, start)]
        seen = {start: first}
        while heap:
            d, r = heapq.heappop(heap)
            if d > seen.get(r, math.inf) or d > max_m:
                continue
            for b, extra in self._moves(r, uturn_penalty):
                if public_only and not self.road_public[b]:
                    continue
                if d + extra < out.get(b, math.inf):
                    out[b] = d + extra
                nd = d + extra + float(self.road_length[b])
                if nd < seen.get(b, math.inf):
                    seen[b] = nd
                    heapq.heappush(heap, (nd, b))
        return out

    # ------------------------------------------------------------------
    # Spatial index and snapping
    # ------------------------------------------------------------------

    def _index(self) -> tuple[cKDTree, np.ndarray, np.ndarray]:
        if self._tree is None:
            mids, owner, seg_of = [], [], []
            pts = self.node_xy[self.edge_nodes]
            for e in range(len(self.way_index)):
                a, b = self.edge_ptr[e], self.edge_ptr[e + 1]
                p = pts[a:b]
                for k in range(len(p) - 1):
                    seglen = float(np.hypot(*(p[k + 1] - p[k])))
                    n = max(1, int(math.ceil(seglen / _INDEX_PIECE_M)))
                    t = (np.arange(n) + 0.5) / n
                    mids.append(p[k] + t[:, None] * (p[k + 1] - p[k]))
                    owner.append(np.full(n, e))
                    seg_of.append(np.full(n, a + k))
            if mids:
                self._tree = cKDTree(np.concatenate(mids))
                self._tree_edge = np.concatenate(owner)
            else:
                self._tree = cKDTree(np.zeros((1, 2)) + 1e12)
                self._tree_edge = np.zeros(1, np.int64) - 1
        return self._tree, self._tree_edge, None

    def edges_near(self, xy: np.ndarray, radius: float) -> np.ndarray:
        """Edges with a piece within about *radius* of *xy* (a superset)."""
        tree, owner, _ = self._index()
        idx = tree.query_ball_point(np.asarray(xy, float), radius + _INDEX_PIECE_M / 2)
        e = np.unique(owner[np.asarray(idx, dtype=np.int64)]) if len(idx) else np.zeros(0, np.int64)
        return e[e >= 0]

    def edge_boxes(self) -> np.ndarray:
        if self._edge_boxes is None:
            pts = self.node_xy[self.edge_nodes]
            if len(self.way_index):
                lo = np.minimum.reduceat(pts, self.edge_ptr[:-1], axis=0)
                hi = np.maximum.reduceat(pts, self.edge_ptr[:-1], axis=0)
                self._edge_boxes = np.hstack([lo, hi])
            else:
                self._edge_boxes = np.zeros((0, 4))
        return self._edge_boxes

    def project(self, r: int, xy: np.ndarray) -> tuple[float, float, float]:
        """``(s, distance, direction)`` of the closest point of road *r* to *xy*: arc length, metres, heading there."""
        return project_polyline(self.road_points(r), np.asarray(xy, float))

    def snap(self, xy: np.ndarray, yaw: Optional[float], max_distance: float, max_heading_error: float,
             public_only: bool = False) -> list[tuple[int, float, float, float]]:
        """Roads within *max_distance* of *xy* whose direction differs from *yaw* by at most
        *max_heading_error*: ``(road, s, distance, heading_error)``, best first."""
        out = []
        for e in self.edges_near(xy, max_distance):
            for r in self.edge_road[e]:
                if r < 0 or (public_only and not self.road_public[r]):
                    continue
                s, dist, direction = self.project(int(r), xy)
                if dist > max_distance:
                    continue
                err = 0.0 if yaw is None else abs(_wrap(direction - yaw))
                if err <= max_heading_error:
                    out.append((int(r), s, dist, err))
        out.sort(key=lambda m: (m[2] + 5.0 * m[3], m[0]))
        return out


def project_polyline(p: np.ndarray, q: np.ndarray) -> tuple[float, float, float]:
    """``(s, distance, direction)`` of the closest point of polyline *p* (N, 2) to *q*."""
    if len(p) == 1:
        return 0.0, float(np.hypot(*(p[0] - q))), 0.0
    a, b = p[:-1], p[1:]
    ab = b - a
    l2 = np.einsum("ij,ij->i", ab, ab)
    t = np.clip(np.einsum("ij,ij->i", q - a, ab) / np.maximum(l2, 1e-12), 0.0, 1.0)
    proj = a + t[:, None] * ab
    d = np.hypot(*(proj - q).T)
    i = int(np.argmin(d))
    seglen = np.sqrt(l2)
    s = float(seglen[:i].sum() + t[i] * seglen[i])
    return s, float(d[i]), float(math.atan2(ab[i, 1], ab[i, 0]))


def _routing_speed(tags, forward: bool) -> float:
    v, _ = attrs.speed_limit(tags, forward)  # km/h
    return min(attrs.URBAN_DEFAULT_KPH if v is None else v, 130.0) / 3.6  # m/s for the time weights


def _turn_fits(kind: str, angle: float) -> bool:
    """Whether a turn by *angle* (left positive) is the turn of a restriction *kind* (``no_left_turn``, ...)."""
    deg = math.degrees(angle)
    if kind.endswith("left_turn"):
        return deg > 10.0
    if kind.endswith("right_turn"):
        return deg < -10.0
    if kind.endswith("straight_on"):
        return abs(deg) < 60.0
    if kind.endswith("u_turn"):
        return abs(deg) > 120.0
    return True


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _csr(keys: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(keys, kind="stable")
    counts = np.bincount(keys, minlength=n) if len(keys) else np.zeros(n, np.int64)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return ptr, order.astype(np.int64)


# ----------------------------------------------------------------------
# Junctions
# ----------------------------------------------------------------------


@dataclass
class JunctionIndex:
    """Junction clusters: graph nodes where three or more edges meet, merged across short internal edges."""

    cluster_of: dict[int, int]            # node index -> cluster id
    members: list[list[int]]              # cluster id -> node indices
    internal_roads: set[int]

    @classmethod
    def build(cls, g: RoadGraph, merge_m: float) -> "JunctionIndex":
        n_edges = len(g.way_index)
        first = g.edge_nodes[g.edge_ptr[:-1]] if n_edges else np.zeros(0, np.int64)
        last = g.edge_nodes[g.edge_ptr[1:] - 1] if n_edges else np.zeros(0, np.int64)
        neighbours: dict[int, set[int]] = {}
        for a, b in zip(first.tolist(), last.tolist()):
            if a == b:
                continue
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
        junction_nodes = {n for n, nb in neighbours.items() if len(nb) >= 3}
        parent = {n: n for n in junction_nodes}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        internal: set[int] = set()
        lengths = g.road_length
        for e in range(n_edges):
            a, b = int(first[e]), int(last[e])
            if a in junction_nodes and b in junction_nodes and a != b:
                roads = [int(r) for r in g.edge_road[e] if r >= 0]
                if roads and float(lengths[roads[0]]) < merge_m:
                    ra, rb = find(a), find(b)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
                    internal.update(roads)
        groups: dict[int, list[int]] = {}
        for n in sorted(junction_nodes):
            groups.setdefault(find(n), []).append(n)
        members = [groups[k] for k in sorted(groups)]
        cluster_of = {n: i for i, ms in enumerate(members) for n in ms}
        # internal roads only between nodes of the same cluster
        internal = {r for r in internal if cluster_of.get(int(g.road_from[r])) == cluster_of.get(int(g.road_to[r]))}
        return cls(cluster_of, members, internal)
