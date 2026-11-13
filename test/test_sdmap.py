"""kitscenes.sdmap on a small synthetic OpenStreetMap file (test/sdmap_fixture.py)."""

from __future__ import annotations

import math
import pickle

import numpy as np
import pytest

from kitscenes.map import Box, Circle
from kitscenes.sdmap import Direction, RoadClass, SDMap, SDRoute
from kitscenes.sdmap import _cache, _parse
from kitscenes.sdmap import attributes as attrs

from map_fixture import ego_pose
from sdmap_fixture import MOJIBAKE_NAME, ORIGIN, OSM_BASE, write_osm

EAST, WEST, NORTH, SOUTH = 0.0, math.pi, math.pi / 2, -math.pi / 2


@pytest.fixture(scope="module")
def osm_file(tmp_path_factory):
    return write_osm(tmp_path_factory.mktemp("sdmap") / "Karlsruhe.osm")


@pytest.fixture(scope="module")
def smap(osm_file, tmp_path_factory) -> SDMap:
    return SDMap(osm_file, ORIGIN, city="Karlsruhe", cache_dir=tmp_path_factory.mktemp("cache"))


def _road_on(smap: SDMap, way_id: int, heading: float):
    roads = [r for r in smap.roads() if r.way_id == way_id]
    return min(roads, key=lambda r: abs((math.atan2(*(r.points[-1] - r.points[0])[::-1]) - heading + math.pi)
                                        % (2 * math.pi) - math.pi))


# ----------------------------------------------------------------------
# Parsing and cache
# ----------------------------------------------------------------------


def test_parse_keeps_everything(osm_file):
    d = _parse.parse(osm_file)
    assert d.osm_base == OSM_BASE
    assert d.num_ways == 17 and d.num_relations == 3
    w = int(np.flatnonzero(d.way_ids == 10)[0])
    assert d.tags("way", w)["turn:lanes:forward"] == "left|through|through;right"
    r = int(np.flatnonzero(d.rel_ids == 190)[0])
    a, b = d.rel_member_ptr[r], d.rel_member_ptr[r + 1]
    assert [d.strings[k] for k in d.member_role[a:b]] == ["from", "via", "to"]
    assert d.member_ref[a:b].tolist() == [110, 111, 112]
    assert (d.way_nodes >= 0).all()


def test_cache_is_written_and_read(osm_file, tmp_path, monkeypatch):
    _cache.clear_memory()
    data = _cache.load_osm(osm_file, cache_dir=tmp_path)
    assert list(tmp_path.glob("Karlsruhe-*.pkl"))
    _cache.clear_memory()
    monkeypatch.setattr(_cache, "parse", lambda *a, **k: pytest.fail("parsed again instead of reading the cache"))
    again = _cache.load_osm(osm_file, cache_dir=tmp_path)
    assert again.node_ids.tolist() == data.node_ids.tolist()


def test_default_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("KITSCENES_CACHE", str(tmp_path))
    assert _cache.default_cache_dir() == tmp_path / "sdmap"


def test_sdmap_pickles_to_its_path(smap):
    clone = pickle.loads(pickle.dumps(smap))
    assert clone.path == smap.path and clone.origin == smap.origin and clone.num_roads == smap.num_roads


# ----------------------------------------------------------------------
# Attributes
# ----------------------------------------------------------------------


@pytest.mark.parametrize("value, kph, source", [
    ("50", 50.0, "tag"), ("30 mph", 30 * 1.609344, "tag"), ("none", math.inf, "tag"),
    ("DE:urban", 50.0, "implicit"), ("DE:zone30", 30.0, "implicit"), ("DE:motorway", math.inf, "implicit"),
    ("signals", None, None), ("50;30", 50.0, "tag"),
])
def test_parse_speed(value, kph, source):
    v, s = attrs.parse_speed(value)
    assert s == source and (v == pytest.approx(kph) if kph is not None else v is None)


def test_directions_and_access():
    assert attrs.directions({"highway": "residential"}) == (True, True)
    assert attrs.directions({"highway": "residential", "oneway": "yes"}) == (True, False)
    assert attrs.directions({"highway": "residential", "oneway": "-1"}) == (False, True)
    assert attrs.directions({"highway": "motorway"}) == (True, False)
    assert attrs.directions({"highway": "tertiary", "junction": "roundabout"}) == (True, False)
    assert attrs.directions({"highway": "motorway", "oneway": "no"}) == (True, True)
    assert attrs.access({"highway": "service", "access": "private"}) == "restricted"
    assert attrs.access({"highway": "residential", "access": "permissive", "motor_vehicle": "private"}) == "restricted"
    assert attrs.access({"highway": "track", "access": "no"}) == "no"
    assert not attrs.is_public({"highway": "service", "access": "private"})
    assert attrs.is_public({"highway": "service", "access": "private", "motor_vehicle": "yes"})
    assert not attrs.is_public({"highway": "busway"})


def test_lanes_and_turn_lanes():
    assert attrs.lanes({"highway": "primary", "lanes": "4"}) == (2, "tag_split")
    assert attrs.lanes({"highway": "primary", "lanes": "3", "lanes:forward": "2"}) == (2, "tag")
    assert attrs.lanes({"highway": "primary", "lanes": "3", "lanes:forward": "2"}, forward=False) == (1, "tag")
    assert attrs.lanes({"highway": "motorway"}) == (2, "default")
    turn = attrs.turn_lanes({"highway": "primary", "oneway": "yes", "turn:lanes": "left||through;right"})
    assert turn == (frozenset({"left"}), frozenset({"none"}), frozenset({"through", "right"}))
    assert attrs.turn_lanes({"highway": "primary", "turn:lanes": "left|through"}) is None  # two-way: ambiguous


def test_road_attributes(smap):
    east = _road_on(smap, 10, EAST)
    assert east.road_class is RoadClass.PRIMARY and not east.one_way and east.public
    assert east.name == "Kaiserstraße"  # repaired
    assert east.speed_limit_mps == pytest.approx(50 / 3.6) and east.speed_limit_source == "tag"
    assert east.speed_limit_kph == pytest.approx(50.0)
    assert (east.lanes, east.lanes_source) == (2, "tag_split")
    assert east.turn_lanes == (frozenset({"left"}), frozenset({"through"}), frozenset({"through", "right"}))
    assert east.reverse_id is not None and smap.road(east.reverse_id).turn_lanes is None
    south = _road_on(smap, 11, NORTH)
    assert south.speed_limit_mps == pytest.approx(30 / 3.6) and south.speed_limit_source == "implicit"
    assert _road_on(smap, 12, NORTH).speed_limit_source == "zone"
    assert _road_on(smap, 14, EAST).speed_limit_source == "implicit"
    oneway = [r for r in smap.roads() if r.way_id == 13]
    assert len(oneway) == 1 and oneway[0].one_way and oneway[0].points[-1][1] > oneway[0].points[0][1]
    private = [r for r in smap.roads() if r.way_id == 15]
    assert private and not any(r.public for r in private)
    assert not any(r.way_id == 17 for r in smap.roads())  # a footway is no road
    raw = SDMap(smap.path, ORIGIN, cache_dir=smap.cache_dir, repair_encoding=False)
    assert _road_on(raw, 10, EAST).name == MOJIBAKE_NAME


def test_roads_in_a_region(smap):
    roads = smap.roads(Circle((300.0, 300.0), 5.0))
    assert {r.way_id for r in roads} == {12}
    assert smap.roads(Circle((300.0, 300.0), 5.0), classes=["primary"]) == []
    assert smap.roads(Circle((50_000.0, 0.0), 10.0)) == []  # outside the map: empty, no error


# ----------------------------------------------------------------------
# Junctions and restrictions
# ----------------------------------------------------------------------


def test_junctions_merge_dual_carriageways(smap):
    crossroads = smap.junctions(Circle((300.0, 200.0), 1.0))
    assert len(crossroads) == 1 and crossroads[0].control == "traffic_signals"
    dual = smap.junctions(Circle((1000.0, 5.0), 8.0))
    assert len(dual) == 1 and len(dual[0].node_ids) == 2 and len(dual[0].internal) == 2


def test_restrictions(smap):
    res = {r.id: r for r in smap.restrictions()}
    assert set(res) == {90, 190}
    assert res[90].kind == "no_left_turn" and res[90].via_nodes == (3,)
    (seq,) = res[90].road_sequences
    assert smap.road(seq[0]).way_id == 10 and smap.road(seq[1]).way_id == 12 and smap.road(seq[1]).forward
    assert res[190].via_ways == (111,) and all(len(s) == 3 for s in res[190].road_sequences)


def test_route_respects_a_via_node_restriction(smap):
    start, goal = (150.0, 200.0, EAST), (300.0, 300.0, NORTH)
    free = smap.route(start, goal, restrictions=False)
    assert free.length_m == pytest.approx(250.0, abs=1.0)
    legal = smap.route(start, goal)
    assert legal is not None and legal.length_m > 400.0
    banned = res_pairs = next(r for r in smap.restrictions() if r.id == 90).road_sequences[0]
    pairs = set(zip(legal.road_ids[:-1], legal.road_ids[1:]))
    assert tuple(banned) not in pairs and res_pairs


def test_route_respects_a_via_way_restriction(smap):
    start, goal = (1950.0, 0.0, EAST), (2150.0, 50.0, EAST)
    free = smap.route(start, goal, restrictions=False)
    assert {smap.road(r).way_id for r in free.road_ids} >= {110, 111, 112, 113}
    legal = smap.route(start, goal)
    ways = [smap.road(r).way_id for r in legal.road_ids]
    assert 111 not in ways and 115 in ways and legal.length_m > free.length_m


def test_route_is_deterministic_and_time_weight_works(smap):
    a = smap.route((150.0, 200.0, EAST), (498.0, 380.0, NORTH))
    b = smap.route((150.0, 200.0, EAST), (498.0, 380.0, NORTH))
    assert a.road_ids == b.road_ids and np.array_equal(a.points, b.points)
    t = smap.route((150.0, 200.0, EAST), (498.0, 380.0, NORTH), weight="time")
    assert t is not None and t.road_ids


def test_route_without_a_match_is_none(smap):
    assert smap.route((50_000.0, 0.0, EAST), (150.0, 200.0, EAST)) is None
    assert smap.route((480.0, 300.0, SOUTH), (150.0, 200.0, EAST)) is None  # against the one-way


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def test_match_is_heading_aware(smap):
    east = smap.match(ego_pose(150.0, 198.0, EAST))
    assert east and smap.road(east[0].road_id).way_id == 10 and smap.road(east[0].road_id).forward
    # on the piece from node 2 (x = 100); OSM coordinates are stored to 1e-7 degrees (about 1 cm)
    assert east[0].distance_m == pytest.approx(2.0, abs=0.02) and east[0].s_m == pytest.approx(50.0, abs=0.02)
    west = smap.match(ego_pose(150.0, 202.0, WEST))
    assert west and not smap.road(west[0].road_id).forward
    assert smap.match(ego_pose(500.0, 300.0, SOUTH)) == []  # against the one-way
    assert smap.match(ego_pose(50_000.0, 0.0, EAST)) == []
    road = smap.attributes_at(ego_pose(150.0, 198.0, EAST))
    assert road.name == "Kaiserstraße"


def _trajectory(points, step=1.0):
    """Positions every *step* metres along a polyline, with headings."""
    from kitscenes.sdmap._geometry import arc_length, heading_at, point_at

    p = np.asarray(points, float)
    s = arc_length(p)
    at = np.arange(0.0, s[-1], step)
    return point_at(p, s, at), np.array([heading_at(p, s, x) for x in at])


def test_match_trajectory_left_turn_and_u_turn(smap):
    xy, yaw = _trajectory([(120, 198), (298, 198), (298, 350)])
    xy[:, 1] += 0.0
    track = smap.match_trajectory(xy, yaw)
    assert track.matched.all() and len(track.routes) == 1
    route = track.routes[0]
    assert [smap.road(r).way_id for r in route.road_ids] == [10, 12]  # matching ignores restrictions
    assert route.engine == "matched"
    turn = [m for m in route.maneuvers if m.is_decision]
    assert len(turn) == 1 and turn[0].kind == "turn" and turn[0].direction is Direction.LEFT
    assert turn[0].angle_rad == pytest.approx(math.pi / 2, abs=0.05) and turn[0].control == "traffic_signals"
    assert turn[0].lanes_valid == (True, False, False)
    # out and back: a U-turn at the crossroads
    xy2, yaw2 = _trajectory([(120, 198), (300, 198)])
    xy3, yaw3 = _trajectory([(300, 202), (150, 202)])
    back = smap.route_along(np.concatenate([xy2, xy3]), np.concatenate([yaw2, yaw3]))
    assert back is not None and any(m.kind == "u_turn" for m in back.maneuvers)
    assert back.length_m < 400


def test_match_trajectory_off_the_map_is_unmatched(smap):
    xy, yaw = _trajectory([(40_000, 0), (40_200, 0)])
    track = smap.match_trajectory(xy, yaw)
    assert not track.matched.any() and track.routes == ()
    assert smap.route_along(xy, yaw) is None


def test_dual_carriageway_left_turn_is_one_maneuver(smap):
    route = smap.route((950.0, 0.0, EAST), (1000.0, 80.0, NORTH))
    decisions = [m for m in route.maneuvers if m.is_decision]
    assert len(decisions) == 1 and decisions[0].direction is Direction.LEFT
    assert decisions[0].junction_id is not None


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------


def test_route_geometry(smap):
    route = smap.route((150.0, 200.0, EAST), (450.0, 200.0, EAST))
    assert isinstance(route, SDRoute) and route.length_m == pytest.approx(300.0, abs=0.02)
    assert route.project((250.0, 210.0)) == pytest.approx(100.0, abs=0.02)
    assert np.allclose(route.point_at(100.0), [250.0, 200.0], atol=0.02)
    assert route.heading_at(10.0) == pytest.approx(0.0, abs=0.02)
    assert route.distance_to((250.0, 203.0)) == pytest.approx(3.0, abs=0.02)
    pose = ego_pose(150.0, 198.0, EAST)
    ego = route.in_ego(pose, road_height_m=2.0)
    assert np.allclose(ego[0], [0.0, 2.0, -2.0], atol=0.02)
    speeds = route.speed_limits()
    assert speeds.shape == (len(route.points),) and np.allclose(speeds, 50 / 3.6)
    assert np.allclose(route.speed_limits(unit="kph"), 50.0)
    kinds = [m.kind for m in route.maneuvers]
    assert kinds[0] == "depart" and kinds[-1] == "arrive" and "continue" in kinds
    m, dist = route.next_maneuver(pose)
    assert m.kind == "continue" and dist == pytest.approx(150.0, abs=0.02)
    targets = route.target_points(pose, max_distance_m=200.0)
    assert np.allclose(targets, [[150.0, 2.0], [200.0, 2.0]], atol=0.02)
    spaced = route.target_points(pose, max_distance_m=50.0, spacing_m=25.0)
    assert np.allclose(spaced, [[25.0, 2.0], [50.0, 2.0]], atol=0.02)


def test_route_pickles_with_its_map(smap):
    route = smap.route((150.0, 200.0, EAST), (450.0, 200.0, EAST))
    clone = pickle.loads(pickle.dumps(route))
    assert clone.road_ids == route.road_ids and clone.sd_map.path == smap.path
    assert [m.kind for m in clone.maneuvers] == [m.kind for m in route.maneuvers]


def test_direction_bins():
    assert Direction.of(math.radians(10)) is Direction.STRAIGHT
    assert Direction.of(math.radians(30)) is Direction.SLIGHT_LEFT
    assert Direction.of(math.radians(-90)) is Direction.RIGHT
    assert Direction.of(math.radians(150)) is Direction.SHARP_LEFT
    assert Direction.of(math.radians(-175)) is Direction.UTURN


# ----------------------------------------------------------------------
# Raw elements
# ----------------------------------------------------------------------


def test_elements_in_a_region(smap):
    roi = Box((160.0, 230.0), 0.0, 100.0, 60.0, 40.0, 40.0)  # x 100..260, y 190..270
    el = smap.elements(roi)
    assert 16 in el.way_ids.tolist() and 10 in el.way_ids.tolist() and 17 in el.way_ids.tolist()
    assert set(el.node_ids.tolist()) == {40, 41}  # tagged nodes only
    assert np.isnan(el.node_points[:, 2]).all() and el.frame == "map"
    main = el.way_points[el.way_ids.tolist().index(10)]
    assert main[:, 0].min() == pytest.approx(100.0, abs=1e-3) and main[:, 0].max() == pytest.approx(260.0)  # clipped
    bus = [r for r in el.relations if r.id == 91]
    assert bus and bus[0].member_index[0] == el.way_ids.tolist().index(10) and bus[0].member_index[2] == -1
    roads = smap.elements(roi, ways="roads")
    assert set(roads.way_ids.tolist()) == {10, 15}
    whole = smap.elements(roi, clip=False)
    assert len(whole.way_points[whole.way_ids.tolist().index(10)]) == 4
    pose = ego_pose(160.0, 230.0, NORTH)
    ego = el.in_ego(pose, road_height_m=2.0)
    assert ego.frame == "ego" and np.allclose(ego.node_points[:, 2], -2.0)
    k = el.node_ids.tolist().index(41)
    assert np.allclose(ego.node_points[k, :2], [20.0, -90.0], atol=1e-3)  # (250, 250) heading north from (160, 230)
    assert len(el.resampled(5).way_points[0]) == 5


def test_empty_region_is_an_empty_result(smap):
    far = Circle((80_000.0, 80_000.0), 30.0)
    el = smap.elements(far)
    assert el.empty and len(el) == 0 and el.relations == []
    assert smap.junctions(far) == [] and smap.restrictions(far) == []
    assert not smap.covers(np.array([[80_000.0, 80_000.0]]))[0]
    assert smap.covers(np.array([[300.0, 200.0]]))[0] and not smap.covers(np.array([[300.0, 200.0]]), 500.0)[0]


def test_relations_of_the_file(smap):
    rels = smap.relations()
    assert [r.id for r in rels] == [90, 91, 190]  # file order
    bus = rels[1]
    assert bus.tags == {"type": "route", "route": "bus", "ref": "62"}
    assert (bus.member_types, bus.member_ids, bus.member_roles) == (("way", "way", "node"), (10, 12, 3), ("", "", "stop"))
    assert bus.member_index == (-1, -1, -1)
    assert smap.relations() is not rels and smap.relations()[0] is rels[0]


def test_to_networkx(smap):
    nx = pytest.importorskip("networkx")
    graph = smap.to_networkx()
    assert isinstance(graph, nx.DiGraph) and graph.number_of_edges() > 0
    east = _road_on(smap, 10, EAST)
    edge = graph.edges[east.from_node, east.to_node]
    assert edge["way_id"] == 10 and edge["length_m"] == pytest.approx(east.length_m)
    assert edge["road_id"] in {r.id for r in smap.roads() if (r.from_node, r.to_node) == (east.from_node, east.to_node)}
    assert np.allclose(graph.nodes[east.from_node]["xy"], east.points[0])
    assert nx.has_path(graph, 1, 7) and graph.has_edge(4, 7) and not graph.has_edge(7, 4)  # way 13 is one-way north
    near = smap.to_networkx(Circle((300.0, 300.0), 5.0))
    assert set(near.nodes) == {3, 6}


def test_route_u_turns_and_restricted_access(smap):
    back = smap.route((150.0, 200.0, EAST), (120.0, 200.0, WEST))  # turn back at the next junction
    assert back is not None and any(m.kind == "u_turn" for m in back.maneuvers)
    into_private = smap.route((50.0, 200.0, EAST), (100.0, 140.0, SOUTH))  # the private service road
    assert into_private is not None and smap.road(into_private.road_ids[-1]).access == "restricted"
    out_of_private = smap.route((100.0, 140.0, NORTH), (250.0, 200.0, EAST))
    assert out_of_private is not None and out_of_private.length_m == pytest.approx(210.0, abs=1.0)


def test_route_to_lanelets_with_a_fake_hd_map(smap):
    """SDRoute.to_lanelets joins the HD lanes at the route's ends with the HD routing graph (here a stand-in)."""
    from types import SimpleNamespace

    from kitscenes.map.elements import LaneMatch

    lanes = {  # two lanes 2 m right of way 10 (eastbound), one 30 m away
        1: np.array([[140.0, 198.0, 0.0], [250.0, 198.0, 0.0]]),
        2: np.array([[250.0, 198.0, 0.0], [460.0, 198.0, 0.0]]),
        3: np.array([[140.0, 168.0, 0.0], [460.0, 168.0, 0.0]]),
    }

    class FakeHD:
        def match(self, pose, max_distance, max_heading_error):
            x, y = pose.xyz[:2]
            out = []
            for i, c in lanes.items():
                if c[0, 0] - 1 <= x <= c[-1, 0] + 1 and abs(c[0, 1] - y) <= max_distance:
                    out.append(LaneMatch(i, False, abs(c[0, 1] - y), 0.0))
            return sorted(out, key=lambda m: m.distance)

        def route(self, a, b):
            return {(1, 2): [1, 2]}.get((a, b))

        def lane(self, i):
            return SimpleNamespace(centerline=lanes[i])

    route = smap.route((150.0, 200.0, EAST), (450.0, 200.0, EAST))
    assert route.to_lanelets(FakeHD()) == [1, 2]
    lanes[2] = np.array([[250.0, 198.0, 0.0], [350.0, 175.0, 0.0], [460.0, 198.0, 0.0]])  # bends away: rejected
    assert route.to_lanelets(FakeHD()) is None
