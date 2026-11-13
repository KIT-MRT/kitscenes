"""kitscenes.map on a synthetic map with the cases the release maps have."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from kitscenes.map import Box, Circle, LaneKind, MapOrigin, Region, SceneMap, project_to_camera
from kitscenes.rig import Camera
from map_fixture import NORTH, ORIGIN, SOUTH, ego_pose, write_bike_lane_osm, write_osm
from release_fixture import make_scene


@pytest.fixture(scope="module")
def smap(tmp_path_factory) -> SceneMap:
    pytest.importorskip("lanelet2")
    return SceneMap(write_osm(tmp_path_factory.mktemp("map") / "map.osm"), ORIGIN)


# ---------------------------------------------------------------------------
# No lanelet2 needed
# ---------------------------------------------------------------------------


def test_origin() -> None:
    assert ORIGIN.epsg == 32632 and MapOrigin(-33.9, 151.2).epsg == 32756 and MapOrigin(52.5, 13.4).epsg == 32633
    xy = ORIGIN.from_latlon([49.0154, 49.0144], [8.4172, 8.4182])
    np.testing.assert_allclose(xy[0], [0, 111.2], atol=1.0)  # ~111 m north (grid convergence: 0.85 m east)
    np.testing.assert_allclose(ORIGIN.to_latlon(xy), [[49.0154, 8.4172], [49.0144, 8.4182]], atol=1e-9)
    np.testing.assert_allclose(ORIGIN.from_utm(ORIGIN.to_utm(np.array([[1.0, 2.0, 3.0]]))), [[1, 2, 3]])
    assert pickle.loads(pickle.dumps(ORIGIN)) == ORIGIN


def test_origin_matches_lanelet2() -> None:
    pytest.importorskip("lanelet2")
    from lanelet2.core import GPSPoint
    from lanelet2.io import Origin
    from lanelet2.projection import UtmProjector

    proj = UtmProjector(Origin(ORIGIN.lat, ORIGIN.lon))
    p = proj.forward(GPSPoint(49.05, 8.47, 0.0))
    np.testing.assert_allclose(ORIGIN.from_latlon(49.05, 8.47)[0], [p.x, p.y], atol=1e-6)


def test_roi_geometry() -> None:
    long_line = np.array([[-100.0, 5.0, 0.0], [100.0, 5.0, 0.0]])  # no vertex near the origin
    assert Circle((0, 0), 6).hits(long_line) and not Circle((0, 0), 4).hits(long_line)
    box = Region(ahead=10, behind=0, left=1, right=1).around(ego_pose(0, 0, NORTH))  # x in [-1, 1], y in [0, 10]
    assert box.hits(long_line)
    assert not box.hits(np.array([[2.0, -5.0], [2.0, 50.0]]))
    assert box.hits(np.array([[0.0, 3.0]]))


# ---------------------------------------------------------------------------
# With lanelet2
# ---------------------------------------------------------------------------


def test_lanes(smap: SceneMap) -> None:
    lanes = {lane.id: lane for lane in smap.lanes()}
    assert {i: l.kind for i, l in lanes.items()} == {
        100: LaneKind.ROAD, 101: LaneKind.ROAD, 102: LaneKind.BICYCLE, 103: LaneKind.CROSSWALK, 104: LaneKind.ROAD}
    assert lanes[100].speed_limit_mps == pytest.approx(50 / 3.6) and lanes[100].speed_limit_kph == pytest.approx(50.0)
    assert lanes[101].speed_limit_mps == pytest.approx(30 / 3.6)
    assert lanes[102].speed_limit_mps is None and lanes[102].speed_limit_kph is None and not lanes[102].vehicle
    assert lanes[100].one_way and lanes[100].vehicle
    assert lanes[100].successors == (104,) and lanes[104].predecessors == (100,)
    assert lanes[100].regulatory_ids == (200,)
    assert lanes[100].centerline.shape[1] == 3
    np.testing.assert_allclose(lanes[100].centerline[:, 2], 100.0, atol=1e-6)
    assert lanes[100].polygon.shape == (4, 3)
    assert [l.id for l in smap.drivable_lanes()] == [100, 101, 104]
    assert {l.id for l in smap.lanes(kinds=["road"])} == {100, 101, 104}
    # Elements are shared by every query of the cached map: read-only.
    for array in (lanes[100].left_points, lanes[100].right_points, lanes[100].centerline,
                  smap.lines()[0].points, smap.stop_lines()[0].points, smap.traffic_signs()[0].points):
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] += 1.0


def test_region_queries(smap: SceneMap) -> None:
    inside_100 = Circle((1.75, 10.0), 0.0)
    assert [l.id for l in smap.lanes(inside_100)] == [100]
    assert {l.id for l in smap.lanes(Circle((3.5, 21.0), 0.5))} == {100, 101, 103}
    ahead = Region(ahead=20, behind=1, left=1, right=1).around(ego_pose(1.75, 30.0, NORTH))
    assert [l.id for l in smap.lanes(ahead)] == [100, 104]
    assert {l.id for l in smap.lines(Circle((7, 30), 0.1))} == {3}
    assert {l.type for l in smap.lines(types=["stop_line", "traffic_light"])} == {"stop_line", "traffic_light"}
    assert smap.lanes(Circle((500, 500), 10)) == []


def test_stop_lines_signs_lights(smap: SceneMap) -> None:
    stops = smap.stop_lines()
    assert [(s.id, s.lane_ids) for s in stops] == [(8, (100, 101))]
    signs = {s.id: s for s in smap.traffic_signs()}
    assert set(signs) == {10, 11} and signs[11].subtype == "unknown"
    assert not signs[10].has_height and np.isnan(signs[10].points[:, 2]).all()
    assert signs[11].has_height
    assert [s.id for s in smap.traffic_signs(require_height=True)] == [11]
    (light,) = smap.traffic_lights()
    assert light.id == 9 and not light.pedestrian
    assert light.lane_ids == (100, 101) and light.stop_line_ids == (8,)
    np.testing.assert_allclose(light.points[:, 2], 105.0)


def test_match_route(smap: SceneMap) -> None:
    assert [m.lane_id for m in smap.match(ego_pose(1.75, 10, NORTH))] == [100]
    assert [m.lane_id for m in smap.match(ego_pose(5.25, 10, SOUTH))] == [101]
    assert smap.match(ego_pose(5.25, 10, NORTH)) == []  # wrong way
    assert smap.match(ego_pose(-1.2, 10, NORTH)) == []  # on the bike lane, 1.2 m from the road
    assert smap.route(100, 104) == [100, 104]
    assert smap.route(104, 100) is None


def test_patch_in_ego_and_projection(smap: SceneMap) -> None:
    pose = ego_pose(1.75, 10.0, NORTH, z=102.0)
    patch = smap.query(Circle((1.75, 10.0), 15.0)).in_ego(pose)
    assert patch.frame == "ego"
    stop = next(s for s in patch.stop_lines if s.id == 8)
    np.testing.assert_allclose(stop.points, [[8.0, 1.75, -2.0], [8.0, -1.75, -2.0]], atol=1e-4)  # 8 m ahead, 2 m below
    sign = next(s for s in patch.traffic_signs if s.id == 10)
    assert np.isnan(sign.points[:, 2]).all() and np.isfinite(sign.points[:, :2]).all()
    # A forward camera 1.5 m above the base origin sees the stop line's centre straight ahead.
    T = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]], dtype=float)
    cam = Camera("camera_ring_front", T, np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]]), (100, 100))
    uv, depth, visible = project_to_camera(np.array([[1.75, 18.0, 100.0], [8.5, 10.0, np.nan]]), pose, cam)
    np.testing.assert_allclose(uv[0], [50, 40 + 100 * 3.5 / 8], atol=1e-4)
    assert visible.tolist() == [True, False]


def test_pickle(smap: SceneMap) -> None:
    smap.lanes()
    clone = pickle.loads(pickle.dumps(smap))
    assert "_index" not in clone.__dict__ and len(clone.lanes()) == 5


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SceneMap(tmp_path / "none.osm", ORIGIN)


def test_lanelet2_version_parsing() -> None:
    from kitscenes.map._lanelet2 import _version_tuple

    assert _version_tuple("1.2.3") == (1, 2, 3)
    assert _version_tuple("1.2.2+kitscenes") == (1, 2, 2)
    assert _version_tuple("1.2.3.post1") == (1, 2, 3)
    assert _version_tuple("1.3.0rc1") == (1, 3, 0)
    assert _version_tuple("1.2.2") < _version_tuple("1.2.3") <= _version_tuple("1.10.0")


def test_scene_map_through_the_dataset(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2")
    import kitscenes
    scene_dir = make_scene(tmp_path / "data" / "train" / "s", maps=True)
    write_osm(scene_dir / "maps" / "map.osm")
    (tmp_path / "lanelet2_hd_maps").mkdir()
    city = write_osm(tmp_path / "lanelet2_hd_maps" / "Karlsruhe.osm")
    ds = kitscenes.open(tmp_path)
    scene = ds[0]
    assert scene.origin == ORIGIN
    assert [l.id for l in scene.map.lanes(kinds=["road"])] == [100, 101, 104]
    assert ds.city_maps() == {"Karlsruhe": city}
    assert len(scene.load_map(city).lanes()) == 5


def _write_broken_osm(path: Path) -> Path:
    """Two lanelets share their left bound, so lanelet 100 has two right neighbours: a routing-graph error."""
    nodes, ways, rels = [], [], []

    def node(nid, x, y):
        lat, lon = ORIGIN.to_latlon(np.array([[x, y]]))[0]
        nodes.append(f'  <node id="{nid}" lat="{lat:.10f}" lon="{lon:.10f}"><tag k="ele" v="100.0"/></node>')

    for i, x in enumerate((0.0, 3.5, 7.0, 7.2)):
        node(10 + i, x, 0.0)
        node(20 + i, x, 40.0)
    line = '<tag k="type" v="line_thin"/><tag k="subtype" v="dashed"/>'
    for wid, i in ((1, 0), (2, 1), (3, 2), (4, 3)):
        ways.append(f'  <way id="{wid}"><nd ref="{10 + i}"/><nd ref="{20 + i}"/>{line}</way>')
    for rid, left, right in ((100, 1, 2), (101, 2, 3), (102, 2, 4)):
        rels.append(f'  <relation id="{rid}"><member type="way" ref="{left}" role="left"/>'
                    f'<member type="way" ref="{right}" role="right"/><tag k="type" v="lanelet"/>'
                    '<tag k="subtype" v="road"/><tag k="location" v="urban"/><tag k="one_way" v="yes"/></relation>')
    path.write_text("\n".join(['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">', *nodes, *ways, *rels, "</osm>"]))
    return path


def test_a_map_with_routing_problems_loads_and_warns(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2")
    from kitscenes.map import MapProblemWarning

    smap = SceneMap(_write_broken_osm(tmp_path / "broken.osm"), ORIGIN)
    with pytest.warns(MapProblemWarning, match="MAP PROBLEMS in broken.osm"):
        lanes = smap.lanes()
    assert [lane.id for lane in lanes] == [100, 101, 102]
    assert smap.routing_problems and any("100" in p for p in smap.routing_problems)
    assert len(smap.match(ego_pose(1.75, 10.0, NORTH))) >= 1  # queries still work


def test_lane_change_across_a_bike_lane(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2")
    from kitscenes.map import _lanelet2

    if not _lanelet2.has_bike_lane_lane_change():
        pytest.skip("the installed lanelet2 has no lane change across bicycle lanes (PR #434)")
    smap = SceneMap(write_bike_lane_osm(tmp_path / "bike.osm"), ORIGIN)
    lanes = {lane.id: lane for lane in smap.lanes()}
    assert lanes[301].kind == LaneKind.of("bicycle_lane") and not lanes[301].vehicle
    assert lanes[300].right_neighbor == 302 and lanes[302].left_neighbor == 300
    assert lanes[300].left_neighbor is None and lanes[302].right_neighbor is None
    assert smap.route(300, 302) == [300, 302]


def test_region_and_circle_around_a_pose() -> None:
    """Region.around gives the rectangle around the vehicle, Circle.around the circle; a 3D centre drops z."""
    pose = ego_pose(1.0, 2.0, NORTH)
    box = Region(ahead=10, behind=1, left=2, right=3).around(pose)
    assert isinstance(box, Box) and box.center == (1.0, 2.0) and (box.ahead, box.behind, box.left, box.right) == (10, 1, 2, 3)
    assert Circle.around(pose, 5.0) == Circle((1.0, 2.0), 5.0)
    assert Circle(np.array([1.0, 2.0, 3.0]), 5.0).center == (1.0, 2.0)


def test_bike_lane_warning_names_what_is_missing_and_the_caller(tmp_path: Path) -> None:
    """The warning says what is off and what to install, attributed to the user's line."""
    pytest.importorskip("lanelet2")
    import warnings

    from kitscenes.map import _lanelet2

    if _lanelet2.has_bike_lane_lane_change():
        pytest.skip("this lanelet2 has TrafficRules.canCrossBoundary")
    _lanelet2._WARNED_NO_SKIP_OVER.discard("SceneMap.routing_graph")
    fresh = SceneMap(write_osm(tmp_path / "map.osm"), ORIGIN)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh.routing_graph
    (warning,) = [w for w in caught if "bicycle lane" in str(w.message)]
    assert "SceneMap.routing_graph" in str(warning.message) and "kitscenes[map]" in str(warning.message)
    assert warning.filename == __file__
