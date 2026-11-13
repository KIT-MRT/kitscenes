"""Map labels (kitscenes.map.labels on lanelet2.ml_converter) and their sampling."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kitscenes.map import Adaptive, FixedPoints, MLConverter, Region, SceneMap, Spacing
from kitscenes.map.sampling import resample, simplify
from map_fixture import ORIGIN, ego_pose, write_bike_lane_osm, write_osm
from release_fixture import make_scene

# ---------------------------------------------------------------------------
# Sampling (numpy only)
# ---------------------------------------------------------------------------

L_SHAPE = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [10.0, 5.0, 0.0]])


def test_fixed_points_are_equally_spaced_along_the_line() -> None:
    out = resample(L_SHAPE, FixedPoints(16))
    assert len(out) == 16
    np.testing.assert_allclose(np.linalg.norm(np.diff(out, axis=0), axis=1)[[0, 5, 14]], 1.0)
    np.testing.assert_allclose(out[[0, -1]], L_SHAPE[[0, -1]])


def test_spacing_grows_with_the_length() -> None:
    assert len(resample(L_SHAPE, Spacing(1.0))) == 16  # 15 m
    assert len(resample(L_SHAPE, Spacing(2.0))) == 9   # 8 steps of 1.875 m
    assert len(resample(L_SHAPE[:2], Spacing(100.0))) == 2


def test_adaptive_keeps_the_corner_and_fills_long_runs() -> None:
    straight = np.column_stack([np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)])
    corner = np.vstack([straight, [[10.0, 5.0, 0.0]]])
    assert len(simplify(corner, 0.05)) == 3                      # the 9 inner points of the straight part go
    out = resample(corner, Adaptive(tolerance=0.05, max_spacing=4.0))
    np.testing.assert_allclose(out, [[0, 0, 0], [10 / 3, 0, 0], [20 / 3, 0, 0], [10, 0, 0], [10, 2.5, 0], [10, 5, 0]])
    assert len(resample(corner, None)) == len(corner)


def test_bad_sampling_arguments() -> None:
    with pytest.raises(ValueError):
        FixedPoints(1)
    with pytest.raises(ValueError):
        Spacing(0)
    with pytest.raises(ValueError):
        Region(ahead=-5, behind=5)


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smap(tmp_path_factory) -> SceneMap:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    return SceneMap(write_osm(tmp_path_factory.mktemp("map") / "map.osm"), ORIGIN)


def _all_points(labels) -> np.ndarray:
    return np.concatenate([m for group in (labels.lines, labels.traffic_elements) for v in group.values() for m in v])


def test_labels_are_in_the_ego_frame(smap: SceneMap) -> None:
    labels = MLConverter(smap)(ego_pose(1.75, 10.0, np.pi / 2))  # on lanelet 100, heading north
    assert "StopLine" in labels.traffic_elements and labels.traffic_element_ids["StopLine"] == [8]
    stop = labels.traffic_elements["StopLine"][0]  # 8 m ahead, x 0..3.5 of the map -> y +1.75..-1.75
    np.testing.assert_allclose(stop[:, 0], 8.0, atol=1e-4)  # the map's lat/lon round trip: micrometres
    np.testing.assert_allclose(sorted(stop[[0, -1], 1]), [-1.75, 1.75], atol=1e-4)
    np.testing.assert_allclose(stop[:, 2], -2.0, atol=1e-9)  # map at 100 m, vehicle at 102 m
    assert all(len(m) == 20 for v in labels.lines.values() for m in v)
    centerline_ids = {i for ids in labels.line_ids["Centerline"] for i in ids}
    assert {100, 104} <= centerline_ids


def test_pitch_and_roll_are_exact(smap: SceneMap) -> None:
    """The converter's own rotation order is not the inverse of yaw-pitch-roll; the wrapper compensates."""
    from scipy.spatial.transform import Rotation

    pose = ego_pose(1.75, 10.0, np.pi / 2 + 0.3, pitch=0.05, roll=-0.04)
    stop = MLConverter(smap, te_sampling=None)(pose).traffic_elements["StopLine"][0]
    raw = next(s for s in smap.stop_lines() if s.id == 8).points  # map frame
    expected = (raw - pose.xyz) @ Rotation.from_quat(pose.quat).as_matrix()
    got = stop if np.allclose(stop[0, :2], expected[0, :2], atol=0.1) else stop[::-1]
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_an_asymmetric_region_gives_the_same_labels(smap: SceneMap) -> None:
    pose = ego_pose(1.75, 10.0, np.pi / 2 + 0.2, pitch=0.03, roll=0.02)
    for dims in (3, 2):
        centred = MLConverter(smap, dims=dims, te_sampling=None)(pose)
        ahead = MLConverter(smap, dims=dims, te_sampling=None, region=Region(ahead=25, behind=2, left=12, right=4))(pose)
        np.testing.assert_allclose(ahead.traffic_elements["StopLine"][0], centred.traffic_elements["StopLine"][0], atol=1e-9)
        pts = _all_points(ahead)
        assert pts.shape[1] == dims
        assert pts[:, 0].min() > -2.1 and pts[:, 0].max() < 25.1 and pts[:, 1].min() > -4.1 and pts[:, 1].max() < 12.1


def test_sampling_options(smap: SceneMap) -> None:
    pose = ego_pose(1.75, 10.0, np.pi / 2)
    vertices = MLConverter(smap, sampling=None)(pose)
    spaced = MLConverter(smap, sampling=Spacing(1.0))(pose)
    for name, lines in vertices.lines.items():
        for raw, even in zip(lines, spaced.lines[name]):
            length = np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()
            assert len(even) == max(2, int(np.ceil(length - 1e-9)) + 1)
    assert spaced.line_ids == vertices.line_ids


def test_map_changed_flag_and_pickling(smap: SceneMap, tmp_path: Path) -> None:
    import kitscenes
    for split, changed in (("train", False), ("map_change_train", True)):
        scene_dir = make_scene(tmp_path / split / "data" / split / "s", maps=True)
        write_osm(scene_dir / "maps" / "map.osm")
        scene = kitscenes.open(tmp_path / split)[0]
        conv = MLConverter.for_scene(scene)
        assert conv(ego_pose(1.75, 10.0, np.pi / 2)).map_changed is changed
    again = pickle.loads(pickle.dumps(conv))  # used once, so it holds converter objects: they are not pickled
    assert len(again(ego_pose(1.75, 10.0, np.pi / 2))) == len(conv(ego_pose(1.75, 10.0, np.pi / 2)))


def test_for_scene_outside_a_known_split(tmp_path: Path) -> None:
    """kitscenes.Scene(path) of a scene in no split folder has no split info; for_scene must not crash."""
    pytest.importorskip("lanelet2")
    import kitscenes
    scene_dir = make_scene(tmp_path / "s", maps=True)
    write_osm(scene_dir / "maps" / "map.osm")
    scene = kitscenes.Scene(scene_dir)
    assert scene.split_info is None
    assert MLConverter.for_scene(scene).map_changed is False


def test_lane_change_edges_across_a_bike_lane(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from lanelet2.ml_converter import MapDataInterface

    if not hasattr(MapDataInterface.Configuration(), "allowLaneChangeAcrossBicycleLane"):
        pytest.skip("the installed ML converter cannot change lanes across bicycle lanes")
    smap = SceneMap(write_bike_lane_osm(tmp_path / "bike.osm"), ORIGIN)
    data = MLConverter(smap).map_data(ego_pose(4.25, 20.0, np.pi / 2))
    edges = {(e.el1, e.el2) for es in data.llEdges.values() for e in es if e.isLaneChange}
    assert {(300, 302), (302, 300)} <= edges
    assert not any(301 in e for e in edges)


def test_bike_marking_classes(tmp_path: Path) -> None:
    """From lanelet2 1.2.3+kitscenes4 on (kitscenes.map.LANELET2_BUILD) bicycle lane markings have their own classes
    in the default and road-border-merged groupings; kitscenes3 (the paper build) labels them Dashed. The MapTR
    and M3TR groupings label them as before."""
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map import _lanelet2

    build = _lanelet2._kitscenes_build(_lanelet2._installed_version() or "")
    if build is None or build < 3:
        pytest.skip("needs a KITScenes lanelet2 build 1.2.3+kitscenes3 or newer")
    smap = SceneMap(write_bike_lane_osm(tmp_path / "bike.osm"), ORIGIN)
    pose = ego_pose(4.25, 20.0, np.pi / 2)

    def members(grouping, cls):
        return {i for ids in MLConverter(smap, grouping=grouping)(pose).line_ids[cls] for i in ids}

    for grouping in ("default", "road_border_merged"):
        marking, plain = ("BikeMarkingDashed", "Dashed") if build >= 4 else ("Dashed", "BikeMarkingDashed")
        assert members(grouping, marking) == {2, 3} and not members(grouping, plain) & {2, 3}
        assert not members(grouping, "BikeMarkingSolid")
    assert members("maptr", "Divider") == {2, 3}
    assert members("m3tr", "Dashed") == {2, 3}


def test_missing_converter_message_names_the_install(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without lanelet2.ml_converter the error says which install brings it (the [map] extra's KITScenes build)."""
    from kitscenes.errors import MissingDependencyError
    from kitscenes.map import _lanelet2
    from kitscenes.map.labels import require_ml_converter

    monkeypatch.setattr(_lanelet2, "available", lambda: (True, ""))
    monkeypatch.setattr(_lanelet2, "has_ml_converter", lambda: False)
    with pytest.raises(MissingDependencyError) as err:
        require_ml_converter()
    assert 'kitscenes[map]' in str(err.value) and "KITScenes/Lanelet2" in str(err.value)


# ---------------------------------------------------------------------------
# Classes, keys, crosswalk polygons, the height filter (the MapTR preset's options)
# ---------------------------------------------------------------------------


def test_every_class_is_a_key(smap: SceneMap) -> None:
    """The key set of the labels is the grouping's (or the classes'), the same at every pose."""
    from kitscenes.map import GROUPING_CLASSES, TE_TYPES

    here, nowhere = ego_pose(1.75, 10.0, np.pi / 2), ego_pose(500.0, 500.0, 0.0)
    for grouping in ("default", "maptr", "m3tr", "road_border_merged"):
        conv = MLConverter(smap, grouping=grouping)
        for pose in (here, nowhere):
            labels = conv(pose)
            assert tuple(labels.lines) == tuple(labels.line_ids) == tuple(labels.line_pieces) == GROUPING_CLASSES[grouping]
            assert tuple(labels.traffic_elements) == tuple(labels.traffic_element_ids) == TE_TYPES
        assert len(conv(nowhere)) == 0 and conv(here).lines["Centerline"]


def test_grouping_classes_match_the_converter() -> None:
    ml = pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map import GROUPING_CLASSES, TE_TYPES
    from kitscenes.map.labels import _GROUPING_FUNCTIONS, _grouping_types

    for name, function in _GROUPING_FUNCTIONS.items():
        assert _grouping_types(getattr(ml, function)()) == GROUPING_CLASSES[name]
    assert tuple(str(t) for _, t in sorted(ml.TEType.values.items())) == TE_TYPES


def test_classes_pick_and_rename_types(smap: SceneMap) -> None:
    from kitscenes.map import MAPTR_CLASSES

    pose = ego_pose(1.75, 10.0, np.pi / 2)
    plain = MLConverter(smap, grouping="maptr")(pose)
    labels = MLConverter(smap, grouping="maptr", classes=MAPTR_CLASSES)(pose)
    assert tuple(labels.lines) == ("divider", "ped_crossing", "boundary")
    for name, (kind,) in MAPTR_CLASSES.items():
        assert labels.line_ids[name] == plain.line_ids[kind]
        for got, expected in zip(labels.lines[name], plain.lines[kind]):
            np.testing.assert_allclose(got, expected)
    assert labels.line_ids["boundary"] == [(3,)]  # the drivable_space_border line, not the curbs (RoadBorder)
    kept = MLConverter(smap, grouping="maptr", classes=["Divider", "RoadBorder"])(pose)
    assert tuple(kept.lines) == ("Divider", "RoadBorder")
    with pytest.raises(ValueError, match="Dashed"):
        MLConverter(smap, grouping="maptr", classes={"divider": ["Dashed"]})  # not a type of this grouping
    with pytest.raises(ValueError, match="two classes"):
        MLConverter(smap, grouping="maptr", classes={"a": ["Divider"], "b": ["Divider"]})


def test_traffic_element_classes_and_their_edges(smap: SceneMap) -> None:
    pose = ego_pose(1.75, 10.0, np.pi / 2)
    plain = MLConverter(smap)(pose)
    renamed = MLConverter(smap, classes={"lane": ["Centerline"]}, te_classes={"light": ["TLCar"], "stop": ["StopLine"]})(pose)
    assert tuple(renamed.traffic_elements) == ("light", "stop")
    assert renamed.traffic_element_ids["stop"] == plain.traffic_element_ids["StopLine"]
    names = {"TLCar": "light", "StopLine": "stop"}
    assert renamed.te_to_te and renamed.te_to_te == [(names[a], i, names[b], j) for a, i, b, j in plain.te_to_te
                                                     if a in names and b in names]
    assert renamed.te_to_centerline == [(names[t], i, j) for t, i, j in plain.te_to_centerline if t in names] != []
    assert MLConverter(smap, te_classes={})(pose).traffic_elements == {}


def test_line_keys_are_unique(smap: SceneMap) -> None:
    labels = MLConverter(smap, grouping="m3tr")(ego_pose(1.75, 10.0, np.pi / 2))
    for name in labels.lines:
        keys = labels.line_keys(name)
        assert len(set(keys)) == len(keys) == len(labels.lines[name])
        assert all(key[0] == name and key[1] == ids for key, ids in zip(keys, labels.line_ids[name]))


def test_crosswalk_polygons(smap: SceneMap) -> None:
    """crosswalks="polygons": a closed outline clipped by the region, where the converter's perimeter line is cut open."""
    pose = ego_pose(1.75, 10.0, np.pi / 2)  # crosswalk 103: 10..13 m ahead, 3.25 m left to 5.25 m right
    full = MLConverter(smap, grouping="maptr", crosswalks="polygons", sampling=None)(pose).lines["PedestrianCrossing"]
    assert len(full) == 1 and np.array_equal(full[0][0], full[0][-1])
    from kitscenes.map._vector import signed_area

    assert signed_area(full[0][:-1]) == pytest.approx(3 * 8.5, abs=1e-3)  # counter-clockwise
    np.testing.assert_allclose(full[0][:, 2], -2.0, atol=1e-6)

    region = Region(ahead=11.5, behind=5, left=10, right=10)  # cuts it at 11.5 m ahead
    for dims in (3, 2):
        lines = MLConverter(smap, grouping="maptr", region=region, dims=dims, sampling=None)(pose)
        assert not any(np.allclose(m[0], m[-1]) for m in lines.lines["PedestrianCrossing"])  # the converter: open
        conv = MLConverter(smap, grouping="maptr", region=region, dims=dims, crosswalks="polygons", sampling=None)
        labels = conv(pose)
        (ring,) = labels.lines["PedestrianCrossing"]
        assert ring.shape[1] == dims and np.array_equal(ring[0], ring[-1])
        assert signed_area(ring[:-1]) == pytest.approx(1.5 * 8.5, abs=1e-3)
        np.testing.assert_allclose(ring[:, 0].max(), 11.5, atol=1e-6)
        assert labels.line_ids["PedestrianCrossing"] == [(103,)] and labels.line_pieces["PedestrianCrossing"] == [0]
        fixed = MLConverter(smap, grouping="maptr", region=region, dims=dims, crosswalks="polygons")(pose)
        (ring20,) = fixed.lines["PedestrianCrossing"]
        assert len(ring20) == 20 and np.array_equal(ring20[0], ring20[-1])  # FixedPoints keeps it closed
    default = MLConverter(smap, crosswalks="polygons")(pose)  # the default grouping keeps zebra crossings apart
    assert len(default.lines["ZebraCrossing"]) == 1 and default.lines["PedestrianCrossing"] == []
    tiny = Region(ahead=10.1, behind=5, left=10, right=10)  # 0.1 m x 8.5 m < 1 m^2: dropped
    assert MLConverter(smap, grouping="maptr", region=tiny, crosswalks="polygons")(pose).lines["PedestrianCrossing"] == []


def _write_split_crossing_osm(path: Path) -> Path:
    """Crosswalk lanelets 400 (x -1.5..3) and 401 (x 3..7) at y 20..23: one crossing, split at x = 3."""
    nodes, ways, rels = [], [], []
    for i, x in enumerate((-1.5, 3.0, 7.0)):
        for j, y in enumerate((20.0, 23.0)):
            lat, lon = ORIGIN.to_latlon(np.array([[x, y]]))[0]
            nodes.append(f'  <node id="{10 + 10 * j + i}" lat="{lat:.10f}" lon="{lon:.10f}"><tag k="ele" v="100.0"/></node>')
    # ways along the walking direction (+x): 1/2 the bounds of 400 at y 23/20, 3/4 those of 401
    for wid, (a, b) in {1: (20, 21), 2: (10, 11), 3: (21, 22), 4: (11, 12)}.items():
        ways.append(f'  <way id="{wid}"><nd ref="{a}"/><nd ref="{b}"/><tag k="type" v="pedestrian_marking"/></way>')
    for rid, left, right in ((400, 1, 2), (401, 3, 4)):
        rels.append(f'  <relation id="{rid}"><member type="way" ref="{left}" role="left"/>'
                    f'<member type="way" ref="{right}" role="right"/><tag k="type" v="lanelet"/>'
                    f'<tag k="subtype" v="crosswalk"/><tag k="location" v="urban"/></relation>')
    path.write_text("\n".join(['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">', *nodes, *ways, *rels, "</osm>"]))
    return path


def test_crosswalk_lanelets_of_one_crossing_are_merged(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map._vector import signed_area

    split = SceneMap(_write_split_crossing_osm(tmp_path / "split.osm"), ORIGIN)
    pose = ego_pose(1.75, 10.0, np.pi / 2)
    assert len(MLConverter(split, grouping="maptr")(pose).lines["PedestrianCrossing"]) == 2  # the converter: two
    labels = MLConverter(split, grouping="maptr", crosswalks="polygons", sampling=None)(pose)
    (ring,) = labels.lines["PedestrianCrossing"]
    assert labels.line_ids["PedestrianCrossing"] == [(400, 401)]
    assert signed_area(ring[:-1]) == pytest.approx(3 * 8.5, abs=1e-3)


def test_height_filter_drops_other_levels(smap: SceneMap) -> None:
    """max_height_diff: lines far above or below the road are cut away; traffic elements stay."""
    on_road = ego_pose(1.75, 10.0, np.pi / 2)
    on_bridge = ego_pose(1.75, 10.0, np.pi / 2, z=111.0)  # the map's roads 9 m below the vehicle's road level
    for dims in (3, 2):
        conv = MLConverter(smap, dims=dims, max_height_diff=4.0, crosswalks="polygons")
        assert sum(len(v) for v in conv(on_bridge).lines.values()) == 0
        assert conv(on_bridge).traffic_elements["StopLine"]
        unfiltered = MLConverter(smap, dims=dims, crosswalks="polygons")(on_road)
        filtered = conv(on_road)
        for name, lines in unfiltered.lines.items():  # nothing to cut on the vehicle's level: the same labels
            assert len(filtered.lines[name]) == len(lines)
            for got, expected in zip(filtered.lines[name], lines):
                np.testing.assert_allclose(got, expected, atol=1e-6)
    with pytest.raises(ValueError, match="ignore_elevation"):
        MLConverter(smap, max_height_diff=4.0, ignore_elevation=True)


def test_height_filter_follows_the_road_poses(smap: SceneMap) -> None:
    """Without road poses the band is the plane under the vehicle; with them, the road along the drive."""
    pitched = ego_pose(1.75, 10.0, np.pi / 2, pitch=0.2)  # nose 11 degrees down on a flat map: the plane leaves it
    cut = MLConverter(smap, grouping="maptr", max_height_diff=4.0, sampling=None, road_height_m=2.0)(pitched)
    heights = np.concatenate([m[:, 2] + 2.0 for m in cut.lines["Divider"]])  # above the road plane under the vehicle
    assert np.abs(heights).max() == pytest.approx(4.0)  # cut where the lines leave the band
    whole = MLConverter(smap, grouping="maptr", sampling=None)(pitched)
    assert max(m[:, 0].max() for m in cut.lines["Divider"]) < max(m[:, 0].max() for m in whole.lines["Divider"]) - 5
    drive = [ego_pose(1.75, y, np.pi / 2) for y in np.arange(0.0, 60.0, 2.0)]
    conv = MLConverter(smap, grouping="maptr", max_height_diff=4.0, sampling=None, road_poses=drive, road_height_m=2.0)
    along = conv(pitched)
    assert [len(m) for m in along.lines["Divider"]] == [len(m) for m in whole.lines["Divider"]]
    again = pickle.loads(pickle.dumps(conv))(pitched)  # used once: the road's search tree is not pickled
    assert [len(m) for m in again.lines["Divider"]] == [len(m) for m in along.lines["Divider"]]


def test_dims_2_with_heights_matches_the_2d_mode(smap: SceneMap) -> None:
    """The height filter asks the converter for 3D at the heading; x-y stay those of its 2D mode."""
    pose = ego_pose(1.75, 10.0, np.pi / 2 + 0.2, pitch=0.03, roll=0.02)
    for region in (Region(), Region(ahead=25, behind=2, left=12, right=4)):
        flat = MLConverter(smap, dims=2, region=region, sampling=None)(pose)
        with_heights = MLConverter(smap, dims=2, region=region, max_height_diff=10.0, sampling=None)(pose)
        for name, lines in flat.lines.items():
            assert len(with_heights.lines[name]) == len(lines)
            for got, expected in zip(with_heights.lines[name], lines):
                assert got.shape == expected.shape
                np.testing.assert_allclose(got, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Scene-wide chaining
# ---------------------------------------------------------------------------


def test_scene_chaining_needs_the_fixed_converter(smap: SceneMap, monkeypatch: pytest.MonkeyPatch) -> None:
    from kitscenes.errors import MissingDependencyError
    from kitscenes.map import Circle, _lanelet2

    monkeypatch.setattr(_lanelet2, "_installed_version", lambda: "1.2.3+kitscenes1")
    with pytest.raises(MissingDependencyError, match=r"1\.2\.3\+kitscenes2.*installed one is 1\.2\.3\+kitscenes1"):
        MLConverter(smap, chaining=Circle((0.0, 20.0), 100.0))
    with pytest.raises(ValueError, match="for_scene"):
        MLConverter(smap, chaining="scene")


def test_scene_chain_area_covers_every_pose() -> None:
    from types import SimpleNamespace

    from kitscenes.map.labels import scene_chain_area

    poses = [ego_pose(x, 0.0, 0.0) for x in np.linspace(0.0, 100.0, 11)]
    region = Region(ahead=40, behind=10, left=15, right=15)
    area = scene_chain_area(SimpleNamespace(id="s", _poses=tuple(poses)), region)
    for pose in poses:  # every corner of every pose's converter square is inside
        centre = np.asarray(pose.xyz[:2]) + [15.0, 0.0]
        half = 1.1 * np.hypot(*region.half_extents)
        for corner in centre + half * np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]]):
            assert np.hypot(*(corner - area.center)) <= area.radius
    with pytest.raises(ValueError, match="no poses"):
        scene_chain_area(SimpleNamespace(id="s", _poses=()))


def test_scene_chaining_with_the_converter(smap: SceneMap) -> None:
    """Chains built once, processed per pose: an element that left the region is back when the region is
    (fails with lanelet2 1.2.3+kitscenes1, whose MapData keeps it invalid)."""
    from kitscenes.map import Circle, _lanelet2

    if not _lanelet2.has_reusable_map_data():
        pytest.skip("needs lanelet2 1.2.3+kitscenes2 or newer (MapData processed for one box after another)")
    region = Region(ahead=8, behind=8, left=8, right=8)
    chained = MLConverter(smap, grouping="maptr", region=region, chaining=Circle((2.0, 30.0), 80.0))
    per_frame = MLConverter(smap, grouping="maptr", region=region)
    poses = [ego_pose(1.75, 10.0, np.pi / 2), ego_pose(1.75, 50.0, np.pi / 2), ego_pose(1.75, 10.0, np.pi / 2)]
    first = chained(poses[0])
    away = chained(poses[1])
    assert not away.traffic_elements["StopLine"]  # the stop line at y 18 is outside the region at y 50
    again = chained(poses[2])
    for name in first.lines:
        assert again.line_ids[name] == first.line_ids[name] and again.line_pieces[name] == first.line_pieces[name]
        for got, expected in zip(again.lines[name], first.lines[name]):
            np.testing.assert_allclose(got, expected, atol=1e-9)
    assert again.traffic_element_ids == first.traffic_element_ids
    for pose in poses:  # a scene chain continues past the region: every per-frame chain is part of one
        a, b = chained(pose), per_frame(pose)
        assert set(a.line_ids) == set(b.line_ids)
        for name, frame_chains in b.line_ids.items():
            for ids in frame_chains:
                assert any(set(ids) <= set(scene_ids) for scene_ids in a.line_ids[name]), (name, ids, a.line_ids[name])
    assert any(len(ids) > 1 for ids in first.line_ids["Divider"])  # the divider chain reaches beyond the region
    # a chain seen from both poses keeps one id tuple (divider 4 is only inside the region at the first pose)
    assert set(away.line_ids["Divider"]) and set(away.line_ids["Divider"]) <= set(first.line_ids["Divider"])
    for labels in (first, away):  # the pieces the region cuts a chain into keep its ids and are numbered from 0
        for name, chains in labels.line_ids.items():
            for chain in set(chains):
                pieces = [piece for ids, piece in zip(chains, labels.line_pieces[name]) if ids == chain]
                assert sorted(pieces) == list(range(len(pieces))), (name, chain, pieces)
            assert len(set(labels.line_keys(name))) == len(chains)


def test_height_filter_uses_the_scene_road_height() -> None:
    from kitscenes.constants import DEFAULT_ROAD_HEIGHT_M
    from kitscenes.map.labels import _RoadLevel

    pose = ego_pose(0.0, 0.0, 0.0)
    for height in (DEFAULT_ROAD_HEIGHT_M, 2.055, 2.040):
        road = _RoadLevel.of([pose], height)
        assert road.height_above(np.asarray(pose.xyz, float)[None])[0] == pytest.approx(height)  # the base origin


def test_fixed_points_are_sampled_once_from_the_vertices(smap: SceneMap) -> None:
    """FixedPoints(n) is one even resampling of the converter's vertices, not a resampling of its own n points."""
    from kitscenes.map.sampling import FixedPoints, resample

    pose = ego_pose(1.75, 10.0, np.pi / 2)
    raw = MLConverter(smap, grouping="maptr", sampling=None, te_sampling=None)(pose)
    fixed = MLConverter(smap, grouping="maptr", sampling=FixedPoints(20), te_sampling=FixedPoints(10))(pose)
    assert raw.lines.keys() == fixed.lines.keys()
    for name in raw.lines:
        assert len(raw.lines[name]) == len(fixed.lines[name])
        for vertices, sampled in zip(raw.lines[name], fixed.lines[name]):
            np.testing.assert_allclose(sampled, resample(vertices, FixedPoints(20)), atol=1e-9)
    for name in raw.traffic_elements:
        for vertices, sampled in zip(raw.traffic_elements[name], fixed.traffic_elements[name]):
            np.testing.assert_allclose(sampled, resample(vertices, FixedPoints(10)), atol=1e-9)


# ---------------------------------------------------------------------------
# Lane segments and their successor graph
# ---------------------------------------------------------------------------


def _write_fork_osm(path: Path) -> Path:
    """Northbound 10 (y 0..20), 11 (y 20..40), then a fork: 12 straight on (to y 60), 13 to the right (x + 5);
    14 (y 80..60) is bidirectional, drawn southbound: it ends where 12 ends."""
    points = {"a0": (0, 0), "b0": (3.5, 0), "a1": (0, 20), "b1": (3.5, 20), "a2": (0, 40), "b2": (3.5, 40),
              "a3": (0, 60), "b3": (3.5, 60), "c3": (5, 60), "d3": (8.5, 60), "a4": (0, 80), "b4": (3.5, 80)}
    node_id = {name: 1000 + k for k, name in enumerate(points)}
    nodes = []
    for name, (x, y) in points.items():
        lat, lon = ORIGIN.to_latlon(np.array([[x, y]]))[0]
        nodes.append(f'  <node id="{node_id[name]}" lat="{lat:.10f}" lon="{lon:.10f}"><tag k="ele" v="100.0"/></node>')
    bounds = {1: "a0 a1", 2: "b0 b1", 3: "a1 a2", 4: "b1 b2", 5: "a2 a3", 6: "b2 b3", 7: "a2 c3", 8: "b2 d3",
              9: "b4 b3", 10: "a4 a3"}
    ways = [f'  <way id="{wid}">' + "".join(f'<nd ref="{node_id[n]}"/>' for n in ends.split())
            + '<tag k="type" v="line_thin"/><tag k="subtype" v="solid"/></way>' for wid, ends in bounds.items()]
    rels = [f'  <relation id="{rid}"><member type="way" ref="{left}" role="left"/><member type="way" ref="{right}" '
            'role="right"/><tag k="type" v="lanelet"/><tag k="subtype" v="road"/><tag k="location" v="urban"/>'
            f'<tag k="one_way" v="{one_way}"/></relation>'
            for rid, left, right, one_way in ((10, 1, 2, "yes"), (11, 3, 4, "yes"), (12, 5, 6, "yes"), (13, 7, 8, "yes"),
                                              (14, 9, 10, "no"))]
    path.write_text("\n".join(['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6">', *nodes, *ways, *rels, "</osm>"]))
    return path


def test_lane_segments_and_their_successors(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    fork = SceneMap(_write_fork_osm(tmp_path / "fork.osm"), ORIGIN)
    labels = MLConverter(fork, sampling=None)(ego_pose(1.75, 30.0, np.pi / 2, z=102.0))  # the box: y 0..60
    assert labels.lane_segment_ids == [(10, 11), (12,), (13,)] and labels.lane_segment_pieces == [0, 0, 0]
    assert labels.lane_to_lane == [(0, 1), (0, 2)]
    trunk, straight, right = labels.lane_segments
    np.testing.assert_allclose(trunk[[0, -1]], [[-30, 0, -2], [10, 0, -2]], atol=1e-4)  # 10 and 11 as one line
    np.testing.assert_allclose(straight[-1], [30, 0, -2], atol=1e-4)
    np.testing.assert_allclose(right[-1], [30, -5, -2], atol=1e-4)
    for i, j in labels.lane_to_lane:
        np.testing.assert_allclose(labels.lane_segments[i][-1], labels.lane_segments[j][0], atol=1e-6)
    # Cut by the region: the trunk starts at the box border, the ids stay those of the whole run.
    short = MLConverter(fork, region=Region(ahead=15, behind=15, left=15, right=15))(ego_pose(1.75, 30.0, np.pi / 2))
    assert short.lane_segment_ids == [(10, 11), (12,), (13,)] and short.lane_to_lane == [(0, 1), (0, 2)]
    np.testing.assert_allclose(short.lane_segments[0][0, :2], [-15, 0], atol=1e-4)
    flat = MLConverter(fork, dims=2)(ego_pose(1.75, 30.0, np.pi / 2))
    assert flat.lane_segments[0].shape == (20, 2) and flat.lane_to_lane == labels.lane_to_lane
    # 12 continues into 14 only against 14's direction (the routing graph's inverted 14): no edge
    far = MLConverter(fork, region=Region(ahead=60, behind=30, left=15, right=15))(ego_pose(1.75, 30.0, np.pi / 2))
    assert far.lane_segment_ids == [(10, 11), (12,), (13,), (14,)] and far.lane_to_lane == [(0, 1), (0, 2)]
    assert fork.lane(12).successors == (14,)  # what Lane.successors says


def test_centerline_paths_that_part_outside_the_region_are_one_label(tmp_path: Path) -> None:
    """The fork (y 40) is ahead of the box (y 5..35): both paths through it are the same line inside the box, one
    label. With the fork inside the box (y 15..45) they are two."""
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    fork = SceneMap(_write_fork_osm(tmp_path / "fork.osm"), ORIGIN)
    region = Region(ahead=15, behind=15, left=15, right=15)
    before = MLConverter(fork, grouping="maptr", region=region, sampling=None)(ego_pose(1.75, 20.0, np.pi / 2))
    (line,) = before.lines["Centerline"]
    np.testing.assert_allclose(line[[0, -1], :2], [[-15, 0], [15, 0]], atol=1e-4)
    over = MLConverter(fork, grouping="maptr", region=region, sampling=None)(ego_pose(1.75, 30.0, np.pi / 2))
    assert len(over.lines["Centerline"]) == 2


def test_clip_polyline_cuts_at_the_box() -> None:
    from kitscenes.map._vector import clip_polyline

    line = np.array([[-20, 0, 0], [0, 0, 1], [20, 0, 2], [20, 20, 3], [0, 20, 4], [0, -5, 5], [0, -30, 6.0]])
    pieces = clip_polyline(line, 10, 10)
    np.testing.assert_allclose(pieces[0], [[-10, 0, 0.5], [0, 0, 1], [10, 0, 1.5]])
    np.testing.assert_allclose(pieces[1], [[0, 10, 4.4], [0, -5, 5], [0, -10, 5.2]])
    assert len(pieces) == 2 and clip_polyline(np.array([[20.0, 20], [30, 30]]), 10, 10) == []


def test_lines_are_cut_exactly_to_the_region(smap: SceneMap) -> None:
    """The converter cuts in its ground plane: in a base frame pitched nose down the lines would reach 6 cm past
    the box (they are cut at it), nose up they end 14 cm short of it (the converter's cut, left as it is)."""
    for pitch, end in ((0.05, 30.0), (-0.05, 29.8625)):
        labels = MLConverter(smap, grouping="maptr", crosswalks="polygons", sampling=None)(
            ego_pose(1.75, 10.0, np.pi / 2, pitch=pitch))
        x = np.concatenate([m[:, 0] for v in labels.lines.values() for m in v])
        assert x.max() == pytest.approx(end, abs=1e-4)
        assert max(m[:, 0].max() for m in labels.lane_segments) <= 30.0 + 1e-9
