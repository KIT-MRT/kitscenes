"""kitscenes.navigation and the SD-map accessors of Dataset, Scene and Frame, on synthetic scenes."""

from __future__ import annotations

import dataclasses
import json
import math

import numpy as np
import pytest

import kitscenes
from kitscenes.errors import Availability, MapUnavailable
from kitscenes.map.origin import CITY_ORIGINS
from kitscenes.navigation import (
    KITSCENES, NAVSIM, NUSCENES, Command, CommandRule, driving_commands, frame_poses, path_command,
)
from kitscenes.splits import SceneInfo

from release_fixture import STEP_NS, frame_ns, make_scene, seconds, write_poses
from sdmap_fixture import write_osm


def _polyline_poses(points, speed: float = 10.0):
    """(x, y, yaw) every 0.1 s at *speed* along a polyline."""
    from kitscenes.sdmap._geometry import arc_length, heading_at, point_at

    p = np.asarray(points, float)
    s = arc_length(p)
    at = np.arange(0.0, s[-1], speed * STEP_NS / 1e9)
    xy = point_at(p, s, at)
    yaw = np.array([heading_at(p, s, a, 3.0) for a in at])
    return xy, yaw


def _write_track(scene, xy, yaw, z: float = 115.0) -> None:
    lines = [f"{seconds(frame_ns(i), 9)} {x:.6f} {y:.6f} {z:.6f} 0.0 0.0 {math.sin(h / 2):.9f} {math.cos(h / 2):.9f}"
             for i, ((x, y), h) in enumerate(zip(xy, yaw))]
    (scene / "poses.txt").write_text("\n".join(lines) + "\n")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    """A dataset with the SD-map fixture and three scenes: a left turn at the crossroads (train, with a map
    origin), the same drive without maps (test; city from a patched scene table) and a test-e2e scene."""
    root = tmp_path_factory.mktemp("sdroot")
    write_osm(root / "openstreetmap_sd_maps" / "Karlsruhe_2026-W06.osm")
    xy, yaw = _polyline_poses([(120, 198), (298, 198), (298, 380)])
    n = len(xy)
    for split, sid in (("train", "turn-left"), ("test", "test-turn")):
        scene = make_scene(root / "data" / split / sid, n=n, cameras=(), lidars=(), radars=(), poses=False)
        _write_track(scene, xy, yaw)
        if split == "train":
            (scene / "maps").mkdir()
            o = CITY_ORIGINS["Karlsruhe"]
            (scene / "maps" / "origin.json").write_text(json.dumps({"latitude": o.lat, "longitude": o.lon}))
    e2e = make_scene(root / "data" / "test-e2e" / "e2e-scene", n=41, cameras=(), lidars=(), radars=(), poses=False)
    write_poses(e2e, 41, origin=(0.0, 0.0, 0.0))
    return root


@pytest.fixture()
def ds(root, monkeypatch, tmp_path):
    monkeypatch.setenv("KITSCENES_CACHE", str(tmp_path))
    real = kitscenes.splits.scene_info

    def scene_info(scene_id, *args, **kwargs):
        if scene_id == "test-turn":
            return SceneInfo("test-turn", "test", "Karlsruhe", "hesai", 0)
        return real(scene_id, *args, **kwargs)

    monkeypatch.setattr(kitscenes.splits, "scene_info", scene_info)
    return kitscenes.open(root)


def test_command_onehot():
    assert Command.LEFT.onehot().tolist() == [1, 0, 0, 0]
    assert Command.UNKNOWN.onehot("navsim").tolist() == [0, 0, 0, 1]
    assert Command.RIGHT.onehot("vad").tolist() == [1, 0, 0] and Command.UNKNOWN.onehot("vad").tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        Command.LEFT.onehot("carla")


def test_rules_validate():
    with pytest.raises(ValueError):
        CommandRule("x", "lateral")
    with pytest.raises(ValueError):
        CommandRule("x", "future_pose")
    assert KITSCENES.needs_route and not NAVSIM.needs_route


def test_path_command():
    p = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 30.0]])
    s = np.array([0.0, 20.0, 50.0])
    assert path_command(p, s, 0.0, p[0], 0.0, NAVSIM) == Command.STRAIGHT
    assert path_command(p, s, 10.0, [10.0, 0.0], 0.0, NAVSIM) == Command.LEFT
    assert path_command(p, s, 40.0, [20.0, 20.0], math.pi / 2, NAVSIM) == Command.UNKNOWN
    measured_from_ego = CommandRule("ego", "lateral", lookahead_m=20.0, from_projection=False)
    assert path_command(p, s, 0.0, [0.0, -3.0], 0.0, measured_from_ego) == Command.LEFT


def test_sd_map_accessors(ds, root):
    smap = ds.sd_map("karlsruhe")
    assert smap.city == "Karlsruhe" and smap is ds["turn-left"].sd_map  # one map per file and frame
    assert ds["test-turn"].sd_map is smap  # test: placed with the city's origin
    assert ds["turn-left"].availability("sd_map") is Availability.PRESENT
    e2e = ds["e2e-scene"]
    assert e2e.availability("sd_map") is Availability.WITHHELD_BY_SPLIT
    with pytest.raises(MapUnavailable, match="relative to the scene's first pose"):
        e2e.sd_map
    with pytest.raises(MapUnavailable):
        e2e.sd_route
    with pytest.raises(ValueError):
        ds.sd_map("Berlin")
    assert ds["turn-left"].load_sd_map(root / "openstreetmap_sd_maps" / "Karlsruhe_2026-W06.osm") is smap


def test_missing_sd_map_file(tmp_path):
    scene = make_scene(tmp_path / "data" / "train" / "s1", n=5, cameras=(), lidars=(), radars=())
    (scene / "maps").mkdir()
    o = CITY_ORIGINS["Karlsruhe"]
    (scene / "maps" / "origin.json").write_text(json.dumps({"latitude": o.lat, "longitude": o.lon}))
    ds = kitscenes.open(tmp_path)
    assert ds["s1"].availability("sd_map") is Availability.NOT_DOWNLOADED
    with pytest.raises(MapUnavailable):
        ds.sd_map("Karlsruhe")


def test_sd_route_and_commands(ds):
    for sid in ("turn-left", "test-turn"):
        scene = ds[sid]
        route = scene.sd_route
        assert route is not None and route is scene.sd_route  # computed once
        assert [scene.sd_map.road(r).way_id for r in route.road_ids] == [10, 12]
        ours = driving_commands(scene, KITSCENES)
        navsim = driving_commands(scene, NAVSIM)
        nusc = driving_commands(scene, NUSCENES)
        navsim_sd = driving_commands(scene, NAVSIM, source="sd_route")
        turn_frame = 178  # at the crossroads (178 m driven)
        assert ours[0] == Command.STRAIGHT and ours[150] == Command.LEFT and ours[turn_frame] == Command.LEFT
        assert ours[230] == Command.STRAIGHT
        assert navsim[170] == Command.LEFT and navsim[0] == Command.STRAIGHT and navsim[-1] == Command.UNKNOWN
        assert nusc[160] == Command.LEFT and nusc[-1] == Command.UNKNOWN
        assert navsim_sd[170] == Command.LEFT
        assert ours.dtype == np.int8 and len(ours) == scene.num_frames
    with pytest.raises(ValueError):
        driving_commands(ds["turn-left"], KITSCENES, source="path")


def test_frame_navigation(ds):
    from kitscenes.sdmap.elements import T_base_world

    scene = ds["turn-left"]
    frame = scene.frame(150)
    # default: one goal point, the end of the scene's SD route, the same in the world for every frame
    nav = frame.navigation(KITSCENES)
    assert nav.command == Command.LEFT and nav.rule == "kitscenes" and nav.source == "sd_route" and nav.targets == "goal"
    assert nav.target_points.shape == (1, 2) and nav.route_ego is None and nav.maneuver is None
    end = scene.sd_route.points[-1, :2]

    def in_world(f, xy):
        T = np.linalg.inv(T_base_world(f.pose))
        return xy @ T[:2, :2].T + T[:2, 3]

    assert np.allclose(in_world(frame, nav.target_points), end, atol=1e-6)
    early = scene.frame(20)
    assert np.allclose(in_world(early, early.navigation(KITSCENES).target_points), end, atol=1e-6)
    # the goal is the same for every rule: the SD route's end by default, the exact last pose with goal="path"
    navsim = frame.navigation(NAVSIM, source="path")
    assert navsim.goal == "sd_route" and np.allclose(in_world(frame, navsim.target_points), end, atol=1e-6)
    path_goal = frame.navigation(NAVSIM, source="path", goal="path")
    fp = frame_poses(scene)
    assert path_goal.goal == "path" and np.allclose(in_world(frame, path_goal.target_points), fp.xyz[fp.valid][-1, :2])
    with pytest.raises(ValueError):
        frame.navigation(NAVSIM, goal="lane")
    with pytest.raises(ValueError):
        frame.navigation(KITSCENES, targets="ahead")
    # targets="route": the exit of the turn (ahead, a little left), then the point 50 m ahead
    nav = frame.navigation(KITSCENES, targets="route")
    assert nav.maneuver.kind == "turn" and nav.distance_to_maneuver_m == pytest.approx(30.0, abs=3.0)
    assert nav.target_points.shape == (2, 2) and nav.target_points[0, 0] > 25 and nav.target_points[1, 1] > 15
    path = frame.navigation(NAVSIM, source="path", targets="route", spacing_m=10.0)
    assert path.source == "path" and path.target_points.shape == (5, 2) and path.maneuver is None
    assert np.allclose(path.target_points[0], [10.0, 0.0], atol=0.5)
    e2e = ds["e2e-scene"].frame(10)
    e2e_nav = e2e.navigation(NAVSIM)
    assert e2e_nav.command in (Command.STRAIGHT, Command.UNKNOWN)
    assert e2e_nav.target_points.shape == (0, 2)  # the map frame is withheld there: no goal
    with pytest.raises(MapUnavailable):
        e2e.navigation(KITSCENES)


def test_frame_poses(ds):
    fp = frame_poses(ds["turn-left"])
    assert fp.valid.all() and fp.s_m[0] == 0 and fp.s_m[100] == pytest.approx(100.0, abs=0.5)


def test_samples_with_sd_map_and_navigation(ds):
    from kitscenes.map import Region

    spec = kitscenes.SampleSpec(sd_map=kitscenes.SDMapSpec(region=Region(40, 20, 15, 15), points=10),
                                navigation=kitscenes.NavigationSpec(KITSCENES, num_targets=3), require=False,
                                frame="road")
    frames = kitscenes.FrameList(ds, [("turn-left", 150), ("e2e-scene", 5), ("test-turn", 20)])
    samples = kitscenes.SampleDataset(ds, spec, frames)
    a, e2e, test = samples[0], samples[1], samples[2]
    assert a["sd_map"]["available"] and a["sd_map"]["xyz"].shape == (10 * len(a["sd_map"]["length"]), 3)
    # the road frame: the roads on the road, z = 0; in the base frame (the default) 2.05 m below it
    assert 10 in a["sd_map"]["way_id"].tolist() and np.allclose(a["sd_map"]["xyz"][:, 2], 0.0, atol=0.01)
    base = kitscenes.SampleDataset(ds, dataclasses.replace(spec, frame="base"), frames)[0]
    assert np.allclose(base["sd_map"]["xyz"][:, 2], -2.05, atol=0.01)
    assert np.allclose(base["sd_map"]["xyz"][:, :2], a["sd_map"]["xyz"][:, :2])
    k = a["sd_map"]["way_id"].tolist().index(10)
    assert a["sd_map"]["road_class"][k] == kitscenes.sdmap.RoadClass.PRIMARY.rank and a["sd_map"]["lanes"][k] == 4
    assert a["navigation"]["command"] == Command.LEFT and a["navigation"]["target_valid"].tolist() == [True, False, False]
    route = kitscenes.SampleSpec(navigation=kitscenes.NavigationSpec(KITSCENES, targets="route", num_targets=3), require=False)
    assert kitscenes.SampleDataset(ds, route, frames)[0]["navigation"]["target_valid"].tolist() == [True, True, False]
    assert not e2e["sd_map"]["available"] and e2e["sd_map"]["state"] == "withheld_by_split"
    assert e2e["navigation"]["state"] == "withheld_by_split" and e2e["navigation"]["command"] == Command.UNKNOWN
    assert test["sd_map"]["available"] and test["navigation"]["available"]
    batch = kitscenes.collate([a, e2e, test])
    assert batch["sd_map"]["available"].tolist() == [True, False, True]
    assert batch["sd_map"]["batch"].tolist().count(1) == 0
    assert tuple(batch["navigation"]["target_points"].shape) == (3, 3, 2)
    road = kitscenes.SampleSpec(frame="road", sd_map=kitscenes.SDMapSpec(points=5), require=False)
    r = kitscenes.SampleDataset(ds, road, frames)[0]
    assert np.allclose(r["sd_map"]["xyz"][:, 2], 0.0, atol=0.01)
    with pytest.raises(TypeError):
        kitscenes.NavigationSpec("navsim")
