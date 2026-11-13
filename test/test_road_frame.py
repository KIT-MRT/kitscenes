"""The road frame: the base frame moved down onto the road (``scene.T_road_base``, ``frame="road"``)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import kitscenes
from kitscenes import Scene, invert_transform, transform_points
from kitscenes.constants import DEFAULT_ROAD_HEIGHT_M, ROAD_HEIGHT_M
from kitscenes.ego_trajectory import relative_trajectory, slot_times
from kitscenes.motion import Trajectory, apply_deskew_inputs
from kitscenes.schema import EgoPose
from release_fixture import make_scene, pack_scene

HESAI_SCENE = "000253bb-8ceb-acea-861e-3a080871f935"  # in the scene table: hesai, train
VELODYNE_SCENE = "000e3d6f-b2cb-ad4c-3714-123efb8c1950"  # velodyne, train


@pytest.fixture(params=["folder", "zip"])
def scene(tmp_path: Path, request) -> Scene:
    path = make_scene(tmp_path / "data" / "train" / HESAI_SCENE, lidars=("lidar_top", "lidar_front"),
                      radars=("radar_front",))
    if request.param == "zip":
        pack_scene(path)
    return Scene(path)


def test_road_height_per_lidar_generation(tmp_path: Path) -> None:
    for scene_id, generation in ((HESAI_SCENE, "hesai"), (VELODYNE_SCENE, "velodyne"), ("not_listed", None)):
        scene = Scene(make_scene(tmp_path / "data" / "train" / scene_id))
        expected = ROAD_HEIGHT_M[generation] if generation else DEFAULT_ROAD_HEIGHT_M
        assert scene.road_height_m == expected
        T = scene.T_road_base
        np.testing.assert_array_equal(T[:3, :3], np.eye(3))
        np.testing.assert_array_equal(T[:3, 3], [0.0, 0.0, expected])
        T[0, 0] = 5.0  # a new array every time
        assert scene.T_road_base[0, 0] == 1.0
    assert ROAD_HEIGHT_M["hesai"] == pytest.approx(2.055) and ROAD_HEIGHT_M["velodyne"] == pytest.approx(2.04)


def test_point_on_the_road_is_at_zero(scene: Scene) -> None:
    on_road_base = np.array([[5.0, 1.0, -scene.road_height_m]])
    np.testing.assert_allclose(transform_points(scene.T_road_base, on_road_base), [[5.0, 1.0, 0.0]])


def test_sensor_data_in_the_road_frame(scene: Scene) -> None:
    f = scene.frame(1)
    h = scene.road_height_m
    base, road = f.lidar("lidar_top", rows=True), f.lidar("lidar_top", frame="road", rows=True)
    assert road.frame == "road" and road.motion_compensated == base.motion_compensated
    np.testing.assert_allclose(road.xyz[road.measured], base.xyz[base.measured] + [0.0, 0.0, h], atol=1e-5)
    assert not road.xyz[~road.measured].any()  # rows without a point stay at the origin
    np.testing.assert_allclose(f.lidar("lidar_top", frame="road").xyz, road.xyz[road.measured])
    # The road pose puts road-frame points where the base pose puts the base-frame points.
    np.testing.assert_allclose(transform_points(f.T_world_road, road.xyz[road.measured]),
                               transform_points(f.T_world_base, base.xyz[base.measured]), atol=1e-3)
    np.testing.assert_allclose(f.T_world_road, f.T_world_base @ invert_transform(scene.T_road_base))

    fused = f.fused_lidar(frame="road")
    assert fused.frame == "road"
    np.testing.assert_allclose(fused.xyz, f.fused_lidar().xyz + [0.0, 0.0, h], atol=1e-5)

    radar = f.radar("radar_front", frame="road")
    plain = f.radar("radar_front")
    assert radar.frame == "road" and radar.doppler_compensated == plain.doppler_compensated
    np.testing.assert_allclose(radar.xyz[radar.measured], plain.xyz[plain.measured] + [0.0, 0.0, h], atol=1e-5)

    inputs = f.lidar_deskew_inputs("lidar_top", frame="road")
    np.testing.assert_allclose(apply_deskew_inputs(inputs), road.xyz, atol=1e-5)
    np.testing.assert_allclose(apply_deskew_inputs(f.lidar_deskew_inputs("lidar_top")), base.xyz, atol=1e-6)


def test_road_frame_choices_are_checked(scene: Scene) -> None:
    f = scene.frame(0)
    for call in (lambda: f.lidar_deskew_inputs("lidar_top", frame="world"), lambda: f.future(1.0, frame="sensor"),
                 lambda: f.past(1.0, frame="world"), lambda: f.lidar("lidar_top", frame="ground")):
        with pytest.raises(ValueError, match="frame"):
            call()
    with pytest.raises(ValueError, match="frame"):
        kitscenes.PointCloud(np.zeros((0, 3), np.float32), {}, np.zeros(0, bool), "ground", 0, "x")
    assert kitscenes.PointCloud(np.zeros((0, 3), np.float32), {}, np.zeros(0, bool), "road", 0, "x").frame == "road"


def test_accumulate_in_the_road_frame(scene: Scene) -> None:
    base = scene.accumulate([0, 1, 2], "lidar_top", frame="base", reference=1)
    road = scene.accumulate([0, 1, 2], "lidar_top", frame="road", reference=1)
    assert road.frame == "road" and road.xyz.dtype == np.float32 and road.t_ns == base.t_ns
    np.testing.assert_allclose(road.xyz, base.xyz + [0.0, 0.0, scene.road_height_m], atol=1e-4)


def test_future_in_the_road_frame_of_a_flat_drive(scene: Scene) -> None:
    f = scene.frame(0)
    base, road = f.future(0.2), f.future(0.2, frame="road")
    assert base.frame == "base" and road.frame == "road"
    np.testing.assert_allclose(road.xyz_ego, base.xyz_ego, atol=1e-9)  # no pitch or roll: the same
    np.testing.assert_allclose(road.xyz_world, base.xyz_world - [0.0, 0.0, scene.road_height_m], atol=1e-9)
    past = f.past(0.2, frame="road")
    assert past.frame == "road" and not past.valid.any()


def test_relative_trajectory_of_a_frame_on_the_vehicle() -> None:
    """Pitching up a ramp: the road frame's path differs from the base frame's, as the matrices say."""
    pitch = np.deg2rad([0.0, -3.0, -6.0, -6.0])  # nose up (rotation about y, negative = up)
    poses = tuple(EgoPose(t_ns=k * 10**9, xyz=np.array([10.0 * k, 0.0, 0.3 * k * k]),
                          quat=Rotation.from_euler("y", p).as_quat()) for k, p in enumerate(pitch))
    trajectory = Trajectory.from_ego_poses(poses, max_gap_s=1.5)
    T_base_road = np.eye(4)
    T_base_road[2, 3] = -2.05
    times = slot_times(0, 3.0, 1.0)
    road = relative_trajectory(trajectory, 0, times, T_base_frame=T_base_road, frame="road")
    base = relative_trajectory(trajectory, 0, times)
    T_world_base = [p.T_world_base for p in poses]
    T_road0_world = invert_transform(T_world_base[0] @ T_base_road)
    for k in range(3):
        T_rel = T_road0_world @ T_world_base[k + 1] @ T_base_road
        np.testing.assert_allclose(road.xyz_ego[k], T_rel[:3, 3], atol=1e-6)
        np.testing.assert_allclose(Rotation.from_quat(road.quat_ego[k]).as_matrix(), T_rel[:3, :3], atol=1e-6)
        np.testing.assert_allclose(road.xyz_world[k], (T_world_base[k + 1] @ T_base_road)[:3, 3], atol=1e-6)
    assert np.abs(road.xyz_ego - base.xyz_ego).max() > 0.05  # pitch moves the road origin differently
    np.testing.assert_allclose(road.quat_ego, base.quat_ego, atol=1e-9)  # a pure translation keeps rotations


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    for name, n in (("scene_a", 4), ("scene_b", 3)):
        pack_scene(make_scene(tmp_path / "data" / "train" / name, n=n, cameras=("camera_ring_front",),
                              lidars=("lidar_top",), radars=("radar_front",)))
    return tmp_path


def test_samples_in_the_road_frame(sample_root: Path) -> None:
    from kitscenes.samples import SampleDataset, SampleSpec, collate

    ds = kitscenes.open(sample_root)
    h = ds["scene_a"].road_height_m
    assert SampleSpec().frame == "base"  # the default, as the Frame accessors
    spec = SampleSpec(cameras=("camera_ring_front",), lidars=("lidar_top",), radars=("radar_front",), future_s=0.2,
                      frame="base")
    base = SampleDataset(ds, spec)[1]
    road = SampleDataset(ds, dataclasses.replace(spec, frame="road"))[1]
    assert base["meta"]["frame"] == "base" and road["meta"]["frame"] == "road"
    assert set(road["pose"]) == {"T_world_road"} and set(road["camera"]["camera_ring_front"]) == {"image", "K", "T_cam_road", "valid"}
    T_road_base = ds["scene_a"].T_road_base
    np.testing.assert_allclose(road["pose"]["T_world_road"] @ T_road_base, base["pose"]["T_world_base"], atol=1e-9)
    np.testing.assert_allclose(road["camera"]["camera_ring_front"]["T_cam_road"] @ T_road_base,
                               base["camera"]["camera_ring_front"]["T_cam_base"], atol=1e-9)
    np.testing.assert_allclose(road["lidar"]["xyz"], base["lidar"]["xyz"] + [0.0, 0.0, h], atol=1e-5)
    np.testing.assert_allclose(road["radar"]["xyz"], base["radar"]["xyz"] + [0.0, 0.0, h], atol=1e-5)
    np.testing.assert_allclose(road["future"]["xyz_ego"], base["future"]["xyz_ego"], atol=1e-9)  # flat drive

    batch = collate([road, SampleDataset(ds, dataclasses.replace(spec, frame="road"))[5]])
    assert tuple(batch["camera"]["camera_ring_front"]["T_cam_road"].shape) == (2, 4, 4)
    assert tuple(batch["pose"]["T_world_road"].shape) == (2, 4, 4) and batch["meta"]["frame"] == ["road", "road"]

    deskew = SampleDataset(ds, dataclasses.replace(spec, frame="road", lidar="deskew_inputs"))[1]
    np.testing.assert_allclose(apply_deskew_inputs(deskew["lidar"]["lidar_top"]),
                               ds["scene_a"].frame(1).lidar("lidar_top", frame="road").xyz, atol=1e-5)

    clip = SampleDataset(ds, dataclasses.replace(spec, frame="road", clip_length=2))[0]
    assert "T_last_road" in clip[0]["pose"] and "T_last_base" not in clip[0]["pose"]
    np.testing.assert_allclose(clip[1]["pose"]["T_last_road"], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(clip[0]["pose"]["T_last_road"][:3, 3], [-1.0, 0.0, 0.0], atol=1e-6)  # 10 m/s, 0.1 s

    with pytest.raises(ValueError, match="frame"):
        SampleSpec(frame="world")


def test_map_labels_are_lifted_into_the_road_frame() -> None:
    from kitscenes.samples import _lift_polylines

    def line():
        return {"xyz": np.array([[1.0, 0.0, -2.0], [2.0, 0.0, -2.1]], np.float32), "length": np.array([2], np.int32)}

    sample = {"available": True, "lines": {"Dashed": line()}, "traffic_elements": {"StopLine": line()},
              "lane_segments": line()}
    _lift_polylines(sample, 2.0)
    for lifted in (sample["lines"]["Dashed"], sample["traffic_elements"]["StopLine"], sample["lane_segments"]):
        np.testing.assert_allclose(lifted["xyz"][:, 2], [0.0, -0.1], atol=1e-6)


def test_map_labels_in_road_frame_samples_with_the_real_converter(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map import Region
    from kitscenes.samples import SampleDataset, SampleSpec
    from map_fixture import map_root

    ds = kitscenes.open(map_root(tmp_path), splits="train")
    labels = kitscenes.MapLabelSpec(region=Region(ahead=15, behind=5, left=5, right=5))
    base = SampleDataset(ds, SampleSpec(map_labels=labels, frame="base"))[1]["map"]["lines"]
    road = SampleDataset(ds, SampleSpec(map_labels=labels, frame="road"))[1]["map"]["lines"]
    assert set(road) == set(base) != set()
    for name in base:
        np.testing.assert_allclose(road[name]["xyz"], base[name]["xyz"] + [0.0, 0.0, ds[0].road_height_m], atol=1e-5)
    for flat in (dataclasses.replace(labels, ignore_elevation=True), dataclasses.replace(labels, dims=2)):
        b = SampleDataset(ds, SampleSpec(map_labels=flat, frame="base"))[1]["map"]["lines"]
        r = SampleDataset(ds, SampleSpec(map_labels=flat, frame="road"))[1]["map"]["lines"]
        for name in b:
            np.testing.assert_allclose(r[name]["xyz"], b[name]["xyz"])
