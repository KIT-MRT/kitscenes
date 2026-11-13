"""Sensor name enums and Literal types; typed point columns."""

from __future__ import annotations

import json
import pickle
import typing
from pathlib import Path

import numpy as np
import pytest

import kitscenes
from kitscenes import CameraName, LidarName, RadarName, Scene
from kitscenes.constants import (
    ALL_SENSOR_NAMES,
    CAMERA_BASE_NAMES,
    CAMERA_NAMES,
    CAMERA_RING_NAMES,
    LIDAR_NAMES,
    RADAR_NAMES,
    CameraNameStr,
    LidarNameStr,
    RadarNameStr,
    sensor_name,
    sensor_names,
)
from release_fixture import make_scene, write_ground_labels


def _plain(value) -> bool:
    return type(value) is str


# ---------------------------------------------------------------------------
# The names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "enum, literal, names",
    [(CameraName, CameraNameStr, CAMERA_NAMES), (LidarName, LidarNameStr, LIDAR_NAMES), (RadarName, RadarNameStr, RADAR_NAMES)],
)
def test_enum_literal_and_tuple_agree(enum, literal, names) -> None:
    assert tuple(m.value for m in enum) == typing.get_args(literal) == names
    assert all(_plain(n) for n in names)


def test_camera_groups_and_all_names() -> None:
    assert CAMERA_BASE_NAMES == ("camera_base_front_center", "camera_base_front_left_rect", "camera_base_front_right_rect")
    assert CAMERA_RING_NAMES[0] == "camera_ring_front" and len(CAMERA_RING_NAMES) == 6
    assert ALL_SENSOR_NAMES == CAMERA_NAMES + LIDAR_NAMES + RADAR_NAMES


def test_members_behave_like_their_strings() -> None:
    top = LidarName.TOP
    assert top == "lidar_top" and hash(top) == hash("lidar_top")
    assert {"lidar_top": 1}[top] == 1 and "lidar_top" in {top}
    assert str(top) == f"{top}" == "{}".format(top) == "%s" % top == "lidar_top"
    assert f"{top}/0000000001.parquet" == "lidar_top/0000000001.parquet"
    assert json.dumps({top: top}) == '{"lidar_top": "lidar_top"}'
    assert LidarName("lidar_top") is top


def test_sensor_name_normalizes_to_plain_strings() -> None:
    assert _plain(sensor_name(CameraName.RING_FRONT)) and sensor_name(CameraName.RING_FRONT) == "camera_ring_front"
    assert sensor_name("lidar_top") == "lidar_top"
    assert sensor_names(LidarName.TOP) == ("lidar_top",)
    names = sensor_names([LidarName.TOP, "lidar_front"])
    assert names == ("lidar_top", "lidar_front") and all(_plain(n) for n in names)
    with pytest.raises(TypeError, match="sensor name"):
        sensor_name(3)


# ---------------------------------------------------------------------------
# Accessors take members and return plain strings
# ---------------------------------------------------------------------------


@pytest.fixture
def scene(tmp_path: Path) -> Scene:
    path = make_scene(
        tmp_path / "data" / "train" / "scene_a",
        cameras=("camera_ring_front",), lidars=("lidar_top", "lidar_front"), radars=("radar_front",),
    )
    return Scene(path, split="train")


def test_frame_accessors_accept_members(scene: Scene) -> None:
    frame = scene.frame(1)
    np.testing.assert_array_equal(frame.image(CameraName.RING_FRONT), frame.image("camera_ring_front"))
    assert frame.image_bytes(CameraName.RING_FRONT) == frame.image_bytes("camera_ring_front")
    assert frame.sensor_file(CameraName.RING_FRONT).sensor == "camera_ring_front"
    assert _plain(frame.sensor_file(CameraName.RING_FRONT).sensor)
    assert frame.availability(LidarName.TOP) is kitscenes.Availability.PRESENT
    assert frame.availability(LidarName.CORNER_LEFT) is kitscenes.Availability.NOT_DOWNLOADED

    cloud = frame.lidar(LidarName.TOP)
    np.testing.assert_array_equal(cloud.xyz, frame.lidar("lidar_top").xyz)
    assert _plain(cloud.sensor)
    fused = frame.fused_lidar([LidarName.TOP, "lidar_front"])
    assert fused.sensor == ("lidar_top", "lidar_front") and all(_plain(s) for s in fused.sensor)
    assert frame.lidar_deskew_inputs(LidarName.TOP) is not None
    radar = frame.radar(RadarName.FRONT)
    assert _plain(radar.sensor) and len(radar) == len(frame.radar("radar_front"))

    assert scene.frame_indices(LidarName.TOP).tolist() == [0, 1, 2]
    assert scene.availability(RadarName.FRONT) is kitscenes.Availability.PRESENT
    assert scene.frame(0).sensor_file(LidarName.TOP).sensor == "lidar_top"
    assert scene.rig.cameras[CameraName.RING_FRONT].name == "camera_ring_front"
    assert scene.rig.sensor(LidarName.TOP) is scene.rig.lidars["lidar_top"]


def test_outputs_pickle_without_the_enums(scene: Scene) -> None:
    frame = scene.frame(0)
    payload = pickle.dumps([frame.lidar(LidarName.TOP), frame.fused_lidar([LidarName.TOP]), frame.radar(RadarName.FRONT)])
    assert b"LidarName" not in payload and b"RadarName" not in payload
    try:
        frame.lidar(LidarName.CORNER_LEFT)
    except kitscenes.SensorUnavailable as err:
        assert b"LidarName" not in pickle.dumps(err)


def test_ground_mask_accepts_members(scene: Scene) -> None:
    n = len(scene.frame(0).lidar("lidar_top"))
    write_ground_labels(scene.files.path, "lidar_top", 0, [True] * n)
    scene._refresh()
    assert scene.frame(0).ground_mask(LidarName.TOP).tolist() == [True] * n


def test_sample_spec_stores_plain_strings() -> None:
    spec = kitscenes.SampleSpec(cameras=CameraName.RING_FRONT, lidars=[LidarName.TOP, "lidar_front"], radars=(RadarName.FRONT,))
    assert spec.cameras == ("camera_ring_front",) and spec.lidars == ("lidar_top", "lidar_front")
    assert all(_plain(s) for s in spec.sensors)
    assert b"Name" not in pickle.dumps(spec.sensors)


# ---------------------------------------------------------------------------
# Typed columns
# ---------------------------------------------------------------------------


def test_lidar_columns_as_properties(scene: Scene) -> None:
    cloud = scene.frame(1).lidar(LidarName.TOP)
    assert cloud.reflectivity is cloud.attrs["reflectivity"]
    assert cloud.ring is cloud.attrs["ring"] and cloud.row is cloud.attrs["row"]
    t = cloud.point_t_ns
    assert t.dtype == np.int64
    np.testing.assert_array_equal(t, np.rint(cloud.attrs["timestamp"] * 1e9).astype(np.int64))
    with pytest.raises(AttributeError, match="no 'frame_index' column"):
        cloud.frame_index


def test_radar_columns_as_properties(scene: Scene) -> None:
    radar = scene.frame(1).radar(RadarName.FRONT)
    for name in ("range", "azimuth", "elevation", "range_rate", "rcs"):
        assert getattr(radar, name) is radar.attrs[name]
    assert radar.point_t_ns.dtype == np.int64
    assert not hasattr(radar, "reflectivity")
