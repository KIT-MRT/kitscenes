"""Calibration: Rig, Camera projection, validation."""

from __future__ import annotations

import numpy as np
import pytest

from kitscenes.errors import CalibrationError
from kitscenes.rig import Camera, Rig, invert_transform, transform_points
from release_fixture import camera_entry, transform


def _rig(**extra) -> Rig:
    return Rig.from_calib({
        "reference": "base_frame",
        "camera_ring_front": camera_entry(),
        "lidar_top": {"T_to_reference": transform((0, 0, 1.2), yaw_deg=90)},
        "radar_front": {"T_to_reference": transform((3.5, 0, 0.5))},
        "gnss_antenna": {"T_to_reference": transform((0.5, 0, 1.5))},
        **extra,
    })


def test_rig_from_calib() -> None:
    rig = _rig()
    assert rig.sensor_names == ("camera_ring_front", "lidar_top", "radar_front")
    assert set(rig.transforms) == {"camera_ring_front", "lidar_top", "radar_front", "gnss_antenna"}
    with pytest.raises(ValueError, match="read-only"):  # shared by the frames of a cached scene
        rig.transforms["gnss_antenna"][0, 3] += 1.0
    np.testing.assert_allclose(rig.lidars["lidar_top"].T_base_sensor[:3, 3], [0, 0, 1.2])
    assert rig.sensor("lidar_top") is rig.lidars["lidar_top"]
    with pytest.raises(CalibrationError, match="no entry for 'lidar_rear'"):
        rig.sensor("lidar_rear")
    with pytest.raises(KeyError, match="Unknown sensor"):
        rig.sensor("lidar_roof")


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda e: e.update(camera_model="fisheye"), "pinhole"),
        (lambda e: e["intrinsics"].update(k1=0.1), "intrinsics"),
        (lambda e: e.pop("resolution"), "missing"),
        (lambda e: e["intrinsics"].pop("focal_length"), "intrinsics"),
        (lambda e: e.update(resolution={"width": 64}), "invalid entry for 'camera_ring_rear'"),
        (lambda e: e.update(T_to_reference=[[1, 0], [0, 1]]), r"shape \(2, 2\)"),
    ],
)
def test_invalid_camera_fails_the_rig(change, match) -> None:
    """One invalid entry: the rig does not load (no rig without that sensor)."""
    entry = camera_entry()
    change(entry)
    with pytest.raises(CalibrationError, match=match):
        _rig(camera_ring_rear=entry)


@pytest.mark.parametrize(
    "entry, match",
    [
        ({"no_transform": []}, "no T_to_reference"),
        ({"T_to_reference": [[1, 2, 3]]}, r"shape \(1, 3\)"),
        ({"T_to_reference": [[1, 2], [3]]}, "invalid entry for 'lidar_rear'"),  # ragged
        ("not an object", "invalid entry for 'lidar_rear'"),
    ],
)
def test_invalid_lidar_fails_the_rig(entry, match) -> None:
    with pytest.raises(CalibrationError, match=match):
        _rig(lidar_rear=entry)


def test_invalid_gnss_transform_fails_the_rig() -> None:
    with pytest.raises(CalibrationError, match="gnss_antenna"):
        _rig(gnss_antenna={"T_to_reference": [[1.0]]})


@pytest.mark.parametrize("data, match", [(b"{", "not valid JSON"), (b"[]", "not a JSON object"), (b"\xff", "not valid JSON")])
def test_malformed_calib_json(data, match) -> None:
    with pytest.raises(CalibrationError, match=match) as err:
        Rig.from_json_bytes(data, "scene/calibration/calib.json")
    assert "scene/calibration/calib.json" in str(err.value) and isinstance(err.value, ValueError)


def test_camera_projection() -> None:
    cam = _rig().cameras["camera_ring_front"]  # at x = 1.5, z = 1.6, looking along +x
    points = np.array([[11.5, 0.0, 1.6], [11.5, 1.0, 1.6], [11.5, 0.0, 0.6], [-5.0, 0.0, 1.6], [11.5, 50.0, 1.6]])
    uv, depth, visible = cam.project(points)
    np.testing.assert_allclose(uv[0], [32, 24])          # principal point
    np.testing.assert_allclose(uv[1], [27, 24])          # left in the world -> smaller u
    np.testing.assert_allclose(uv[2], [32, 29])          # lower -> larger v
    np.testing.assert_allclose(depth[:3], 10.0)
    assert visible.tolist() == [True, True, True, False, False]
    assert np.isnan(uv[3]).all()
    half = cam.scaled(0.5)
    assert half.image_size == (32, 24)
    # pixel centres on integers: full-size pixel 32 is the right half of half-size pixel 16
    np.testing.assert_allclose(half.project(points[:1])[0][0], [15.75, 11.75])
    np.testing.assert_allclose(cam.scaled(1 / 8).K[:2, 2], [(32 + 0.5) / 8 - 0.5, (24 + 0.5) / 8 - 0.5])
    crop = cam.cropped(10, 4, 20, 16)
    assert crop.image_size == (20, 16)
    np.testing.assert_allclose(crop.project(points[:1])[0][0], cam.project(points[:1])[0][0] - [10, 4])


def test_transform_helpers() -> None:
    T = np.array(transform((1, 2, 3), yaw_deg=30))
    p = np.random.default_rng(0).normal(size=(5, 3)).astype(np.float32)
    q = transform_points(T, p)
    assert q.dtype == np.float32
    np.testing.assert_allclose(transform_points(invert_transform(T), q), p, atol=1e-5)
    np.testing.assert_allclose(invert_transform(T), np.linalg.inv(T), atol=1e-12)
    assert isinstance(_rig().cameras["camera_ring_front"], Camera)


def test_transform_names_follow_T_to_from() -> None:
    """T_<to>_<from> everywhere, like T_world_base: T_base_sensor maps sensor points into the base frame."""
    lidar = _rig().lidars["lidar_top"]
    origin_in_base = transform_points(lidar.T_base_sensor, np.zeros((1, 3)))
    np.testing.assert_allclose(origin_in_base, [[0, 0, 1.2]])
    np.testing.assert_allclose(transform_points(lidar.T_sensor_base, origin_in_base), [[0, 0, 0]], atol=1e-12)
