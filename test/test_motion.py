"""Tests for kitscenes.motion (trajectory interpolation and LiDAR motion compensation)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from kitscenes.motion import (
    ExtrapolationError,
    InterpolationGapError,
    Trajectory,
    compensate_points,
    point_transforms,
    measured_point_mask,
)
from kitscenes.schema import EgoPose

SPEED = 10.0  # m/s along x
YAW_RATE = 0.5  # rad/s


def _straight(times: np.ndarray) -> Trajectory:
    trans = np.column_stack([SPEED * times, np.zeros_like(times), np.zeros_like(times)])
    quats = np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1))
    return Trajectory(times, trans, quats)


def _circle_pose(t: np.ndarray) -> tuple[Rotation, np.ndarray]:
    """Constant body velocity (SPEED forward, YAW_RATE): an exact circle."""
    yaw = YAW_RATE * t
    radius = SPEED / YAW_RATE
    trans = np.column_stack([radius * np.sin(yaw), radius * (1 - np.cos(yaw)), np.zeros_like(t)])
    return Rotation.from_euler("z", yaw[:, None]), trans


def _circle(times: np.ndarray) -> Trajectory:
    rot, trans = _circle_pose(times)
    return Trajectory(times, trans, rot.as_quat())


def _observe(world_points: np.ndarray, t: np.ndarray, pose_fn) -> np.ndarray:
    """Base-frame coordinates of static world points seen at times t."""
    rot, trans = pose_fn(t)
    return rot.inv().apply(world_points - trans)


def _straight_pose(t: np.ndarray) -> tuple[Rotation, np.ndarray]:
    return Rotation.identity(len(t)), np.column_stack([SPEED * t, np.zeros_like(t), np.zeros_like(t)])


@pytest.mark.parametrize("time_bin_s", [None, 1e-4])
def test_straight_motion_inside_and_beyond_poses(time_bin_s) -> None:
    times = np.arange(0.0, 1.0001, 0.1)
    traj = _straight(times)
    rng = np.random.default_rng(0)
    world = rng.uniform(-30, 30, (500, 3))
    # Scan 0...+100 ms after the last pose, as the Hesai lidar_top does.
    t_points = np.sort(rng.uniform(1.0, 1.1, 500))
    xyz = _observe(world, t_points, _straight_pose)
    out = compensate_points(xyz, t_points, None, traj, 1.0, time_bin_s=time_bin_s, compute_dtype=np.float64)
    expected = _observe(world, np.full(500, 1.0), _straight_pose)
    tol = 1e-9 if time_bin_s is None else SPEED * time_bin_s
    np.testing.assert_allclose(out, expected, atol=tol)


def test_constant_turn_extrapolation_is_exact() -> None:
    times = np.arange(0.0, 2.0001, 0.1)
    traj = _circle(times)
    rng = np.random.default_rng(1)
    world = rng.uniform(-40, 40, (300, 3))
    t_points = rng.uniform(-0.05, 0.05, 300)  # straddles the first pose
    xyz = _observe(world, t_points, _circle_pose)
    out = compensate_points(xyz, t_points, None, traj, 0.0, time_bin_s=None, compute_dtype=np.float64)
    expected = _observe(world, np.zeros(300), _circle_pose)
    before = t_points < 0
    # Extrapolation continues the body twist, which is exact for a constant turn ...
    np.testing.assert_allclose(out[before], expected[before], atol=1e-6)
    # ... while interpolation is linear (chord, not arc): a few mm at 40 m here.
    np.testing.assert_allclose(out[~before], expected[~before], atol=1e-2)


def test_float32_default_is_close_to_float64() -> None:
    traj = _circle(np.arange(0.0, 2.0001, 0.1))
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-300, 300, (2000, 3))
    t = rng.uniform(0.9, 1.0, 2000)
    fast = compensate_points(xyz, t, None, traj, 0.95)
    precise = compensate_points(xyz, t, None, traj, 0.95, compute_dtype=np.float64)
    np.testing.assert_allclose(fast, precise, atol=1e-4)  # 0.1 mm at 300 m


def test_extrinsic_and_target_frames() -> None:
    times = np.arange(0.0, 1.0001, 0.1)
    traj = _straight(times)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    T[:3, 3] = [1.0, 2.0, 1.5]
    xyz = np.array([[5.0, 0.0, 0.0], [0.0, 3.0, 1.0]])
    t = np.array([0.5, 0.5])
    base = compensate_points(xyz, t, T, traj, 0.5, target_frame="base", time_bin_s=None)
    np.testing.assert_allclose(base, xyz @ T[:3, :3].T + T[:3, 3])  # same time: frame change only
    sensor = compensate_points(xyz, t, T, traj, 0.5, target_frame="sensor", time_bin_s=None)
    np.testing.assert_allclose(sensor, xyz, atol=1e-12)
    world = compensate_points(xyz, t, T, traj, 0.5, target_frame="world", time_bin_s=None)
    np.testing.assert_allclose(world, base + [SPEED * 0.5, 0, 0])


def test_origin_points_stay_and_time_zero_is_valid() -> None:
    times = np.arange(0.0, 1.0001, 0.1)
    traj = _straight(times)
    xyz = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    t = np.array([0.3, 0.0])  # the first point of a sequence is measured at t = 0
    out = compensate_points(xyz, t, None, traj, 0.5, time_bin_s=None)
    np.testing.assert_array_equal(out[0], 0.0)
    np.testing.assert_allclose(out[1], [10.0 - SPEED * 0.5, 0.0, 0.0])
    np.testing.assert_array_equal(measured_point_mask(xyz), [False, True])


def test_extrapolation_limit_and_gap() -> None:
    traj = _straight(np.arange(0.0, 1.0001, 0.1))
    with pytest.raises(ExtrapolationError):
        traj.interpolate(np.array([1.25]))
    gappy = _straight(np.array([0.0, 0.1, 0.5, 0.6]))
    with pytest.raises(InterpolationGapError):
        gappy.interpolate(np.array([0.3]))
    gappy.interpolate(np.array([0.1, 0.5]))  # exactly at poses around a gap is fine


def test_binned_transforms_cover_all_points() -> None:
    traj = _straight(np.arange(0.0, 1.0001, 0.1))
    t = np.linspace(0.2, 0.3, 10_001)
    rot, trans, index = point_transforms(t, traj, 0.25, time_bin_s=1e-4)
    assert rot.shape[0] == trans.shape[0] == index.max() + 1 <= 1002
    assert index.min() == 0 and len(index) == len(t)


from kitscenes.motion import compensate_radar_points  # noqa: E402

_RADAR_DTYPE = [
    ("x", "f4"), ("y", "f4"), ("z", "f4"), ("azimuth", "f4"), ("elevation", "f4"), ("range", "f4"),
    ("range_rate", "f4"), ("range_rate_std", "f4"), ("timestamp", "f8"),
]


def _radar_cloud(world: np.ndarray, t_capture: float, pose_fn, T: np.ndarray, twist_fn) -> np.ndarray:
    """Detections of static world points by a radar mounted at T, measured at t_capture."""
    rot, trans = pose_fn(np.array([t_capture]))
    p_base = rot.inv().apply(world - trans[0])                 # (N, 3) base frame at capture
    xyz = (p_base - T[:3, 3]) @ T[:3, :3]                       # sensor frame
    v, w = twist_fn(t_capture)                                 # body velocity
    u = (xyz / np.linalg.norm(xyz, axis=1)[:, None]) @ T[:3, :3].T
    pts = np.zeros(len(world), dtype=_RADAR_DTYPE)
    pts["x"], pts["y"], pts["z"] = xyz.T
    r = np.linalg.norm(xyz, axis=1)
    pts["range"], pts["azimuth"], pts["elevation"] = r, np.arctan2(xyz[:, 1], xyz[:, 0]), np.arcsin(xyz[:, 2] / r)
    pts["range_rate"] = -np.einsum("ij,ij->i", u, v + np.cross(w, p_base))  # static target
    pts["range_rate_std"] = 0.1
    pts["timestamp"] = t_capture
    return pts


def _mount() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    T[:3, 3] = [3.5, 0.8, 0.5]
    return T


def test_radar_positions_moved_to_reference_time_and_doppler_removed() -> None:
    times = np.arange(0.0, 2.0001, 0.1)
    traj = _straight(times)
    T = _mount()
    rng = np.random.default_rng(3)
    world = np.column_stack([rng.uniform(20, 80, 50), rng.uniform(-20, 20, 50), rng.uniform(0, 2, 50)])
    t_ref, t_cap = 1.0, 1.0 - 0.019
    raw = _radar_cloud(world, t_cap, _straight_pose, T, lambda t: (np.array([SPEED, 0, 0]), np.zeros(3)))
    out, pos_ok, dop_ok, reason = compensate_radar_points(raw, T, traj, t_ref)
    assert pos_ok and dop_ok and reason is None
    expected = _radar_cloud(world, t_ref, _straight_pose, T, lambda t: (np.array([SPEED, 0, 0]), np.zeros(3)))
    for f in ("x", "y", "z", "range", "azimuth", "elevation"):
        np.testing.assert_allclose(out[f], expected[f], atol=1e-4)
    np.testing.assert_allclose(out["range_rate"], 0.0, atol=1e-3)
    np.testing.assert_array_equal(out["range_rate_std"], raw["range_rate_std"])
    np.testing.assert_array_equal(raw["x"], _radar_cloud(world, t_cap, _straight_pose, T,
                                  lambda t: (np.array([SPEED, 0, 0]), np.zeros(3)))["x"])  # input untouched


def test_radar_doppler_while_turning_with_lever_arm() -> None:
    times = np.arange(0.0, 3.0001, 0.1)
    traj = _circle(times)
    T = _mount()
    rng = np.random.default_rng(4)
    world = np.column_stack([rng.uniform(-60, 60, 80), rng.uniform(-60, 60, 80), rng.uniform(0, 2, 80)])
    raw = _radar_cloud(world, 1.481, _circle_pose, T, lambda t: (np.array([SPEED, 0, 0]), np.array([0, 0, YAW_RATE])))
    out, pos_ok, dop_ok, _ = compensate_radar_points(raw, T, traj, 1.5)
    assert pos_ok and dop_ok
    np.testing.assert_allclose(out["range_rate"], 0.0, atol=0.02)  # central differences on a circle: ~mm/s


def test_radar_without_usable_capture_time_keeps_positions() -> None:
    traj = _straight(np.arange(0.0, 2.0001, 0.1))
    T = _mount()
    world = np.array([[40.0, 5.0, 1.0]])
    raw = _radar_cloud(world, 1.0, _straight_pose, T, lambda t: (np.array([SPEED, 0, 0]), np.zeros(3)))
    raw["timestamp"] = 1e9  # other clock
    out, pos_ok, dop_ok, reason = compensate_radar_points(raw, T, traj, 1.0)
    assert not pos_ok and dop_ok and reason.startswith("capture time unusable")
    np.testing.assert_array_equal(out["x"], raw["x"])
    np.testing.assert_allclose(out["range_rate"], 0.0, atol=1e-3)  # Doppler with the velocity at t_target


def test_radar_base_frame_and_no_trajectory() -> None:
    traj = _straight(np.arange(0.0, 2.0001, 0.1))
    T = _mount()
    raw = _radar_cloud(np.array([[40.0, 5.0, 1.0]]), 1.0, _straight_pose, T, lambda t: (np.array([SPEED, 0, 0]), np.zeros(3)))
    out, *_ = compensate_radar_points(raw, T, traj, 1.0, target_frame="base")
    xyz = np.column_stack([raw["x"], raw["y"], raw["z"]]).astype(np.float64)
    np.testing.assert_allclose(np.column_stack([out["x"], out["y"], out["z"]]), xyz @ T[:3, :3].T + T[:3, 3], atol=1e-4)
    np.testing.assert_allclose(out["range"], raw["range"], atol=1e-4)  # stays relative to the radar
    out, pos_ok, dop_ok, reason = compensate_radar_points(raw, T, None, 1.0)
    assert not pos_ok and not dop_ok and reason == "missing or insufficient poses.txt data"
    np.testing.assert_array_equal(out, raw)
