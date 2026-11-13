"""Tests for kitscenes.ego_motion: the gyro-refined ego motion, its fallbacks and how the API uses it."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import kitscenes
from kitscenes import Scene
from kitscenes.ego_motion import ImuTrajectory, refine_trajectory
from kitscenes.messages import IMU_TO_BASE
from kitscenes.motion import Trajectory
from kitscenes.rig import Rig

from release_fixture import frame_ns, imu_message, make_dataset, make_scene, seconds, write_calib, write_lidar, write_messages

MOUNT = Rotation.from_euler("xyz", [0.2, -0.3, 0.9], degrees=True) * Rotation.from_matrix(IMU_TO_BASE)
BIAS = np.array([0.004, -0.002, 0.001])  # rad/s, IMU axes
RATE_HZ = 50.0


def true_attitude(t) -> Rotation:
    """A yaw sweep plus roll and pitch rocking at 2 and 1.3 Hz, what poses at 10 Hz get wrong."""
    t = np.atleast_1d(t)
    return Rotation.from_euler("ZYX", np.c_[0.4 * np.sin(0.5 * t), 0.015 * np.sin(2 * np.pi * 1.3 * t + 0.5),
                                            0.02 * np.sin(2 * np.pi * 2.0 * t)])


def imu_samples(t0: float, t1: float, mount: Rotation = MOUNT) -> tuple[np.ndarray, np.ndarray]:
    times = np.arange(t0, t1, 1.0 / RATE_HZ)
    eps = 1e-5
    body = (true_attitude(times - eps).inv() * true_attitude(times + eps)).as_rotvec() / (2 * eps)
    return times, mount.inv().apply(body) + BIAS


def noisy_poses(times: np.ndarray, sigma_rad: float = 3e-3) -> Trajectory:
    noise = Rotation.from_rotvec(np.random.default_rng(1).normal(0.0, sigma_rad, (len(times), 3)))
    return Trajectory(times, np.c_[10.0 * times, np.zeros((len(times), 2))], (true_attitude(times) * noise).as_quat())


def imu_entry(rotation: Rotation = MOUNT) -> dict:
    T = np.eye(4)
    T[:3, :3] = rotation.as_matrix()
    return {"T_to_reference": T.tolist()}


def scene_with(tmp_path: Path, *, imu: bool = True, entry: dict | None = None, frames: int = 80, **kwargs) -> Path:
    """A scene with a moving vehicle; *entry*: its ``gnss_ins_imu`` calibration, default the valid :data:`MOUNT`."""
    path = make_scene(tmp_path / "train" / "scene", n=frames, **kwargs)
    write_lidar(path, "lidar_top", range(frames), sweep_s=0.09)
    calib = json.loads((path / "calibration" / "calib.json").read_text())
    calib.pop("reference")
    calib["gnss_ins_imu"] = imu_entry() if entry is None else entry
    write_calib(path, calib)
    write_poses_like(path, frames)
    if imu:
        times, rates = imu_samples(frame_ns(0) / 1e9 - 0.05, frame_ns(frames) / 1e9 + 0.1)
        messages = []
        for t, w in zip(times, rates):
            message = imu_message(t, w[2])
            message["angular_velocity"].update(x=float(w[0]), y=float(w[1]))
            messages.append(message)
        write_messages(path, "gnss_ins", "imu", messages)
    return path


def write_poses_like(path: Path, frames: int) -> None:
    times = np.array([frame_ns(i) for i in range(frames)]) / 1e9
    poses = noisy_poses(times)
    (path / "poses.txt").write_text("".join(
        f"{seconds(frame_ns(i), 9)} {x:.4f} {y:.4f} {z:.4f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n"
        for i, ((x, y, z), q) in enumerate(zip(poses.translations, poses.rotations.as_quat()))))


# ----------------------------------------------------------------------
# The refinement
# ----------------------------------------------------------------------


def test_refined_attitude_follows_the_truth_better_than_the_poses() -> None:
    times = np.arange(0.0, 12.0, 0.1)
    poses = noisy_poses(times)
    refined = refine_trajectory(poses, *imu_samples(-0.1, 12.2), MOUNT.as_matrix())

    t = np.arange(1.0, 11.0, 0.013)  # also between the poses
    truth = true_attitude(t)
    error_poses = (truth.inv() * poses.interpolate(t)[0]).magnitude()
    error_refined = (truth.inv() * refined.interpolate(t)[0]).magnitude()
    assert np.sqrt(np.mean(error_refined ** 2)) < 0.5 * np.sqrt(np.mean(error_poses ** 2))  # slower errors stay, by design
    np.testing.assert_allclose(refined.qa["gyro_bias_rad_s"], BIAS, atol=1e-3)
    assert refined.qa["imu_to_base_deviation_from_nominal_deg"] == pytest.approx(MOUNT_DEVIATION_DEG, abs=1e-6)
    assert max(refined.qa["residual_after_mrad"]) < max(refined.qa["residual_before_mrad"])


MOUNT_DEVIATION_DEG = float(np.degrees((MOUNT * Rotation.from_matrix(IMU_TO_BASE).inv()).magnitude()))


def test_translations_and_limits_stay_those_of_the_poses() -> None:
    times = np.arange(0.0, 6.0, 0.1)
    poses = noisy_poses(times)
    refined = refine_trajectory(poses, *imu_samples(-0.1, 6.2), MOUNT.as_matrix())
    t = np.array([0.04, 1.234, 5.85])
    np.testing.assert_allclose(refined.interpolate(t)[1], poses.interpolate(t)[1])
    np.testing.assert_allclose(refined.interpolate_matrices(t)[0], refined.interpolate(t)[0].as_matrix(), atol=1e-9)
    assert refined.covers(np.array([5.9]))[0] and not refined.covers(np.array([7.0]))[0]
    with pytest.raises(ValueError):
        refined.interpolate(np.array([7.0]))


def test_attitude_past_the_imu_data_continues_with_the_last_body_rate() -> None:
    poses = noisy_poses(np.arange(0.0, 3.0, 0.1))
    times, rates = imu_samples(-0.1, 2.905)
    refined = refine_trajectory(poses, times, rates, MOUNT.as_matrix())
    last = refined.interpolate(times[-1:])[0]
    bias = np.array(refined.qa["gyro_bias_rad_s"])
    expected = last * Rotation.from_rotvec(MOUNT.apply(rates[-1] - bias) * 0.05)
    assert (refined.interpolate(times[-1:] + 0.05)[0].inv() * expected).magnitude()[0] < 1e-12
    np.testing.assert_allclose(refined.interpolate_matrices(times[-1:] + 0.05)[0], expected.as_matrix(), atol=1e-12)


def test_trajectory_pickles() -> None:
    times = np.arange(0.0, 3.0, 0.1)
    refined = refine_trajectory(noisy_poses(times), *imu_samples(-0.1, 3.2), MOUNT.as_matrix())
    t = np.array([0.55, 1.71])
    again = pickle.loads(pickle.dumps(refined))
    np.testing.assert_array_equal(again.interpolate_matrices(t)[0], refined.interpolate_matrices(t)[0])


@pytest.mark.parametrize("cut, message", [
    (lambda t, r: (t[t < 2.0], r[t < 2.0]), "does not cover"),
    (lambda t, r: (t[(t < 1.0) | (t > 1.5)], r[(t < 1.0) | (t > 1.5)]), "gap"),
    (lambda t, r: (t[::-1], r), "strictly increasing"),
    (lambda t, r: (t[:1], r[:1]), "fewer than two"),
])
def test_unusable_imu_data_says_why(cut, message) -> None:
    poses = noisy_poses(np.arange(0.0, 3.0, 0.1))
    with pytest.raises(ValueError, match=message):
        refine_trajectory(poses, *cut(*imu_samples(-0.1, 3.2)), MOUNT.as_matrix())


# ----------------------------------------------------------------------
# The calibration
# ----------------------------------------------------------------------


def test_rig_accepts_only_a_plausible_imu_rotation() -> None:
    np.testing.assert_allclose(Rig.from_calib({"gnss_ins_imu": imu_entry()}).imu_to_base, MOUNT.as_matrix())
    assert Rig.from_calib({"gnss_ins_imu": imu_entry(Rotation.identity())}).imu_to_base is None
    tilted = Rotation.from_euler("z", 6.0, degrees=True) * Rotation.from_matrix(IMU_TO_BASE)
    assert Rig.from_calib({"gnss_ins_imu": imu_entry(tilted)}).imu_to_base is None
    assert Rig.from_calib({}).imu_to_base is None


def test_scene_imu_uses_the_calibrated_rotation(tmp_path: Path) -> None:
    scene = Scene(scene_with(tmp_path))
    raw, base = scene.imu("imu"), scene.imu("base")
    np.testing.assert_allclose(base["angular_velocity"], raw["angular_velocity"] @ MOUNT.as_matrix().T)


def test_scene_imu_uses_the_nominal_axes_without_a_valid_rotation(tmp_path: Path) -> None:
    scene = Scene(scene_with(tmp_path, entry=imu_entry(Rotation.identity())))
    raw, base = scene.imu("imu"), scene.imu("base")
    np.testing.assert_allclose(base["angular_velocity"], raw["angular_velocity"] @ IMU_TO_BASE.T)


def test_calib_file_replaces_the_scene_calibration(tmp_path: Path) -> None:
    path = scene_with(tmp_path, entry=imu_entry(Rotation.identity()))
    assert Scene(path).ego_motion.source == "poses"
    calib = json.loads((path / "calibration" / "calib.json").read_text())
    calib["gnss_ins_imu"] = imu_entry()
    calib_file = tmp_path / "rc.json"
    calib_file.write_text(json.dumps(calib))
    scene = Scene(path, calib_file=calib_file)
    assert scene.ego_motion.source == "imu"
    assert pickle.loads(pickle.dumps(scene)).ego_motion.source == "imu"


# ----------------------------------------------------------------------
# Sources and fallbacks
# ----------------------------------------------------------------------


def test_imu_is_the_default(tmp_path: Path) -> None:
    ego = Scene(scene_with(tmp_path)).ego_motion
    assert ego.source == "imu" and ego.reason is None and isinstance(ego.trajectory, ImuTrajectory) and ego.qa
    assert "imu" in repr(ego)


@pytest.mark.parametrize("kwargs, opened, reason", [
    ({"imu": False}, {}, "no IMU data"),
    ({"entry": imu_entry(Rotation.identity())}, {}, "gnss_ins_imu in the calibration is 1"),
    ({}, {"use_imu": False}, "use_imu=False"),
])
def test_fallback_is_exactly_the_pose_based_deskewing(tmp_path: Path, kwargs, opened, reason) -> None:
    path = scene_with(tmp_path, **kwargs)
    scene = Scene(path, **opened)
    assert scene.ego_motion.source == "poses" and reason in scene.ego_motion.reason
    assert type(scene.ego_motion.trajectory) is Trajectory and not scene.ego_motion.qa
    cloud = scene.frame(5).lidar("lidar_top")
    assert cloud.motion_compensated and cloud.reason is None

    poses = scene._pose_trajectory[0]
    t = scene.frame(5).t_s
    np.testing.assert_array_equal(scene.frame(5).pose.quat, poses.interpolate(np.array([t]))[0].as_quat()[0])


def test_missing_gnss_ins_imu_entry(tmp_path: Path) -> None:
    path = scene_with(tmp_path)
    calib = json.loads((path / "calibration" / "calib.json").read_text())
    del calib["gnss_ins_imu"], calib["reference"]
    write_calib(path, calib)
    ego = Scene(path).ego_motion
    assert ego.source == "poses" and ego.reason == "no gnss_ins_imu in the calibration"


def test_unusable_imu_data_falls_back(tmp_path: Path) -> None:
    path = scene_with(tmp_path, imu=False)
    write_messages(path, "gnss_ins", "imu", [imu_message(frame_ns(0) / 1e9 + t, 0.0) for t in np.arange(0.0, 2.0, 0.02)])
    ego = Scene(path).ego_motion
    assert ego.source == "poses" and "does not cover" in ego.reason


def test_without_poses_nothing_is_compensated(tmp_path: Path) -> None:
    path = scene_with(tmp_path)
    (path / "poses.txt").unlink()
    scene = Scene(path)
    assert scene.ego_motion.source is None and scene.ego_motion.trajectory is None
    cloud = scene.frame(5).lidar("lidar_top")
    assert not cloud.motion_compensated and "poses" in cloud.reason
    with pytest.raises(kitscenes.PoseUnavailable):
        scene.frame(5).pose


def test_use_imu_passes_through_the_dataset(tmp_path: Path) -> None:
    root = tmp_path / "root"
    make_dataset(root, {"val": ("scene_a", "scene_b")})
    ds = kitscenes.open(root, use_imu=False)
    assert ds[0].use_imu is False
    assert pickle.loads(pickle.dumps(ds))[0].use_imu is False
    assert ds.filter(splits="val")[0].use_imu is False
    assert kitscenes.open(root)[0].use_imu is True


def test_availability_does_not_build_the_ego_motion(tmp_path: Path) -> None:
    scene = Scene(scene_with(tmp_path))
    assert scene.frame(5).availability("pose") is kitscenes.Availability.PRESENT
    assert "ego_motion" not in scene.__dict__


# ----------------------------------------------------------------------
# What the API gives
# ----------------------------------------------------------------------


def test_frame_pose_is_refined_and_poses_stay_as_stored(tmp_path: Path) -> None:
    path = scene_with(tmp_path)
    scene, plain = Scene(path), Scene(path, use_imu=False)
    refined, stored = scene.frame(40).pose, plain.frame(40).pose
    np.testing.assert_array_equal(refined.xyz, stored.xyz)
    truth = true_attitude(scene.frame(40).t_s)
    assert (truth.inv() * Rotation.from_quat(refined.quat)).magnitude()[0] < (truth.inv() * Rotation.from_quat(stored.quat)).magnitude()[0]
    np.testing.assert_array_equal(scene.track.quat[40], plain.track.quat[40])
    np.testing.assert_array_equal(scene.track.xyz, plain.track.xyz)
    assert np.isfinite(scene.ego(scene.frame(40).t_ns).yaw_rate)


def test_deskewing_uses_the_refined_attitude(tmp_path: Path) -> None:
    path = scene_with(tmp_path)
    refined = Scene(path).frame(40).lidar("lidar_top")
    plain = Scene(path, use_imu=False).frame(40).lidar("lidar_top")
    assert refined.motion_compensated and np.abs(refined.xyz - plain.xyz).max() > 1e-5
    radar = Scene(path).frame(40).radar("radar_front")
    assert len(radar) and radar.motion_compensated


@pytest.mark.parametrize("target", ["base", "sensor", "world"])
@pytest.mark.parametrize("source", ["sensor", "base"])
def test_compensate_equals_reading_it_deskewed(tmp_path: Path, target: str, source: str) -> None:
    scene = Scene(scene_with(tmp_path))
    frame = scene.frame(40)
    raw = frame.lidar("lidar_top", frame=source, deskew=False)
    out = scene.ego_motion.compensate(raw, frame=target)
    expected = frame.lidar("lidar_top", frame=target)
    assert out.motion_compensated and out.frame == target
    np.testing.assert_allclose(out.xyz, expected.xyz, atol=2e-4)


def test_compensate_without_poses_only_transforms(tmp_path: Path) -> None:
    path = scene_with(tmp_path)
    (path / "poses.txt").unlink()
    scene = Scene(path)
    raw = scene.frame(5).lidar("lidar_top", frame="sensor", deskew=False)
    out = scene.ego_motion.compensate(raw)
    assert not out.motion_compensated and "poses" in out.reason
    np.testing.assert_allclose(out.xyz, scene.frame(5).lidar("lidar_top").xyz, atol=1e-5)
    with pytest.raises(kitscenes.PoseUnavailable):
        scene.ego_motion.compensate(raw, frame="world")


def test_compensate_refuses_what_it_cannot_do(tmp_path: Path) -> None:
    scene = Scene(scene_with(tmp_path))
    frame = scene.frame(40)
    ego = scene.ego_motion
    with pytest.raises(ValueError, match="already deskewed"):
        ego.compensate(frame.lidar("lidar_top"))
    with pytest.raises(ValueError, match="radar"):
        ego.compensate(frame.radar("radar_front"))
    with pytest.raises(ValueError, match="'sensor' or 'base'"):
        ego.compensate(frame.lidar("lidar_top", deskew=False, frame="world"))
    with pytest.raises(ValueError, match="frame must be"):
        ego.compensate(frame.lidar("lidar_top", deskew=False), frame="road")


def test_a_read_only_imu_rotation_is_accepted() -> None:
    """``rig.transforms`` are read-only; older scipy (1.13, the py3.9 images) refuses such input in from_matrix."""
    times = np.arange(0.0, 12.0, 0.1)
    mount = MOUNT.as_matrix()
    mount.setflags(write=False)
    refined = refine_trajectory(noisy_poses(times), *imu_samples(-0.1, 12.2), mount)
    assert refined.qa["imu_to_base_deviation_from_nominal_deg"] == pytest.approx(MOUNT_DEVIATION_DEG, abs=1e-6)
