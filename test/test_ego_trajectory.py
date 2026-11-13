"""Tests for kitscenes.ego_trajectory: a frame's future and past, and the ego state."""

from __future__ import annotations

import numpy as np
import pytest

from kitscenes.ego_trajectory import num_slots, relative_trajectory, slot_times
from kitscenes.motion import Trajectory
from kitscenes.schema import EgoPose, Provenance


_SECOND_NS = 1_000_000_000


def _yaw_quaternion(yaw_rad: float) -> np.ndarray:
    """Quaternion (qx, qy, qz, qw) for a rotation about the z axis."""
    return np.array(
        [0.0, 0.0, np.sin(yaw_rad / 2.0), np.cos(yaw_rad / 2.0)],
        dtype=np.float64,
    )


def _make_poses(
    positions: tuple[tuple[float, float, float], ...],
    yaws_rad: tuple[float, ...] | None = None,
    step_ns: int = _SECOND_NS,
) -> tuple[EgoPose, ...]:
    """Build an EgoPose sequence, one pose every *step_ns* from t=0."""
    if yaws_rad is None:
        yaws_rad = tuple(0.0 for _ in positions)
    return tuple(
        EgoPose(
            t_ns=index * step_ns,
            xyz=np.array(position, dtype=np.float64),
            quat=_yaw_quaternion(yaw),
        )
        for index, (position, yaw) in enumerate(zip(positions, yaws_rad))
    )


def _trajectory(poses) -> Trajectory:
    return Trajectory.from_ego_poses(poses, max_gap_s=1.5)  # the poses here are 1 s apart


def _future(poses, anchor_ns: int = 0, horizon_s: float = 3.0, step_s: float = 1.0):
    return relative_trajectory(_trajectory(poses), anchor_ns, slot_times(anchor_ns, horizon_s, step_s))


def test_slots_are_fixed_and_by_time() -> None:
    assert num_slots(3.0) == 30 and num_slots(0.0) == 0 and num_slots(0.25, 0.1) == 2
    np.testing.assert_array_equal(slot_times(10, 0.3, 0.1), [100_000_010, 200_000_010, 300_000_010])
    np.testing.assert_array_equal(slot_times(10, 0.3, 0.1, past=True), [-299_999_990, -199_999_990, -99_999_990])
    with pytest.raises(ValueError):
        num_slots(1.0, 0.0)
    with pytest.raises(ValueError):
        num_slots(-1.0)


def test_slots_past_the_poses_are_invalid_not_dropped() -> None:
    # Poses at 0..2 s: from the pose at 1 s, a 3 s horizon has one covered slot; there are still 3.
    poses = _make_poses(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    trajectory = _future(poses, anchor_ns=_SECOND_NS)
    assert len(trajectory) == 3 and trajectory.valid.tolist() == [True, False, False]
    np.testing.assert_allclose(trajectory.xyz_ego[0], [1.0, 0.0, 0.0])
    assert np.isnan(trajectory.xyz_ego[1:]).all() and np.isnan(trajectory.quat_ego[1:]).all()
    assert len(_future(poses, horizon_s=0.0)) == 0
    # The past of the same anchor: the slot at -1 s is before the first pose.
    past = relative_trajectory(_trajectory(poses), _SECOND_NS, slot_times(_SECOND_NS, 2.0, 1.0, past=True))
    assert past.valid.tolist() == [False, True]
    np.testing.assert_allclose(past.xyz_ego[1], [-1.0, 0.0, 0.0])


def test_slots_are_interpolated_between_poses() -> None:
    poses = _make_poses(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    trajectory = _future(poses, anchor_ns=_SECOND_NS // 2, horizon_s=1.0, step_s=0.5)
    np.testing.assert_allclose(trajectory.t_ns, [_SECOND_NS, 3 * _SECOND_NS // 2])
    np.testing.assert_allclose(trajectory.xyz_ego[:, 0], [0.5, 1.0])
    np.testing.assert_allclose(trajectory.xyz_world[:, 0], [1.0, 1.5])


def test_an_anchor_outside_the_poses_raises_like_frame_pose() -> None:
    from kitscenes.motion import ExtrapolationError

    poses = _make_poses(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    with pytest.raises(ExtrapolationError):
        _future(poses, anchor_ns=5 * _SECOND_NS)


def test_relative_trajectory_anchors_positions_at_the_query_pose() -> None:
    poses = _make_poses(((5.0, 5.0, 0.0), (5.0, 5.0, 0.0), (7.0, 5.0, 0.0)))

    trajectory = _future(poses, horizon_s=2.0)

    # A future pose identical to the anchor sits at the ego-frame origin,
    # while world coordinates stay absolute.
    np.testing.assert_allclose(trajectory.xyz_ego[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(trajectory.xyz_world[0], [5.0, 5.0, 0.0])
    np.testing.assert_allclose(trajectory.xyz_ego[1], [2.0, 0.0, 0.0], atol=1e-9)


def test_relative_trajectory_rotates_into_anchor_frame() -> None:
    # Anchor faces world +y (90 deg yaw). A future pose 10 m along world +y is
    # therefore 10 m straight ahead in the anchor's own frame: ego +x.
    poses = _make_poses(((0.0, 0.0, 0.0), (0.0, 10.0, 0.0)), yaws_rad=(np.pi / 2, np.pi / 2))

    trajectory = _future(poses, horizon_s=1.0)

    np.testing.assert_allclose(trajectory.xyz_ego[0], [10.0, 0.0, 0.0], atol=1e-9)
    # Same heading as the anchor means no relative rotation.
    relative_yaw = 2.0 * np.arctan2(trajectory.quat_ego[0][2], trajectory.quat_ego[0][3])
    assert np.isclose(relative_yaw, 0.0, atol=1e-9)


def test_relative_trajectory_reports_relative_heading_change() -> None:
    # Anchor faces world +x, the next pose has turned 90 deg to the left.
    poses = _make_poses(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), yaws_rad=(0.0, np.pi / 2))

    trajectory = _future(poses, horizon_s=1.0)

    relative_yaw = 2.0 * np.arctan2(trajectory.quat_ego[0][2], trajectory.quat_ego[0][3])
    assert np.isclose(relative_yaw, np.pi / 2, atol=1e-9)
    assert trajectory.provenance is Provenance.GEOMETRY_DERIVED


def test_a_trajectory_projects_like_any_base_frame_geometry() -> None:
    from kitscenes.projection import project_lines
    from release_fixture import real_camera  # 1.6 m up, looking along +x

    poses = _make_poses(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)))
    trajectory = _future(poses, horizon_s=2.0)
    camera = real_camera()
    uv, depth, visible = camera.project(trajectory.xyz_ego[trajectory.valid])
    assert visible.all()
    np.testing.assert_allclose(uv[0], [1000, 600 + 1000 * 1.6 / 5])  # 5 m ahead, 1.6 m below the camera
    (piece,), = project_lines(camera, [np.vstack([[0.0, 0.0, 0.0], trajectory.xyz_ego])])
    np.testing.assert_allclose(piece[1][0], 0.1)  # the path starts at the camera's near plane


def test_frame_future_and_pose_agree_on_a_frame_without_its_own_poses_line(tmp_path) -> None:
    """A frame between two poses (its own line missing) has a pose, so it has a future and a past too."""
    import kitscenes
    from release_fixture import make_scene

    scene_dir = make_scene(tmp_path / "data" / "train" / "s", n=8, cameras=(), radars=())
    lines = (scene_dir / "poses.txt").read_text().splitlines()
    lines[3] = lines[3].replace(lines[3].split()[2], "not-a-number")  # frame 3's pose line is broken
    (scene_dir / "poses.txt").write_text("\n".join(lines) + "\n")
    scene = kitscenes.open(tmp_path)[0]
    assert len(scene.track) == 7
    x = [float(line.split()[1]) for line in lines]  # 10 m/s: frame i at x[0] + i
    frame = scene.frame(3)
    assert np.isclose(frame.pose.xyz[0], x[0] + 3)  # interpolated
    future = frame.future(horizon_s=0.5)
    assert future.valid.tolist() == [True, True, True, True, False]  # frames 4..7, then the scene ends
    np.testing.assert_allclose(future.xyz_world[:4, 0], x[4:8], atol=1e-6)
    np.testing.assert_allclose(future.xyz_ego[:4, 0], [1.0, 2.0, 3.0, 4.0], atol=1e-6)
    np.testing.assert_array_equal(future.t_ns, frame.t_ns + np.arange(1, 6) * 100_000_000)
    past = frame.past(horizon_s=0.5)
    assert past.valid.tolist() == [False, False, True, True, True]  # frames 0..2
    np.testing.assert_allclose(past.xyz_ego[2:, 0], [-3.0, -2.0, -1.0], atol=1e-6)
    # Frames 2 and 4 see frame 3's slot filled by the same interpolation.
    np.testing.assert_allclose(scene.frame(2).future(horizon_s=0.1).xyz_world[0], frame.pose.xyz)


def test_a_gap_in_the_poses_masks_its_slots_and_the_frames_in_it(tmp_path) -> None:
    import kitscenes
    from release_fixture import make_scene

    scene_dir = make_scene(tmp_path / "data" / "train" / "s", n=10, cameras=(), radars=())
    lines = (scene_dir / "poses.txt").read_text().splitlines()
    (scene_dir / "poses.txt").write_text("\n".join(lines[:3] + lines[6:]) + "\n")  # frames 3, 4, 5: 0.4 s gap
    scene = kitscenes.open(tmp_path)[0]
    future = scene.frame(1).future(horizon_s=0.6)
    assert future.valid.tolist() == [True, False, False, False, True, True]
    for i in (3, 4, 5):
        assert scene.frame(i).availability("pose") is kitscenes.Availability.DROPPED
        assert scene.frame(i).future(missing="none") is None and scene.frame(i).past(missing="none") is None
        with pytest.raises(kitscenes.PoseUnavailable):
            scene.frame(i).past()
    with pytest.raises(kitscenes.PoseUnavailable):
        scene.frame(4).pose


def test_scene_ego_state(tmp_path) -> None:
    """scene.ego(t_ns): pose, velocity, acceleration, yaw rate and curvature at any time."""
    import kitscenes
    from release_fixture import make_scene

    scene_dir = make_scene(tmp_path / "data" / "train" / "s", n=12, cameras=(), radars=())
    # A circle of radius 20 m at 10 m/s, turning left: yaw rate 0.5 rad/s, curvature 0.05 1/m.
    t = np.array([float(l.split()[0]) for l in (scene_dir / "poses.txt").read_text().splitlines()])
    angle = 0.5 * t
    lines = [f"{ti:.4f} {20 * np.sin(a):.6f} {20 - 20 * np.cos(a):.6f} 0 0 0 {np.sin(a / 2):.9f} {np.cos(a / 2):.9f}"
             for ti, a in zip(t, angle)]
    (scene_dir / "poses.txt").write_text("\n".join(lines) + "\n")
    scene = kitscenes.open(tmp_path)[0]
    state = scene.ego(scene.timestamps_ns[5])
    assert state.T_world_base.shape == (4, 4) and state.xyz.shape == (3,) and state.speed.shape == ()
    np.testing.assert_allclose(state.T_world_base, scene.frame(5).pose.T_world_base, atol=1e-9)
    np.testing.assert_allclose(state.velocity, [10.0, 0.0, 0.0], atol=0.05)
    np.testing.assert_allclose(state.yaw_rate, 0.5, atol=1e-3)
    np.testing.assert_allclose(state.curvature, 0.05, atol=1e-3)
    np.testing.assert_allclose(state.acceleration, [0.0, 5.0, 0.0], atol=0.1)  # centripetal v^2 / r, to the left
    many = scene.ego(scene.timestamps_ns[2:6])
    assert many.T_world_base.shape == (4, 4, 4) and many.speed.shape == (4,) and many.t_ns.dtype == np.int64
    np.testing.assert_allclose(scene.frame(3).ego().velocity, many.velocity[1])
    with pytest.raises(kitscenes.PoseUnavailable) as err:
        scene.ego(scene.timestamps_ns[-1] + 1_000_000_000)
    assert err.value.state is kitscenes.Availability.DROPPED
