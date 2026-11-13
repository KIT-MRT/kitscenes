"""Exact timestamp and pose parsing in the release format."""

from __future__ import annotations

import numpy as np
import pytest

from kitscenes.errors import DataFormatError
from kitscenes._poses import parse_poses
from kitscenes.schema import EgoPose
from kitscenes._timestamps import parse_timestamps, seconds_to_ns


@pytest.mark.parametrize(
    "text, ns",
    [("0.0460", 46_000_000), ("2.0460", 2_046_000_000), ("29.9460", 29_946_000_000),
     ("0.046049118", 46_049_118), ("1", 1_000_000_000), ("1e-3", 1_000_000), ("0.0000000005", 0)],
)
def test_seconds_to_ns_is_exact(text: str, ns: int) -> None:
    assert seconds_to_ns(text) == ns


def test_float_parsing_would_truncate() -> None:
    assert int(float("2.0460") * 1e9) == 2_045_999_999  # why the decimal parse exists


@pytest.mark.parametrize("text", ["abc", "nan", "inf", "1700000000000000000"])
def test_invalid_timestamps_raise(text: str) -> None:
    with pytest.raises(DataFormatError):
        seconds_to_ns(text)


def test_parse_timestamps_names_the_line() -> None:
    assert parse_timestamps("0.0460\n\n0.1460\n").tolist() == [46_000_000, 146_000_000]
    with pytest.raises(DataFormatError, match="ts.txt:2"):
        parse_timestamps("0.0460\nx\n", "ts.txt")


def test_parse_poses(caplog) -> None:
    poses = parse_poses(
        "# comment\n"
        "0.046049118 -2338.5228 -425.0107 165.7079 0.0 0.0 0.0 1.0\n"
        "bad line\n"
        "2.046000000 1 2 3 0 0 0.7071068 0.7071068\n"
    )
    assert [p.t_ns for p in poses] == [46_049_118, 2_046_000_000]
    np.testing.assert_allclose(poses[0].xyz, [-2338.5228, -425.0107, 165.7079])
    assert poses[1].yaw == pytest.approx(np.pi / 2, abs=1e-6)
    np.testing.assert_allclose(poses[1].T_world_base[:3, 3], [1, 2, 3])
    assert "malformed" in caplog.text


def test_yaw_is_the_heading_under_pitch_and_roll() -> None:
    """EgoPose.yaw is where the base x axis points in the x-y plane, also when the vehicle pitches and rolls."""
    from scipy.spatial.transform import Rotation

    from kitscenes.map.roi import Region

    rotation = Rotation.from_euler("ZYX", [0.7, 0.12, -0.1])
    pose = EgoPose(0, np.zeros(3), rotation.as_quat())
    R = rotation.as_matrix()
    assert pose.yaw == pytest.approx(np.arctan2(R[1, 0], R[0, 0]), abs=1e-12)
    assert pose.yaw == pytest.approx(0.7, abs=1e-12)
    ahead = R[:2, 0] / np.linalg.norm(R[:2, 0]) * 9.9  # just inside a 10 m box, straight ahead
    box = Region(ahead=10.0, behind=1.0, left=0.01, right=0.01).around(pose)
    assert box.hits(ahead[None])


def test_ego_pose_uses_the_names_of_ego_state() -> None:
    """EgoPose has xyz, quat and T_world_base like EgoState (the naming rule of the README), not translation/rotation."""
    from scipy.spatial.transform import Rotation

    from kitscenes.ego_trajectory import EgoState

    quat = Rotation.from_euler("z", 0.3).as_quat()
    pose = EgoPose(5, xyz=[1.0, 2.0, 3.0], quat=quat)
    np.testing.assert_allclose(pose.T_world_base[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(pose.T_world_base[:3, :3], Rotation.from_quat(quat).as_matrix())
    shared = {"xyz", "quat", "T_world_base", "t_ns"}
    import dataclasses

    state_names = set(dir(EgoState)) | {f.name for f in dataclasses.fields(EgoState)}
    assert shared <= set(dir(pose)) and shared <= state_names
