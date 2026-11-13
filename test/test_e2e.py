"""kitscenes.e2e (driving commands) and the ego / command fields of the samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import kitscenes
from kitscenes.e2e import driving_command
from kitscenes.navigation import NUSCENES, Command, driving_commands
from kitscenes.samples import SampleDataset, SampleSpec, collate
from release_fixture import make_scene, write_poses


LEFT, STRAIGHT, RIGHT, UNKNOWN = Command.LEFT, Command.STRAIGHT, Command.RIGHT, Command.UNKNOWN


def test_driving_command_from_the_lateral_offset_at_the_horizon() -> None:
    ahead = np.column_stack([np.arange(1, 31) * 0.5, np.zeros(30)])  # 30 slots at 0.1 s
    for y_end, command in ((2.0, LEFT), (1.99, STRAIGHT), (-2.0, RIGHT), (0.0, STRAIGHT)):
        xyz = ahead.copy()
        xyz[29, 1] = y_end  # the 3 s slot
        assert driving_command(xyz) == command
    late = np.vstack([ahead, [[16.0, 9.0]]])  # past the horizon: not looked at
    assert driving_command(late) == STRAIGHT
    xyz = ahead.copy()
    xyz[9, 1], xyz[29, 1] = -3.0, 5.0
    valid = np.arange(30) < 10
    assert driving_command(xyz, valid) == RIGHT  # the last valid slot up to the horizon
    assert driving_command(xyz, np.zeros(30, bool)) is Command.UNKNOWN
    assert driving_command(xyz, valid).onehot("vad").tolist() == [1.0, 0.0, 0.0]  # UniAD / VAD: right first
    assert driving_command(xyz[[4, 9, 14, 19, 24, 29]], step_s=0.5) == LEFT  # 2 Hz


def test_ego_and_command_in_samples(tmp_path: Path) -> None:
    for name, yaw_rate in (("left", 0.5), ("right", -0.5), ("straight", 0.0)):
        write_poses(make_scene(tmp_path / "data" / "train" / name, n=40), 40, speed=10.0, yaw_rate=yaw_rate)
    make_scene(tmp_path / "data" / "train" / "no_poses", n=3, poses=False)
    ds = kitscenes.open(tmp_path)
    samples = SampleDataset(ds, SampleSpec(future_s=3.0, trajectory_step_s=0.5))
    by = {(s["meta"]["scene"], s["meta"]["index"]): s for s in samples}
    s = by[("left", 5)]
    # a quadratic over the last 0.5 s of the circle (0.25 rad of it): within 1 %
    assert s["ego"]["valid"] and s["ego"]["speed"] == pytest.approx(10.0, rel=0.01)
    assert s["ego"]["yaw_rate"] == pytest.approx(0.5, abs=1e-6)
    np.testing.assert_allclose(s["ego"]["velocity"], [10.0, 0.0, 0.0], atol=0.1)
    # v * yaw rate to the left; the quadratic misreads the arc's cubic term as 0.6 m/s² along x in this 20 m turn
    np.testing.assert_allclose(s["ego"]["acceleration"][:2], [0.0, 5.0], atol=0.7)
    assert (s["future"]["command"], s["future"]["command_valid"]) == (LEFT, True)
    assert by[("right", 5)]["future"]["command"] == RIGHT and by[("straight", 5)]["future"]["command"] == STRAIGHT
    end = by[("left", 30)]["future"]  # 0.9 s of poses left: a command from them, not valid
    assert end["valid"].tolist() == [True] + [False] * 5 and end["command_valid"] is False
    assert not by[("left", 1)]["ego"]["valid"] and by[("left", 2)]["ego"]["valid"]  # 0.2 s of past needed
    # causal: the same at the last frame of a scene cut there (no pose after it)
    write_poses(make_scene(tmp_path / "cut" / "data" / "train" / "left", n=6), 6, speed=10.0, yaw_rate=0.5)
    ds2_cut = kitscenes.open(tmp_path / "cut")["left"]
    cut = SampleDataset(kitscenes.open(tmp_path / "cut"), SampleSpec())[5]["ego"]
    np.testing.assert_allclose(cut["acceleration"], by[("left", 5)]["ego"]["acceleration"], atol=1e-9)
    none = by[("no_poses", 1)]
    assert not none["ego"]["valid"] and none["ego"]["speed"] == 0.0 and none["future"]["command"] == UNKNOWN
    batch = collate([s, none])
    assert np.asarray(batch["ego"]["valid"]).tolist() == [True, False]
    assert tuple(np.asarray(batch["ego"]["velocity"]).shape) == (2, 3)
    assert np.asarray(batch["future"]["command"]).tolist() == [LEFT, UNKNOWN]
    short = SampleDataset(ds, SampleSpec(future_s=1.0))[5]["future"]  # shorter than 3 s: at its last slot
    assert short["command_valid"] and short["command"] == LEFT
    # one numbering and one rule: the samples' command is that of the NUSCENES rule
    full = SampleDataset(ds, SampleSpec(future_s=3.0))
    rule = {name: driving_commands(ds[name], NUSCENES) for name in ("left", "right", "straight")}
    for sample in full:
        scene, index = sample["meta"]["scene"], sample["meta"]["index"]
        if scene in rule and sample["future"]["command_valid"]:
            assert sample["future"]["command"] == rule[scene][index]

    # frame.ego(causal=True) is the ego of the samples, as an EgoState with the frame's pose
    frame = ds["left"].frame(5)
    state = frame.ego(causal=True)
    np.testing.assert_allclose(state.velocity, s["ego"]["velocity"])
    np.testing.assert_allclose(state.acceleration, s["ego"]["acceleration"])
    assert state.yaw_rate == pytest.approx(s["ego"]["yaw_rate"]) and state.speed == pytest.approx(s["ego"]["speed"])
    np.testing.assert_allclose(state.T_world_base, frame.T_world_base)
    np.testing.assert_allclose(ds2_cut.frame(5).ego(causal=True).acceleration, state.acceleration, atol=1e-9)
    assert np.isnan(ds["left"].frame(1).ego(causal=True).velocity).all()  # the scene start: no motion yet
    assert np.isfinite(frame.ego().velocity).all()
