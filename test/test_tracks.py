"""Scene trajectories without sensor data (Scene.track, Dataset.read_tracks)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import kitscenes
from kitscenes.map import MapOrigin
from release_fixture import make_dataset


def test_tracks(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "root", {"train": ["a", "b"], "test-e2e": ["c"]}, maps=True)
    ds = kitscenes.open(root)
    tracks, failures = ds.read_tracks(workers=4)
    assert failures == {} and list(tracks) == ds.scene_ids
    a = tracks["a"]
    assert a.split == "train" and a.city is None and len(a) == 3  # not in the scene table
    np.testing.assert_allclose(a.xyz[0], [100.0, 200.0, 150.0])
    origin = MapOrigin(49.0144, 8.4172)
    np.testing.assert_allclose(a.utm()[:, :2], a.xyz[:, :2] + origin.utm_offset)
    np.testing.assert_allclose(a.latlon()[0], origin.to_latlon(a.xyz[:1])[0])
    c = tracks["c"]
    assert c.origin is None  # test-e2e withholds maps
    with pytest.raises(ValueError, match="no map origin"):
        c.utm()


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads any file")
def test_unreadable_scene_is_a_failure(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "root", {"train": ["a", "b"]}, maps=True)
    poses = root / "data" / "train" / "b" / "poses.txt"
    poses.chmod(0)
    try:
        tracks, failures = kitscenes.open(root).read_tracks()
    finally:
        poses.chmod(0o644)
    assert list(tracks) == ["a"] and list(failures) == ["b"] and "PermissionError" in failures["b"]


def test_malformed_scene_is_a_failure(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "root", {"train": ["a", "b"]}, maps=True)
    (root / "data" / "train" / "b" / "timestamp.reference.txt").write_text("1755812794500000000\n")  # epoch ns
    tracks, failures = kitscenes.open(root).read_tracks()
    assert list(tracks) == ["a"] and failures["b"].startswith("DataFormatError")
