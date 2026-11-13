"""PointCloud, the opt-in payload cache and lidar fusion."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from kitscenes._cache import PayloadCache
from kitscenes._fusion import fuse_point_clouds
from kitscenes.points import PointCloud
from kitscenes.schema import Provenance


def _cloud(xyz, sensor="lidar_top", frame="base", compensated=True, t_ns=1000) -> PointCloud:
    xyz = np.asarray(xyz, dtype=np.float32)
    return PointCloud(
        xyz=xyz, attrs={"reflectivity": np.arange(len(xyz), dtype=np.float32)},
        measured=np.any(xyz != 0, axis=1), frame=frame, t_ns=t_ns, sensor=sensor,
        motion_compensated=compensated, reason=None if compensated else "no poses.txt",
    )


def test_point_cloud_columns() -> None:
    c = _cloud([[1, 2, 3], [0, 0, 0], [4, 5, 6]])
    assert len(c) == 3 and c.measured.tolist() == [True, False, True]
    np.testing.assert_array_equal(c["y"], [2, 0, 5])
    np.testing.assert_array_equal(c["reflectivity"], [0, 1, 2])
    compact = c.compact()
    assert len(compact) == 2 and compact.attrs["reflectivity"].tolist() == [0, 2]
    s = c.to_structured()
    assert s.dtype.names == ("x", "y", "z", "reflectivity") and s["z"].tolist() == [3, 0, 6]
    with pytest.raises(ValueError, match="rows"):
        PointCloud(np.zeros((2, 3), np.float32), {"a": np.zeros(3)}, np.ones(2, bool), "base", 0, "x")
    with pytest.raises(ValueError, match="frame"):
        PointCloud(np.zeros((2, 3), np.float32), {}, np.ones(2, bool), "camera", 0, "x")


def test_fusion_drops_unmeasured_rows_and_tracks_sources() -> None:
    a = _cloud([[1, 0, 0], [0, 0, 0]], "lidar_top")
    b = _cloud([[0, 2, 0], [0, 0, 3]], "lidar_front", compensated=False)
    fused = fuse_point_clouds([a, b])
    np.testing.assert_allclose(fused.xyz, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert fused.sensor == ("lidar_top", "lidar_front")
    assert fused.source_index.tolist() == [0, 1, 1]
    assert fused.attrs["reflectivity"].tolist() == [0, 0, 1]
    assert not fused.motion_compensated and "lidar_front" in fused.reason
    assert fused.provenance is Provenance.GEOMETRY_DERIVED
    with pytest.raises(ValueError, match="sensor frame"):
        fuse_point_clouds([a, _cloud([[1, 1, 1]], frame="sensor")])
    with pytest.raises(ValueError, match="different times"):
        fuse_point_clouds([a, _cloud([[1, 1, 1]], t_ns=2000)])


def test_cache_returns_shared_read_only_payloads() -> None:
    cache = PayloadCache(max_bytes=1000)
    calls = []

    def load():
        calls.append(1)
        return np.zeros(100, dtype=np.uint8)

    first = cache.get_or_load("a", load)
    second = cache.get_or_load("a", load)
    assert first is second and len(calls) == 1
    with pytest.raises(ValueError, match="read-only"):
        first[0] = 1
    cloud = cache.get_or_load("c", lambda: _cloud([[1, 2, 3]]))
    assert not cloud.xyz.flags.writeable and not cloud.attrs["reflectivity"].flags.writeable


def test_cache_budget_and_pickling() -> None:
    cache = PayloadCache(max_bytes=250)
    for key in "abc":
        cache.get_or_load(key, lambda: np.zeros(100, np.uint8))
    assert len(cache) == 2 and cache.current_bytes == 200  # "a" evicted
    big = cache.get_or_load("big", lambda: np.zeros(1000, np.uint8))
    assert big.flags.writeable  # too big to keep, so it stays the caller's
    clone = pickle.loads(pickle.dumps(cache))
    assert clone.max_bytes == 250 and len(clone) == 0
