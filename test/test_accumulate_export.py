"""Multi-frame accumulation and PCD/PLY export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kitscenes import Scene
from kitscenes.export import write_pcd, write_ply
from release_fixture import STATIC, frame_ns, make_scene, write_static_lidar


@pytest.fixture
def scene(tmp_path: Path) -> Scene:
    path = make_scene(tmp_path / "s", n=5, cameras=(), radars=())
    write_static_lidar(path, 5)
    return Scene(path, split="train")


def test_world_accumulation_lines_up(scene: Scene) -> None:
    acc = scene.accumulate(range(5))
    assert acc.frame == "world" and acc.xyz.dtype == np.float64 and acc.sensor == ("lidar_top",)
    assert len(acc) == 15 and acc.attrs["frame_index"].tolist() == [0] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3
    np.testing.assert_allclose(acc.xyz, np.tile(STATIC, (5, 1)), atol=0.01)  # 5 mm storage grid
    thin = scene.accumulate(range(5), voxel_size=0.2)
    assert len(thin) == 3 and thin.attrs["frame_index"].tolist() == [0, 0, 0]
    near = scene.accumulate(range(5), max_range=25.0)  # only the point 20 m ahead is within 25 m in every frame
    np.testing.assert_allclose(near.xyz, np.tile(STATIC[:1], (5, 1)), atol=0.01)


def test_base_accumulation(scene: Scene) -> None:
    acc = scene.accumulate([0, 2, 4], frame="base", reference=2)
    assert acc.frame == "base" and acc.t_ns == frame_ns(2) and acc.xyz.dtype == np.float32
    expected = STATIC - [102.0, 200.0, 150.0]
    np.testing.assert_allclose(acc.xyz, np.tile(expected, (3, 1)), atol=0.01)
    with pytest.raises(ValueError):
        scene.accumulate([0], frame="sensor")


def test_without_poses(scene: Scene) -> None:
    (scene.files.path / "poses.txt").unlink()
    with pytest.raises(ValueError, match="no usable poses"):
        Scene(scene.files.path).accumulate([0, 1])


def _read_pcd(path: Path) -> tuple[dict, np.ndarray]:
    data = path.read_bytes()
    head, _, body = data.partition(b"DATA binary\n")
    header = dict(line.split(" ", 1) for line in head.decode().splitlines() if line and not line.startswith("#"))
    names, sizes, types = header["FIELDS"].split(), header["SIZE"].split(), header["TYPE"].split()
    kinds = {"F": "f", "I": "i", "U": "u"}
    dtype = np.dtype([(n, f"<{kinds[t]}{s}") for n, s, t in zip(names, sizes, types)])
    return header, np.frombuffer(body, dtype=dtype)


def test_pcd_and_ply(scene: Scene, tmp_path: Path) -> None:
    cloud = scene.frame(1).lidar("lidar_top", rows=True)
    header, table = _read_pcd(write_pcd(tmp_path / "f.pcd", cloud))
    assert header["POINTS"] == "3" and table.dtype.names == ("x", "y", "z", "reflectivity", "timestamp", "row")
    np.testing.assert_allclose(table["x"], cloud.xyz[cloud.measured, 0])
    assert table.dtype["x"] == np.float32
    _, everything = _read_pcd(write_pcd(tmp_path / "all.pcd", cloud, fields=["reflectivity"], measured_only=False))
    assert len(everything) == 4 and everything.dtype.names == ("x", "y", "z", "reflectivity")

    acc = scene.accumulate(range(3))
    _, world = _read_pcd(write_pcd(tmp_path / "w.pcd", acc))
    assert world.dtype["x"] == np.float64 and "frame_index" in world.dtype.names and "source" in world.dtype.names
    np.testing.assert_allclose(world["y"], acc.xyz[:, 1])

    ply = write_ply(tmp_path / "w.ply", acc).read_bytes()
    head, _, body = ply.partition(b"end_header\n")
    assert b"element vertex 9" in head and b"property double x" in head and b"property int frame_index" in head
    props = [line.split()[1:] for line in head.decode().splitlines() if line.startswith("property")]
    types = {"double": "<f8", "float": "<f4", "int": "<i4", "uchar": "u1"}
    rows = np.frombuffer(body, dtype=[(name, types[t]) for t, name in props])
    np.testing.assert_allclose(rows["z"], acc.xyz[:, 2])


def _devices() -> list[str]:
    torch = pytest.importorskip("torch")
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


# The device path looks poses up on a fixed 0.02 ms grid, the CPU path on bins starting at each
# sweep's first point: up to one bin apart, 0.2 mm at the fixture's 10 m/s.
DEVICE_ATOL = 5e-4


@pytest.mark.parametrize("kwargs", [{}, {"voxel_size": 0.33}, {"max_range": 25.0}, {"frame": "base", "reference": 2}])
def test_device_path_matches_numpy(scene: Scene, kwargs: dict) -> None:
    expected = scene.accumulate(range(5), **kwargs)
    assert len(scene.accumulate(range(5), workers=1, **kwargs)) == len(expected)
    for device in _devices():
        got = scene.accumulate(range(5), device=device, **kwargs)
        assert got.frame == expected.frame and got.xyz.dtype == expected.xyz.dtype
        np.testing.assert_allclose(got.xyz, expected.xyz, atol=DEVICE_ATOL)
        assert got.motion_compensated
        assert got.attrs.keys() == expected.attrs.keys()
        for key in expected.attrs:
            np.testing.assert_array_equal(got.attrs[key], expected.attrs[key])
        np.testing.assert_array_equal(got.source_index, expected.source_index)


def test_device_batches_and_voxels_across_batches(scene: Scene, monkeypatch: pytest.MonkeyPatch) -> None:
    import kitscenes.accumulate as acc

    expected = scene.accumulate(range(5), voxel_size=0.33)
    monkeypatch.setattr(acc, "BATCH_POINTS", 1)  # one frame per device batch
    for device in _devices():
        got = scene.accumulate(range(5), voxel_size=0.33, device=device)
        np.testing.assert_allclose(got.xyz, expected.xyz, atol=DEVICE_ATOL)
        assert got.attrs["frame_index"].tolist() == [0, 0, 0]


def test_device_path_without_point_times(scene: Scene) -> None:
    import pyarrow.parquet as pq

    path = scene.files.path / "lidar_top" / "0000000001.parquet"
    table = pq.read_table(path)
    pq.write_table(table.drop_columns(["timestamp"]), path)
    scene = Scene(scene.files.path, split="train")
    expected = scene.accumulate(range(3))
    for device in _devices():
        got = scene.accumulate(range(3), device=device)
        np.testing.assert_allclose(got.xyz, expected.xyz, atol=DEVICE_ATOL)
        assert not got.motion_compensated and "frame 1 lidar_top" in got.reason


def test_accumulate_does_not_load_the_voxel_map_module(scene: Scene) -> None:
    """Scene.accumulate (CPU and device path) shares kitscenes._sweeps with mapping, not mapping itself."""
    import importlib.util
    import subprocess
    import sys

    device = "'cpu'" if importlib.util.find_spec("torch") else "None"
    code = (
        "import sys, kitscenes\n"
        f"scene = kitscenes.Scene({str(scene.files.path)!r}, split='train')\n"
        "scene.accumulate(range(3))\n"
        f"scene.accumulate(range(3), device={device})\n"
        "print('kitscenes.mapping' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"
