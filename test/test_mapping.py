"""Voxel maps of one or several scenes (kitscenes.mapping)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kitscenes import Scene
from kitscenes.map import MapOrigin
from kitscenes.mapping import ANCHOR_STEP, VoxelMap, keyframes
from release_fixture import STATIC, instances_file, make_scene, real_camera, write_static_lidar

torch = pytest.importorskip("torch")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _scene(path: Path, origin: tuple[float, float] = (49.0144, 8.4172), shift=(0.0, 0.0)) -> Scene:
    """Five frames of the static points; *shift* moves them in this scene's map frame."""
    make_scene(path, n=5, cameras=(), radars=(), maps=True)
    (path / "maps" / "origin.json").write_text(json.dumps({"latitude": origin[0], "longitude": origin[1]}))
    write_static_lidar(path, 5, STATIC + [shift[0], shift[1], 0.0])
    return Scene(path, split="train")


@pytest.mark.parametrize("device", DEVICES)
def test_single_scene(tmp_path: Path, device: str) -> None:
    vmap = VoxelMap(0.5, device=device)
    seen = []
    stats = vmap.add_scene(_scene(tmp_path / "a"), progress=lambda done, total: seen.append((done, total)))
    assert seen == [(i, 5) for i in range(1, 6)]
    assert stats["frames"] == 5 and stats["rows"] == 20 and stats["measured"] == 15 and stats["mapped"] == 15
    cloud = vmap.to_cloud()
    order = np.argsort(cloud.xyz[:, 0])
    np.testing.assert_allclose(cloud.xyz[order], STATIC, atol=0.005)  # 5 mm storage grid
    assert cloud.frame == "world" and cloud.xyz.dtype == np.float64
    assert cloud.attrs["count"].tolist() == [5, 5, 5] and cloud.attrs["frames"].tolist() == [5, 5, 5]
    np.testing.assert_allclose(np.sort(cloud.attrs["reflectivity"]), [10, 20, 30])
    assert cloud.attrs["frame_index"].tolist() == [0, 0, 0] and cloud.attrs["scene_index"].tolist() == [0, 0, 0]
    assert len(vmap.to_cloud(min_frames=6)) == 0


@pytest.mark.parametrize("device", DEVICES)
def test_scenes_with_different_origins_line_up(tmp_path: Path, device: str) -> None:
    a = _scene(tmp_path / "a")
    origin_b = (49.0150, 8.4180)  # ~90 m away
    # The same world points, in b's map frame.
    delta = MapOrigin(*origin_b).utm_offset - MapOrigin(49.0144, 8.4172).utm_offset
    b = _scene(tmp_path / "b", origin_b, shift=-delta)

    together = VoxelMap(0.5, device=device)
    together.add_scene(a)
    together.add_scene(b)
    cloud = together.to_cloud()
    assert together.scenes == ["a", "b"] and cloud.attrs["frames"].tolist() == [10, 10, 10]
    np.testing.assert_allclose(cloud.xyz[np.argsort(cloud.xyz[:, 0])], STATIC, atol=0.005)

    # Built apart, saved, loaded and merged: the same map.
    only_b = VoxelMap(0.5, device=device)
    only_b.add_scene(b)
    assert only_b.origin == MapOrigin(*origin_b)
    merged = VoxelMap(0.5, device=device)
    merged.add_scene(a)
    merged.merge(VoxelMap.load(only_b.save(tmp_path / "b.npz"), device=device))
    other = merged.to_cloud()
    np.testing.assert_allclose(other.xyz, cloud.xyz)
    for key in cloud.attrs:
        np.testing.assert_array_equal(other.attrs[key], cloud.attrs[key])


def test_frames_and_errors(tmp_path: Path) -> None:
    a = _scene(tmp_path / "a")
    assert keyframes(a, min_distance=2.0) == [0, 2, 4]  # 1 m per frame
    assert keyframes(a, min_distance=50.0) == [0]
    vmap = VoxelMap(0.5, device="cpu")
    vmap.add_scene(a, keyframes(a, min_distance=2.0))
    assert vmap.to_cloud().attrs["frames"].tolist() == [3, 3, 3]

    lone = make_scene(tmp_path / "lone", n=5, cameras=(), radars=())
    write_static_lidar(lone, 5)
    alone = VoxelMap(0.33, device="cpu")  # its grid starts at the world origin; 0.33 m keeps the points off cube faces
    alone.add_scene(Scene(lone, split="train"))  # no maps/origin.json: the scene's own world frame
    np.testing.assert_allclose(np.sort(alone.to_cloud().xyz[:, 0]), STATIC[:, 0], atol=0.005)
    with pytest.raises(ValueError, match="cannot join"):
        alone.add_scene(a)
    with pytest.raises(ValueError, match="cannot join"):
        _join(a, Scene(lone, split="train"))

    near, far = VoxelMap(0.3, device="cpu"), VoxelMap(0.3, device="cpu")
    near.add_scene(a)
    far.add_scene(_scene(tmp_path / "far", (49.03, 8.44)))  # ~2 km away: another anchor
    assert np.any(far.anchor != near.anchor) and (ANCHOR_STEP / 0.3) % 1
    with pytest.raises(ValueError, match="divide the anchor step"):
        near.merge(far)


def _join(first: Scene, second: Scene) -> None:
    vmap = VoxelMap(0.5, device="cpu")
    vmap.add_scene(first)
    vmap.add_scene(second)


@pytest.mark.parametrize("device", DEVICES)
def test_merging_in_slices(tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import kitscenes.mapping as mapping

    a = _scene(tmp_path / "a")
    whole = VoxelMap(0.02, device=device)
    whole.add_scene(a)
    expected = whole.to_cloud()
    monkeypatch.setattr(mapping, "BATCH_POINTS", 1)  # a run per frame
    monkeypatch.setattr(mapping, "_MERGE_MIN", 1)  # merge after every run
    monkeypatch.setattr(mapping, "_MERGE_SLICE", 2)  # in slices of two entries
    sliced = VoxelMap(0.02, device=device, max_device_voxels=1)  # every merged run goes to the host
    sliced.add_scene(a)
    got = sliced.to_cloud()
    np.testing.assert_allclose(got.xyz, expected.xyz, atol=1e-6)
    for key in expected.attrs:
        np.testing.assert_array_equal(got.attrs[key], expected.attrs[key])


def _longest_true_run(row: np.ndarray) -> int:
    best = current = 0
    for hit in row:
        current = current + 1 if hit else 0
        best = max(best, current)
    return best


@pytest.mark.parametrize("seed", range(5))
def test_run_statistics_match_a_reference(seed: int) -> None:
    from kitscenes.mapping import _frame_runs, _reduce

    rng = np.random.default_rng(seed)
    hits = rng.random((300, 40)) < rng.uniform(0.3, 0.9)  # voxel x frame
    cuts = np.sort(rng.choice(np.arange(1, 40), size=rng.integers(1, 12), replace=False))
    runs = []
    for start, end in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [40]])):
        voxels, frames = np.nonzero(hits[:, start:end])  # sorted by voxel, then frame
        if not len(voxels):
            continue
        keys, inverse = np.unique(voxels, return_inverse=True)
        n = len(keys)
        run = {
            "keys": torch.from_numpy(keys.astype(np.int64)), "count": torch.zeros(n, dtype=torch.int32),
            "offset_sum": torch.zeros((n, 3)), "reflectivity_sum": torch.zeros(n),
            "first": torch.zeros(n, dtype=torch.long),
        }
        run.update(_frame_runs(torch.from_numpy(inverse), torch.from_numpy(frames), n, int(start)))
        runs.append(run)
    merged = _reduce(runs)
    seen = hits.any(axis=1)
    assert merged["keys"].tolist() == np.flatnonzero(seen).tolist()
    assert merged["frames"].tolist() == hits[seen].sum(axis=1).tolist()
    assert merged["run_best"].tolist() == [_longest_true_run(r) for r in hits[seen]]
    assert merged["run_first"].tolist() == [int(np.flatnonzero(r)[0]) for r in hits[seen]]
    assert merged["run_last"].tolist() == [int(np.flatnonzero(r)[-1]) for r in hits[seen]]


def _scene_with_frames(path: Path, points: list[np.ndarray]) -> Scene:
    """A scene whose frame i sees the world points points[i] (the car drives 1 m per frame along x)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    make_scene(path, n=len(points), cameras=(), radars=(), maps=True)
    for i, world in enumerate(points):
        car = np.array([100.0 + 1.0 * i, 200.0, 150.0])
        grid = np.round((world - car - [0.0, 0.0, 1.2]) / 0.005).astype(np.int32)
        from release_fixture import frame_ns

        table = pa.Table.from_pydict({
            "x": grid[:, 0], "y": grid[:, 1], "z": grid[:, 2],
            "reflectivity": np.full(len(grid), 10, np.float32), "timestamp": np.full(len(grid), frame_ns(i) / 1e9),
        })
        meta = {"discretization_resolution": "0.005", "point_type_name": "PointKITScenes"}
        pq.write_table(table.cast(table.schema.with_metadata(meta)), path / "lidar_top" / f"{i:010d}.parquet")
    return Scene(path, split="train")


@pytest.mark.parametrize("device", DEVICES)
def test_moving_objects_and_the_ego_box(tmp_path: Path, device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import kitscenes.mapping as mapping

    wall = np.array([[130.33, 210.33, 151.33]])
    crossing = np.array([[115.33, 195.33, 150.33]])  # hit in frames 0, 2, 4 only: something passing by
    frames = []
    for i in range(6):
        car = np.array([100.0 + i, 200.0, 150.0])
        body = car + [1.0, 0.5, 0.0]  # a return off the car's own body
        frames.append(np.vstack([wall, body] + ([crossing] if i % 2 == 0 else [])))
    scene = _scene_with_frames(tmp_path / "s", frames)
    monkeypatch.setattr(mapping, "BATCH_POINTS", 1)  # a run per frame: the runs join across merges
    vmap = VoxelMap(0.5, device=device)
    vmap.add_scene(scene)
    everything = vmap.to_cloud()
    assert len(everything) == 2  # the body returns are gone
    by_x = everything.attrs["run"][np.argsort(everything.xyz[:, 0])].tolist()
    assert by_x == [1, 6] and sorted(everything.attrs["frames"].tolist()) == [3, 6]
    static = vmap.to_cloud(min_run=3)
    np.testing.assert_allclose(static.xyz, wall, atol=0.005)
    assert len(vmap.to_cloud(min_frames=3)) == 2  # a frame count keeps the passing object

    with_body = VoxelMap(0.5, device=device)
    with_body.add_scene(scene, ego_box=None)
    assert len(with_body.to_cloud()) == 8  # wall, crossing, and the body return in 6 places along the path
    assert len(with_body.to_cloud(min_run=2)) == 1  # the body smear is hit once per voxel
    with pytest.raises(ValueError, match="increasing"):
        VoxelMap(0.5, device=device).add_scene(scene, [2, 1])


def test_tiles(tmp_path: Path) -> None:
    scene = _scene(tmp_path / "a")
    vmap = VoxelMap(0.5, device="cpu")
    vmap.add_scene(scene)
    tiles = vmap.to_tiles(10.0)
    assert sum(len(t) for t in tiles.values()) == 3
    utm = MapOrigin(49.0144, 8.4172).utm_offset
    for (i, j), part in tiles.items():
        xy = part.xyz[:, :2] + utm
        assert (np.floor(xy / 10.0) == [i, j]).all()


@pytest.mark.parametrize("device, gpu_decode", [("cpu", False)] + [("cuda", False), ("cuda", True)] * ("cuda" in DEVICES))
def test_colour_and_classes_match_painting(tmp_path: Path, device: str, gpu_decode: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    import kitscenes.mapping as mapping
    from kitscenes.painting import label, paint
    path = make_scene(tmp_path / "s", n=5, cameras=("camera_ring_front", "camera_ring_rear"), radars=(), maps=True)
    write_static_lidar(path, 5)
    scene = Scene(path, split="train")
    camera = scene.rig.cameras["camera_ring_front"]
    folder = path / "processed" / "seamseg_instances" / "camera_ring_front"
    folder.mkdir(parents=True)
    for i in range(5):  # the nearest point is a Pole (48) in three frames, a Car (58) in two
        (u, v), = camera.project(scene.frame(i).fused_lidar().xyz[:1])[0].astype(int)
        mask = np.zeros((48, 64), bool)
        mask[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3] = True
        (folder / f"{i:010d}.json").write_text(instances_file([(mask, 0.9, 48 if i < 3 else 58)]))

    expected_rgb, expected_class = [], []
    for i in range(5):
        frame = scene.frame(i)
        xyz = frame.fused_lidar().xyz
        expected_rgb.append(paint(frame, xyz, reduce=1 if gpu_decode else 2)[0].astype(float))  # nvJPEG: full size
        expected_class.append(label(frame, xyz)[0])
    mean = np.floor(np.mean(expected_rgb, axis=0) + 0.5)  # points in the frame's order: nearest first

    monkeypatch.setattr(mapping, "BATCH_POINTS", 1)  # a run per frame: votes and sums merge
    vmap = VoxelMap(0.5, device=device, colour=True, classes=True, gpu_decode=gpu_decode)
    vmap.add_scene(scene)
    cloud = vmap.to_cloud()
    order = np.argsort(cloud.xyz[:, 0])
    np.testing.assert_array_equal(cloud.attrs["rgb"][order], mean)
    assert cloud.attrs["class"][order].tolist() == [48, 255, 255]
    assert [c[0] for c in expected_class] == [48, 48, 48, 58, 58]
    votes = cloud.attrs["class_votes"][order]
    assert votes[0] == 3 and (votes[1:] > 0).all()  # Pole 3 of 5; the others: unlabelled points the camera saw

    saved = VoxelMap.load(vmap.save(tmp_path / "m.npz"), device=device).to_cloud()
    np.testing.assert_array_equal(saved.attrs["class"], cloud.attrs["class"])
    with pytest.raises(ValueError, match="colour or classes"):
        VoxelMap(0.5, device=device).merge(vmap)


def test_label_batch_keeps_frames_apart() -> None:
    from types import SimpleNamespace

    from kitscenes.mapping import _label_batch

    def table(start, end, cls):
        return (np.array([start]), np.array([end]), np.array([0]), np.array([cls]))

    batch = [SimpleNamespace(class_images={"cam": table(0, 6, 48)}), SimpleNamespace(class_images={"cam": table(10, 11, 58)})]
    # Frame 0: a pixel far beyond its last mask interval, one inside it; frame 1: one on its mask, one unseen.
    queries = [(0, "cam", torch.tensor([2999, 5])), (1, "cam", torch.tensor([10, -1]))]
    classes, seen = _label_batch(queries, batch, [0, 2, 4], 4, 64 * 48, torch.device("cpu"))
    assert classes.tolist() == [255, 48, 58, 255] and seen.tolist() == [True, True, True, False]


def test_majority_counts_the_unlabelled_points() -> None:
    from kitscenes.mapping import _majority

    run = {
        "keys": torch.tensor([1, 2, 3, 4]), "class_seen": torch.tensor([5, 5, 0, 4], dtype=torch.int32),
        "vote_keys": torch.tensor([1, 1, 2, 4]), "vote_classes": torch.tensor([48, 58, 58, 13], dtype=torch.int16),
        "vote_counts": torch.tensor([3, 2, 2, 2], dtype=torch.int32),
    }
    classes, votes = _majority(run, 255)
    # Pole 3 of 5; Car 2 of 5 against 3 unlabelled; nothing seen; a tie goes to unlabelled.
    assert classes.tolist() == [48, 255, 255, 255] and votes.tolist() == [3, 3, 0, 2]


@pytest.mark.parametrize("device", DEVICES)
def test_projection_on_device_matches_the_camera(device: str) -> None:
    from kitscenes.mapping import _project_on_device

    camera = real_camera()
    rng = np.random.default_rng(0)
    x, y = np.meshgrid(np.arange(3.0, 60.0, 0.1), np.arange(-15.0, 15.0, 0.1))
    ground = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
    box = np.column_stack([np.full(5000, 10.0), rng.uniform(-1, 1, 5000), rng.uniform(0, 1.5, 5000)])
    points = np.vstack([ground, box]).astype(np.float32)
    uv, depth, seen = camera.project(points)
    to_camera = torch.tensor(camera.T_sensor_base[:3], dtype=torch.float32, device=device)
    # The torch twin of Camera.project: the same pixels and the same visible points.
    u, v, d, twin_seen = (t.cpu().numpy() for t in _project_on_device(torch.from_numpy(points).to(device), to_camera, camera))
    np.testing.assert_array_equal(twin_seen, seen)
    np.testing.assert_allclose(np.column_stack([u, v])[seen], uv[seen], atol=2e-3)
    np.testing.assert_allclose(d, depth, atol=1e-4)
