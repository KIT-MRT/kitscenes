"""Checks against a real dataset copy; run with ``KITSCENES_ROOT=<root> pytest test/test_real_data.py``.

Read-only. Every scene under the root is checked, so point it at a small
copy (e.g. 2 scenes per split). The aerial checks use
``<root>/aerial`` and the tile folders in ``$KITSCENES_AERIAL_TILES``
(separated by ``os.pathsep``; each a dataset root, an aerial folder, or
several folders joined with ``+`` that form one tile set, e.g. one per state).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import kitscenes
import kitscenes.map
from kitscenes import Availability, LabelsUnavailable, MapUnavailable
from kitscenes.constants import CAMERA_NAMES, KITSCENES_ROOT_ENV_VAR
from kitscenes.splits import scene_info

pytestmark = pytest.mark.real_data


@pytest.fixture(scope="module")
def dataset(real_root: Path) -> kitscenes.Dataset:
    return kitscenes.open(real_root)


def test_scene(dataset: kitscenes.Dataset, scene_id: str) -> None:
    scene = dataset[scene_id]
    ts = scene.timestamps_ns
    assert len(ts) > 1 and np.all(np.diff(ts) > 0)
    assert ts[0] >= 0 and np.all(np.diff(ts) % 100_000_000 == 0), "frame times: exact, on a 100 ms grid from >= 0"
    assert len(scene.track) == len(ts)
    assert scene.availability("pose") is kitscenes.Availability.PRESENT
    info = scene.split_info
    if info is not None:
        assert kitscenes.Scene(scene.files.path).split_info == info  # also without the dataset
        first_at_origin = not np.any(scene.track.xyz[0])
        assert first_at_origin == (info.pose_frame == "first_pose")
        assert (scene.files.path / "maps").is_dir() == info.has_maps
        if not info.has_maps:
            with pytest.raises(MapUnavailable):
                scene.map

    frame = scene.frame(len(scene) // 2)
    assert set(scene.rig.cameras) == set(CAMERA_NAMES)
    lidar = frame.lidar("lidar_top")
    assert lidar.frame == "base" and lidar.motion_compensated and lidar.measured.all()
    assert 0.5 < frame.lidar("lidar_top", rows=True).measured.mean() < 1.0
    measured = lidar.xyz
    assert np.median(np.linalg.norm(measured[:, :2], axis=1)) < 60
    radar = frame.radar("radar_front")
    assert radar.frame == "base" and radar.doppler_compensated
    image = frame.image("camera_ring_front", reduce=8)
    width, height = scene.rig.cameras["camera_ring_front"].image_size
    assert image.shape == (height // 8, width // 8, 3)
    if info is None:
        pass  # a folder outside the released splits: no split rules to check
    elif not info.has_labels:
        with pytest.raises(LabelsUnavailable) as err:
            frame.ground_mask("lidar_top")
        assert err.value.state is Availability.WITHHELD_BY_SPLIT
    else:
        for name in scene.rig.lidars:
            cloud = frame.lidar(name, labels=True)
            ground_z = cloud.xyz[cloud["ground"], 2]
            np.testing.assert_array_equal(cloud["ground"], frame.ground_mask(name))
            assert len(ground_z) > 100
            assert -3.0 < np.median(ground_z) < -1.0, f"{name}: ground labels do not lie on the road"


def test_lidar_rows_and_layouts(dataset: kitscenes.Dataset, scene_id: str) -> None:
    """Row index, scan layout, range images and no-returns on both lidar generations."""
    scene = dataset[scene_id]
    frame = scene.frame(len(scene) // 2)
    generation = None if scene.info is None else scene.info.lidar
    for name in frame.sensors("lidar"):
        full = frame.lidar(name, frame="sensor", deskew=False, rows=True, no_returns=True)
        cloud = frame.lidar(name, frame="sensor", deskew=False, rows=True)
        nr = full.no_return
        np.testing.assert_array_equal(np.flatnonzero(~nr).astype(np.uint32), cloud.attrs["row"])
        np.testing.assert_array_equal(full.xyz[~nr], cloud.xyz)
        assert not full.xyz[nr].any() and (full.attrs["timestamp"][nr] > 0).all()  # real firing times
        if name.startswith(("lidar_front", "lidar_left", "lidar_rear", "lidar_right")):  # Seyond
            assert full.layout is None and full.measured.all() and not nr.any()
            continue
        layout = full.layout
        top = {"hesai": "Hesai OT128", "velodyne": "Velodyne VLS-128 AP"}.get(generation, layout.model)
        assert layout.model == (top if name == "lidar_top" else "Hesai PandarXT-32")
        assert layout.num_rows == len(full) and nr.any()
        ring, column, ret = full.scan_position()
        np.testing.assert_array_equal(ring, full.attrs["ring"])  # the stored ring column agrees with the layout
        np.testing.assert_array_equal(ret, (full.attrs["sensor_specific_data"] >> 8) & 1)
        assert (full.range_image("ring") == np.arange(layout.rings)[None, :, None]).all()
        if layout.model == "Hesai OT128":
            assert abs((~full.measured & ~nr).mean() - 0.25) < 0.03  # a quarter of the slots never fire
        world = frame.lidar(name, frame="world", rows=True, no_returns=True)
        assert not world.xyz[world.no_return].any()  # never moved off the origin
    if scene.split_info is not None and scene.split_info.has_labels:
        for name in ("lidar_top", "lidar_corner_left"):
            mask, full = frame.ground_mask(name, rows=True), frame.ground_mask(name, rows=True, no_returns=True)
            nr = frame.lidar(name, rows=True, no_returns=True).no_return
            assert len(full) == len(nr) and not full[nr].any()
            np.testing.assert_array_equal(full[~nr], mask)


def _readme_python_blocks(*sections: str) -> list[str]:
    """The fenced python blocks of the README sections with these headings."""
    import re
    import textwrap

    text = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    blocks = []
    for section in sections:
        body = re.search(rf"^## {section}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
        assert body is not None, f"README.md has no section {section!r}"
        blocks += [textwrap.dedent(b) for b in re.findall(r"^[ \t]*```python\n(.*?)^[ \t]*```", body.group(1), re.M | re.S)]
    return blocks


def test_readme_snippets_run(dataset: kitscenes.Dataset, real_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The python blocks of the README's "Getting started" and "Quickstart" run against the copy."""
    import importlib.util
    import re

    blocks = _readme_python_blocks("Getting started", "Quickstart")
    assert blocks, "README.md has no python block to run"
    monkeypatch.setenv(KITSCENES_ROOT_ENV_VAR, str(real_root))
    # A scene of the copy for what the README names: its example scene, the val split.
    with_map = [s for s in dataset if s.split_info is not None and s.split_info.has_maps and len(s) > 50]
    if not with_map:
        pytest.skip("no scene with a map and more than 50 frames in this copy")
    scene = next((s for s in with_map if s.split == "val"), with_map[0])
    skip = [] if importlib.util.find_spec("lanelet2") else ["scene.map"]
    skip += [] if importlib.util.find_spec("torch") else ["SampleDataset"]
    for code in blocks:
        for scene_id in set(re.findall(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", code)) - set(dataset.scene_ids):
            code = code.replace(scene_id, scene.id)
        code = code.replace('splits="val"', f'splits="{scene.split}"')
        code = "\n".join(line for line in code.split("\n") if not any(name in line for name in skip))
        exec(compile(code, "README.md", "exec"), {"__name__": "__readme__"})


def test_example_tour_runs(real_root: Path) -> None:
    """examples/example_usage.py runs against the copy (it needs a val scene), so it keeps up with the API."""
    import subprocess
    import sys

    if not (real_root / "data" / "val").is_dir():
        pytest.skip("no val split in this copy")
    example = Path(__file__).resolve().parents[1] / "examples" / "example_usage.py"
    env = {**os.environ, KITSCENES_ROOT_ENV_VAR: str(real_root)}
    result = subprocess.run([sys.executable, str(example)], capture_output=True, text=True, env=env, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]


@pytest.mark.parametrize("script, args, needs", [
    ("train_loop.py", ["--split", "val", "--steps", "2", "--batch-size", "2", "--workers", "0", "--device", "cpu"],
     ["torch"]),
    ("export_scene.py", ["--split", "val", "--frames", "0:20:10", "--out", "{tmp}"], []),
    ("render_reprojection.py", ["--split", "val", "--frames", "0:2", "--reduce", "8", "--workers", "1", "--out", "{tmp}"],
     ["cv2", "matplotlib", "lanelet2"]),
    ("download_subset.py", ["{tmp}", "--split", "val", "--count", "2"], []),
])
def test_examples_run(script: str, args: list, needs: list, tmp_path: Path, real_root: Path) -> None:
    """The scripts in examples/ run against the copy (writing only into tmp_path), so they keep up with the API."""
    import importlib.util
    import subprocess
    import sys

    if not (real_root / "data" / "val").is_dir():
        pytest.skip("no val split in this copy")
    missing = [m for m in needs if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs {missing}")
    example = Path(__file__).resolve().parents[1] / "examples" / script
    env = {**os.environ, KITSCENES_ROOT_ENV_VAR: str(real_root)}
    command = [sys.executable, str(example), *(a.replace("{tmp}", str(tmp_path)) for a in args)]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=900)
    assert result.returncode == 0, result.stderr[-2000:]
    if script == "export_scene.py":
        (scene_dir,) = tmp_path.iterdir()
        assert (scene_dir / "poses_tum.txt").is_file() and len(list((scene_dir / "lidar").glob("*.pcd"))) == 2
    if script == "render_reprojection.py":
        assert len(list(tmp_path.glob("*.mp4"))) == 1


def test_gnss_topics(dataset: kitscenes.Dataset, scene_id: str) -> None:
    """GNSS/INS as one Parquet file per topic: 19 + 26 topics, the rates, frame rows on the frame times."""
    scene = dataset[scene_id]
    info = scene.split_info
    if info is None or not info.has_gnss:
        assert scene.message_topics("gnss") == [] and scene.availability("gnss_ins/imu") is not Availability.PRESENT
        return
    assert len(scene.message_topics("gnss")) == 19 and len(scene.message_topics("gnss_ins")) == 26
    ts = scene.timestamps_ns
    duration_s = (ts[-1] - ts[0]) / 1e9
    imu = scene.messages("gnss_ins", "imu")
    assert imu.column_names[:4] == ["index", "t_ns", "t_sync_ns", "frame_index"]
    assert abs(imu.num_rows / duration_s - 50) < 3, "IMU at 50 Hz"
    t_ns = imu.column("t_ns").to_numpy()
    assert np.all(np.diff(t_ns) > 0) and np.all(np.abs(imu.column("t_sync_ns").to_numpy() - t_ns) < 1_000_000)
    for device, topic, per_frame in (("gnss_ins", "imu", 1), ("gnss", "navsatfix", 1), ("gnss", "rfstatus", 10)):
        rows = scene.messages(device, topic, rate="frame")
        frames = rows.column("index").to_numpy()
        np.testing.assert_array_equal(rows.column("t_sync_ns").to_numpy(), ts[frames])  # the per-frame rule
        assert abs(len(frames) - len(ts) / per_frame) <= 2, (topic, len(frames), len(ts))
    assert scene.imu(frame="base")["angular_velocity"].shape == (imu.num_rows, 3)


def test_world_frame_ego_state_and_availability(dataset: kitscenes.Dataset) -> None:
    """frame="world", scene.ego and the one availability method on real data."""
    scene = next(s for s in dataset if s.split_info is not None and s.split_info.has_labels)
    frame = scene[len(scene) // 2]
    base, world = frame.lidar("lidar_top"), frame.lidar("lidar_top", frame="world")
    expected = kitscenes.transform_points(frame.T_world_base, base.xyz.astype(np.float64))
    np.testing.assert_allclose(world.xyz, expected, atol=0.01)  # float32 at km scale
    state = frame.ego()
    assert 0.0 <= state.speed < 40.0 and abs(state.yaw_rate) < 1.5 and np.all(np.abs(state.acceleration) < 15.0)
    np.testing.assert_allclose(state.T_world_base, frame.T_world_base, atol=1e-9)
    assert frame.availability("instances/camera_ring_front") is Availability.PRESENT
    assert frame.availability("instances/camera_ring_front_left") is Availability.NOT_IN_RELEASE
    assert frame.availability("ground_labels/lidar_top") is Availability.PRESENT
    assert scene.availability("pose") is Availability.PRESENT


def test_maptr_preset_on_real_maps(dataset: kitscenes.Dataset) -> None:
    """MapLabelSpec.maptr() on every map scene: the fixed classes, closed crosswalks of 20 points, unique keys,
    nothing far off the road level; the converter's crosswalk lines are often cut open by the box."""
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    spec = kitscenes.MapLabelSpec.maptr(centerline=True)
    open_lines = closed_polygons = 0
    for scene in dataset:
        if scene.split_info is None or not scene.split_info.has_maps:
            continue
        conv = spec.converter(scene)
        lines = kitscenes.map.MLConverter.for_scene(scene, grouping="maptr", sampling=None)
        for index in range(0, len(scene), 20):
            pose = scene.frame(index).pose
            labels = conv(pose)
            assert tuple(labels.lines) == ("divider", "ped_crossing", "boundary", "centerline")
            for name, found in labels.lines.items():
                keys = labels.line_keys(name)
                assert len(set(keys)) == len(keys)
                assert all(m.shape == (20, 3) for m in found)
                assert all(np.abs(m[:, 2] + 2.0).max() < 12.0 for m in found)  # no other level far off the road
            for ring in labels.lines["ped_crossing"]:
                assert np.array_equal(ring[0], ring[-1])
                closed_polygons += 1
            open_lines += sum(not np.allclose(m[0], m[-1]) for m in lines(pose).lines["PedestrianCrossing"])
    assert closed_polygons > 0 and open_lines > 0


def test_road_frame_puts_the_road_at_zero(dataset: kitscenes.Dataset, scene_id: str) -> None:
    """The road height (a constant per lidar generation) puts the ground labels in the own lane at z ~ 0."""
    scene = dataset[scene_id]
    if scene.availability("ground_labels/lidar_top") is not Availability.PRESENT:
        pytest.skip("no ground labels (test splits)")
    heights = []
    for index in np.linspace(0, len(scene) - 1, 5).astype(int):
        frame = scene.frame(int(index))
        cloud = frame.lidar("lidar_top", frame="road", labels=True, missing="none")
        if cloud is None or "ground" not in cloud.attrs:
            continue
        g = cloud.xyz[cloud["ground"]]
        lane = g[(np.abs(g[:, 0]) > 3) & (np.abs(g[:, 0]) < 12) & (np.abs(g[:, 1]) < 2.5)]
        if len(lane) > 100:
            heights.append(np.median(lane[:, 2]))
    assert heights and abs(np.median(heights)) < 0.08, f"road at z = {np.median(heights):+.3f} m in the road frame"


def test_sensor_files(dataset: kitscenes.Dataset, scene_id: str) -> None:
    """frame.sensor_file locates the bytes the frame reads, in the zips (by offset and by member name) and in folders."""
    import zipfile

    scene = dataset[scene_id]
    frame = scene.frame(len(scene) // 2)
    for sensor in ("camera_ring_front", "lidar_top", "radar_front"):
        where = frame.sensor_file(sensor)
        data = where.read()
        assert data == scene.files.storage.read_bytes(where.member) and len(data) == where.size
        if where.in_zip:
            assert where.path.suffix == ".zip" and where.zip_member == f"{scene.files.storage.root.name}/{where.member}"
            if sensor != "radar_front":
                assert where.offset % 4096 == 0
            with zipfile.ZipFile(where.path) as zf:
                info = zf.getinfo(where.zip_member)
                assert info.compress_type == zipfile.ZIP_STORED and info.file_size == where.size
        else:
            assert where.path == scene.files.path / where.member and where.offset == 0
    assert frame.sensor_file("camera_ring_front").read() == frame.image_bytes("camera_ring_front")


def test_bytes_decoded_at_reduce_match_the_decoded_samples(dataset: kitscenes.Dataset) -> None:
    """SampleSpec(image="bytes", reduce=4) + decode_cameras gives the image and K of image="decoded", reduce=4."""
    torch = pytest.importorskip("torch")
    from kitscenes.decode import decode_cameras
    from kitscenes.samples import SampleDataset, SampleSpec, collate

    cams = ("camera_ring_front", "camera_base_front_center")
    as_bytes, decoded = (SampleSpec(cameras=cams, image=image, reduce=4) for image in ("bytes", "decoded"))
    picks = SampleDataset(dataset, as_bytes).frames
    picks = picks[:: max(1, len(picks) // 3)][:3]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        import cv2
    except ImportError:
        cv2 = None
    got = decode_cameras(collate(list(SampleDataset(dataset, as_bytes, picks))), as_bytes, device=device)
    want = collate(list(SampleDataset(dataset, decoded, picks)))
    for cam in cams:
        image = got["camera"][cam]["image"].cpu()
        expected = want["camera"][cam]["image"]
        assert image.shape == expected.shape
        assert (image.float() - expected.float()).abs().mean() < 2.0  # DCT scaling vs. nvJPEG + resize
        if cv2 is not None:  # and no shift between them
            (dx, dy), _ = cv2.phaseCorrelate(image[0].float().mean(0).numpy().astype(np.float64),
                                             expected[0].float().mean(0).numpy().astype(np.float64))
            assert abs(dx) < 0.05 and abs(dy) < 0.05
        np.testing.assert_allclose(got["camera"][cam]["K"].numpy(), want["camera"][cam]["K"].numpy())


def _aerial_tile_sets(root: Path) -> list:
    sets = [root] if (root / "aerial").is_dir() else []
    sets += [s.split("+") for s in os.environ.get("KITSCENES_AERIAL_TILES", "").split(os.pathsep) if s]
    return sets


def test_aerial_crops_agree_across_tile_sets(dataset: kitscenes.Dataset, real_root: Path) -> None:
    """Crops on the fly from every tile set (release zips, extracted JPEG, PNG, a tile store) agree."""
    from kitscenes.aerial import AerialCrops

    sets = _aerial_tile_sets(real_root)
    if not sets:
        pytest.skip("no aerial tiles: neither <root>/aerial nor $KITSCENES_AERIAL_TILES")
    scenes = [s for s in dataset if s.split_info is not None and s.split_info.has_maps][:3]
    frames = [f for s in scenes for f in list(s)[:: max(1, len(s) // 4)]]
    reference = None
    for tiles in sets:
        crops = AerialCrops(tiles, (100.0, 50.0))
        report = crops.coverage(scenes)
        assert report.complete, report.summary()
        got = []
        for frame in frames:
            xy, yaw, area = crops._frame_pose(frame.scene, frame.index)
            crop = crops.crop(xy, yaw, area_id=area)
            assert crop.shape == (250, 500, 3) and crop.dtype == np.uint8
            got.append(crop.astype(np.int16))
        if reference is None:
            reference = got
        for a, b in zip(got, reference):
            assert np.abs(a - b).mean() < 3  # JPEG vs PNG: ~1.2


def test_maps_with_routing_problems_load(real_root: Path) -> None:
    """The maps of two scenes with routing-graph errors load all the same."""
    import warnings

    maps = [real_root / "data/val/7080aa7f-1f6d-0cd9-5efd-352426c5f51d",
            real_root / "data/map_change_val/48c63260-2e7a-8bef-615e-d18869d9c546"]
    maps = [m for m in maps if (m / "maps" / "map.osm").is_file()]
    if not maps:
        pytest.skip("the copy has neither scene")
    pytest.importorskip("lanelet2")
    for scene_dir in maps:
        smap = kitscenes.map.SceneMap.from_scene_dir(scene_dir)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert len(smap.lanes()) > 100 and smap.routing_problems


def _pose_stats(t: np.ndarray, xy: np.ndarray) -> tuple[float, float, float]:
    """Duration, distance and max speed over 1 s windows."""
    dist = float(np.sum(np.hypot(*np.diff(xy, axis=0).T)))
    vmax = 0.0
    for i in range(len(t)):
        j = i
        while j + 1 < len(t) and t[j] - t[i] < 1.0:
            j += 1
        if t[j] - t[i] >= 0.9:
            vmax = max(vmax, float(np.hypot(*(xy[j] - xy[i])) / (t[j] - t[i])))
    return float(t[-1] - t[0]), dist, vmax


def test_e2e_statistics_match_the_shipped_poses(real_root: Path) -> None:
    """The scene table's statistics of the test-e2e scenes are those of the shipped 4 s of poses."""
    e2e = ("test-e2e", "map_change_test-e2e")
    scene_dirs = [d for split in e2e for d in sorted((real_root / "data" / split).glob("*"))
                  if (d / "poses.txt").is_file() and scene_info(d.name) is not None][:4]
    if not scene_dirs:
        pytest.skip("no test-e2e scene in this copy")
    for scene_dir in scene_dirs:
        info = scene_info(scene_dir.name)
        poses = np.loadtxt(scene_dir / "poses.txt", ndmin=2)
        duration, distance, vmax = _pose_stats(poses[:, 0], poses[:, 1:3])
        assert info.duration_s == pytest.approx(round(duration, 1), abs=0.051)
        assert info.distance_m == pytest.approx(round(distance), abs=1)
        assert info.mean_speed_mps == pytest.approx(round(distance / duration, 1), abs=0.051)
        assert info.max_speed_mps == pytest.approx(round(vmax, 1), abs=0.051)
        assert info.duration_s <= 4.0 + 1e-9
        scene = kitscenes.Scene(scene_dir)
        assert scene.info == info and scene.split in e2e

