"""Training samples, collate and DALI indices on synthetic scenes in the release archive format."""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path

import numpy as np
import pytest

import kitscenes
import kitscenes.map
from kitscenes.samples import SampleDataset, SampleSpec, collate, make_sample
from map_fixture import map_root, write_osm
from release_fixture import make_scene, pack_scene, write_lidar, write_poses


@pytest.fixture
def root(tmp_path: Path) -> Path:
    for name, n in (("scene_a", 4), ("scene_b", 5)):
        scene = make_scene(
            tmp_path / "data" / "train" / name, n=n,
            cameras=("camera_ring_front", "camera_ring_rear"), lidars=("lidar_top", "lidar_front"),
            radars=("radar_front", "radar_left"),
        )
        write_lidar(scene, "lidar_front", range(n), n=7, zero_rows=(0,))
        pack_scene(scene)
    return tmp_path


SPEC = SampleSpec(frame="base", 
    cameras=("camera_ring_front", "camera_ring_rear"), reduce=2, lidars=("lidar_top", "lidar_front"),
    radars=("radar_front", "radar_left"), future_s=0.3,
)


def test_sample_layout(root: Path) -> None:
    ds = SampleDataset(kitscenes.open(root), SPEC)
    assert len(ds) == 9
    s = ds[1]
    assert s["meta"] == {"scene": "scene_a", "split": "train", "index": 1, "token": "scene_a_0001", "t_ns": 146_000_000,
                         "pose_frame": "local_utm", "frame": "base"}
    assert s["pose"]["T_world_base"].shape == (4, 4)
    cam = s["camera"]["camera_ring_front"]
    assert cam["image"].shape == (24, 32, 3) and cam["K"][0, 0] == pytest.approx(25.0)  # focal 50 at 1/2
    assert len(s["lidar"]["xyz"]) == 20 + 6 and s["lidar"]["source"].tolist() == [0] * 20 + [1] * 6
    assert len(s["radar"]["xyz"]) == 10 and set(s["radar"]["source"].tolist()) == {0, 1}
    np.testing.assert_allclose(s["future"]["xyz_ego"][:2, 0], [1.0, 2.0], atol=1e-6)  # 10 m/s
    assert s["future"]["valid"].tolist() == [True, True, False]  # 4 frames: only two follow frame 1


def test_camera_transform_maps_base_into_camera(root: Path) -> None:
    """T_cam_base follows the package's T_a_b convention (b -> a), like T_world_base."""
    ds = kitscenes.open(root)
    s = SampleDataset(ds, SPEC)[0]
    cam = s["camera"]["camera_ring_front"]
    ahead = np.array([11.5, 0.0, 1.6, 1.0])  # 10 m in front of the camera (at x = 1.5, z = 1.6)
    np.testing.assert_allclose(cam["T_cam_base"] @ ahead, [0.0, 0.0, 10.0, 1.0], atol=1e-9)
    points = np.array([[11.5, 0.0, 1.6], [12.0, 1.0, 1.0]])
    p = (cam["T_cam_base"][:3, :3] @ points.T).T + cam["T_cam_base"][:3, 3]
    uv = (p @ cam["K"].T)[:, :2] / p[:, 2:]
    camera = ds["scene_a"].rig.cameras["camera_ring_front"].scaled(1 / SPEC.reduce)
    np.testing.assert_allclose(uv, camera.project(points)[0])


def test_collate(root: Path) -> None:
    ds = SampleDataset(kitscenes.open(root), SPEC)
    batch = collate([ds[0], ds[1], ds[7]])
    assert batch["meta"]["scene"] == ["scene_a", "scene_a", "scene_b"]
    assert tuple(batch["camera"]["camera_ring_rear"]["image"].shape) == (3, 3, 24, 32)
    assert tuple(batch["pose"]["T_world_base"].shape) == (3, 4, 4)
    assert len(batch["lidar"]["xyz"]) == 3 * 26 and np.asarray(batch["lidar"]["batch"]).tolist() == [0] * 26 + [1] * 26 + [2] * 26
    assert tuple(batch["future"]["xyz_ego"].shape) == (3, 3, 3)


def test_bytes_and_deskew_inputs(root: Path) -> None:
    spec = SampleSpec(frame="base", cameras=("camera_ring_front",), image="bytes", lidars=("lidar_top",), lidar="deskew_inputs")
    s = SampleDataset(kitscenes.open(root), spec)[2]
    assert s["camera"]["camera_ring_front"]["image"][:2] == b"\xff\xd8"
    from kitscenes.motion import apply_deskew_inputs

    frame = kitscenes.open(root)["scene_a"].frame(2)
    np.testing.assert_allclose(apply_deskew_inputs(s["lidar"]["lidar_top"]), frame.lidar("lidar_top").xyz)
    batch = collate([s, s])
    assert isinstance(batch["camera"]["camera_ring_front"]["image"], list) and len(batch["lidar"]) == 2


def test_require_and_missing(root: Path) -> None:
    (root / "data/train/scene_a/camera_ring_rear.zip").unlink()  # not downloaded for scene_a
    ds = kitscenes.open(root)
    spec = SampleSpec(frame="base", cameras=("camera_ring_front", "camera_ring_rear"))
    with pytest.raises(kitscenes.SensorUnavailable, match="kitscenes download .* --components camera_ring_rear"):
        SampleDataset(ds, spec)  # scene_a would silently give no frames
    loose = SampleDataset(ds, SampleSpec(frame="base", cameras=spec.cameras, require=False))
    assert len(loose) == 9
    assert set(loose[0]["camera"]) == set(loose[5]["camera"]) == set(spec.cameras)  # every camera, with valid
    assert not loose[0]["camera"]["camera_ring_rear"]["valid"] and loose[5]["camera"]["camera_ring_rear"]["valid"]
    assert not loose[0]["camera"]["camera_ring_rear"]["image"].any()
    batch = collate([loose[0], loose[5]])  # stacked, with the mask and the calibration of both
    rear = batch["camera"]["camera_ring_rear"]
    assert tuple(rear["image"].shape) == (2, 3, 48, 64) and np.asarray(rear["valid"]).tolist() == [False, True]
    assert tuple(rear["K"].shape) == (2, 3, 3) and tuple(rear["T_cam_base"].shape) == (2, 4, 4)
    reduced = SampleDataset(ds, SampleSpec(frame="base", cameras=spec.cameras, require=False, reduce=4))
    assert collate([reduced[0], reduced[5]])["camera"]["camera_ring_rear"]["image"].shape == (2, 3, 12, 16)
    as_bytes = SampleDataset(ds, SampleSpec(frame="base", cameras=spec.cameras, require=False, image="bytes"))[0]
    assert as_bytes["camera"]["camera_ring_rear"]["image"][:2] == b"\xff\xd8"  # a black JPEG
    assert not as_bytes["camera"]["camera_ring_rear"]["valid"]
    with pytest.raises(ValueError):
        SampleSpec(frame="base", image="png")


def test_bytes_keep_full_size_intrinsics(root: Path) -> None:
    """JPEG bytes are the full-size image: their K is the full-size K, also with reduce, which
    kitscenes.decode.decode_cameras applies (tested in test_decode.py)."""
    s = SampleDataset(kitscenes.open(root), SampleSpec(frame="base", cameras=("camera_ring_front",), image="bytes", reduce=2))[0]
    from PIL import Image
    import io

    width, _ = Image.open(io.BytesIO(s["camera"]["camera_ring_front"]["image"])).size
    assert width == 64 and s["camera"]["camera_ring_front"]["K"][0, 2] == pytest.approx(32.0)


def test_pickles_small(root: Path) -> None:
    ds = SampleDataset(kitscenes.open(root), SPEC)
    assert len(pickle.dumps(ds)) < 4000
    assert pickle.loads(pickle.dumps(ds))[3]["meta"]["index"] == 3


def test_dataloader_workers(root: Path) -> None:
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    ds = SampleDataset(kitscenes.open(root), SPEC)
    batches = list(DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate))
    assert sum(len(b["meta"]["index"]) for b in batches) == 9
    assert isinstance(batches[0]["lidar"]["xyz"], torch.Tensor)


def test_dali_indices(root: Path, tmp_path: Path) -> None:
    from kitscenes.dali import write_camera_indices

    ds = kitscenes.open(root)
    frames = ds.frames(require=("camera_ring_front",))
    out = write_camera_indices(frames, ("camera_ring_front",), tmp_path / "idx")
    zips, idx = out["camera_ring_front"]
    assert [Path(z).name for z in zips] == ["camera_ring_front.zip"] * 2 and len(idx) == 2
    lines = Path(idx[0]).read_text().splitlines()
    assert lines[0] == "v1.2 4"
    ext, offset, size, name = lines[2].split()
    assert ext == "jpg" and name == "scene_a/camera_ring_front/0000000001.jpg" and int(offset) % 4096 == 0
    with open(zips[0], "rb") as fh:
        fh.seek(int(offset))
        assert fh.read(int(size)) == frames[1].image_bytes("camera_ring_front")
    with pytest.raises(ValueError, match="contiguous"):
        write_camera_indices([frames[0], frames[5], frames[1]], ("camera_ring_front",), tmp_path / "idx2")


def test_dali_indices_need_the_zips(root: Path, tmp_path: Path) -> None:
    from kitscenes.dali import write_camera_indices
    from kitscenes.download.archives import extract_component

    frames = kitscenes.open(root).frames(require=("camera_ring_front", "camera_ring_rear"))
    extract_component(root / "data/train/scene_b/camera_ring_rear.zip")  # readable from the folder, not by DALI
    assert frames[-1].image_bytes("camera_ring_rear")
    with pytest.raises(ValueError, match=r"scene_b.*camera_ring_rear\.zip.*extracted"):
        write_camera_indices(frames, ("camera_ring_rear",), tmp_path / "idx")
    out = write_camera_indices(frames, ("camera_ring_front",), tmp_path / "idx")
    assert all(Path(z).is_file() for z in out["camera_ring_front"][0])
    (root / "data/train/scene_a/camera_ring_front.zip").unlink()
    with pytest.raises(ValueError, match=r"camera_ring_front\.zip.*kitscenes download"):
        write_camera_indices(frames, ("camera_ring_front",), tmp_path / "idx")


def test_dali_reads_zip(root: Path, tmp_path: Path) -> None:
    dali = pytest.importorskip("nvidia.dali")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from nvidia.dali import fn, pipeline_def

    from kitscenes.dali import write_camera_indices

    frames = kitscenes.open(root).frames()
    paths, idx = write_camera_indices(frames, ("camera_ring_rear",), tmp_path / "idx")["camera_ring_rear"]

    @pipeline_def(batch_size=3, num_threads=2, device_id=0)
    def pipe():
        jpg = fn.readers.webdataset(paths=paths, index_paths=idx, ext=["jpg"], random_shuffle=False)
        return fn.decoders.image(jpg, device="mixed")

    p = pipe()
    p.build()
    (images,) = p.run()
    for k in range(3):
        diff = np.abs(np.asarray(images.as_cpu()[k]).astype(int) - frames[k].image("camera_ring_rear").astype(int))
        assert diff.mean() < 2


def test_future_slots_by_time(tmp_path: Path) -> None:
    """Slots come from the interpolated poses, by time: one missing poses.txt line is bridged (as for
    frame.pose), a longer gap leaves its slots masked and the later ones keep their times."""
    scene = make_scene(tmp_path / "data" / "train" / "s", n=8)
    lines = (scene / "poses.txt").read_text().splitlines(keepends=True)
    (scene / "poses.txt").write_text("".join(lines[:3] + lines[4:]))  # no pose line for frame 3
    s = SampleDataset(kitscenes.open(tmp_path), SampleSpec(frame="base", future_s=0.6))[0]
    assert s["future"]["valid"].all()
    np.testing.assert_allclose(s["future"]["xyz_ego"][:, 0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], atol=1e-6)  # 10 m/s
    (scene / "poses.txt").write_text("".join(lines[:3] + lines[6:]))  # frames 3, 4, 5: a 0.4 s gap
    s = SampleDataset(kitscenes.open(tmp_path), SampleSpec(frame="base", future_s=0.6))[0]
    assert s["future"]["valid"].tolist() == [True, True, False, False, False, True]
    np.testing.assert_allclose(s["future"]["xyz_ego"][:, 0], [1.0, 2.0, 0.0, 0.0, 0.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(s["future"]["quat_ego"][2], [0.0, 0.0, 0.0, 1.0])


def test_past_slots(tmp_path: Path) -> None:
    """SampleSpec(frame="base", past_s=...): the oldest slot first, masked before the scene start."""
    make_scene(tmp_path / "data" / "train" / "s", n=8)
    spec = SampleSpec(frame="base", past_s=0.4, future_s=0.2, trajectory_step_s=0.2)
    ds = SampleDataset(kitscenes.open(tmp_path), spec)
    s = ds[4]
    assert s["past"]["valid"].tolist() == [True, True] and s["future"]["valid"].tolist() == [True]
    np.testing.assert_allclose(s["past"]["xyz_ego"][:, 0], [-4.0, -2.0], atol=1e-6)  # frames 0 and 2
    np.testing.assert_allclose(s["future"]["xyz_ego"][:, 0], [2.0], atol=1e-6)        # frame 6
    assert ds[3]["past"]["valid"].tolist() == [False, True]
    batch = collate([ds[3], ds[4]])
    assert tuple(np.asarray(batch["past"]["xyz_ego"]).shape) == (2, 2, 3)
    assert tuple(np.asarray(batch["past"]["quat_ego"]).shape) == (2, 2, 4)
    with pytest.raises(ValueError):
        SampleSpec(frame="base", past_s=1.0, trajectory_step_s=0)


def test_the_training_path_is_public(root: Path) -> None:
    """kitscenes.SampleDataset/SampleSpec/collate in a DataLoader."""
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    ds = kitscenes.open(root)
    spec = kitscenes.SampleSpec(frame="base", cameras="camera_ring_front", reduce=2, lidars="lidar_top", future_s=0.3)
    assert spec.cameras == ("camera_ring_front",) and spec.lidars == ("lidar_top",)  # a name, not its letters
    batch = next(iter(DataLoader(kitscenes.SampleDataset(ds, spec), batch_size=3, collate_fn=kitscenes.collate)))
    assert tuple(batch["camera"]["camera_ring_front"]["image"].shape) == (3, 3, 24, 32)
    assert isinstance(batch["lidar"]["xyz"], torch.Tensor) and batch["lidar"]["batch"].max() == 2


def test_clips(root: Path) -> None:
    """SampleSpec(frame="base", clip_length=, clip_stride=): runs of frames of one scene, collated per position."""
    spec = SampleSpec(frame="base", cameras=("camera_ring_front",), reduce=2, lidars=("lidar_top",), clip_length=2, clip_stride=2)
    ds = SampleDataset(kitscenes.open(root), spec)
    # scene_a has frames 0-3, scene_b 0-4: clips end at frame 2.. of each scene
    assert len(ds) == 2 + 3 and "clips=5" in repr(ds)
    clip = ds[0]
    assert [s["meta"]["index"] for s in clip] == [0, 2] and {s["meta"]["scene"] for s in clip} == {"scene_a"}
    np.testing.assert_allclose(clip[-1]["pose"]["T_last_base"], np.eye(4), atol=1e-9)
    np.testing.assert_allclose(clip[0]["pose"]["T_last_base"][:3, 3], [-2.0, 0.0, 0.0], atol=1e-6)  # 0.2 s at 10 m/s
    batch = collate([ds[0], ds[3]])
    assert isinstance(batch, list) and len(batch) == 2
    assert batch[1]["meta"]["index"] == [2, 3] and batch[1]["meta"]["scene"] == ["scene_a", "scene_b"]
    assert tuple(np.asarray(batch[0]["camera"]["camera_ring_front"]["image"]).shape) == (2, 3, 24, 32)
    assert tuple(np.asarray(batch[0]["pose"]["T_last_base"]).shape) == (2, 4, 4)
    assert len(ds[1:3]) == 2 and ds[1:3][0][0]["meta"]["index"] == 1
    assert pickle.loads(pickle.dumps(ds))[4][1]["meta"]["index"] == 4
    with pytest.raises(ValueError):
        SampleSpec(frame="base", clip_length=0)


def test_clips_skip_dropped_frames(tmp_path: Path) -> None:
    scene = make_scene(tmp_path / "data" / "train" / "s", n=6)
    (scene / "camera_ring_front" / "0000000003.jpg").unlink()
    ds = SampleDataset(kitscenes.open(tmp_path), SampleSpec(frame="base", cameras=("camera_ring_front",), clip_length=3))
    assert [[s["meta"]["index"] for s in clip] for clip in ds] == [[0, 1, 2]]  # every clip with frame 3 is out
    loose = SampleDataset(kitscenes.open(tmp_path), SampleSpec(frame="base", cameras=("camera_ring_front",), clip_length=3, require=False))
    assert len(loose) == 4 and not loose[1][2]["camera"]["camera_ring_front"]["valid"]


class _FakeConverter:
    """Stands in for MLConverter (lanelet2.ml_converter is an optional build)."""

    def __init__(self, scene) -> None:
        self.scene = scene

    def __call__(self, pose):
        from kitscenes.map import MapLabels, Region

        x = float(pose.xyz[0])
        line = np.array([[x, 0.0, 0.0], [x + 1, 0.0, 0.0], [x + 2, 0.0, 0.0]])
        # a dashed chain of ways 1 and 2 cut in two pieces, a centerline of lanelet 100
        return MapLabels(lines={"Dashed": [line, line[:2]], "Centerline": [line]},
                         line_ids={"Dashed": [(1, 2), (1, 2)], "Centerline": [(100,)]},
                         line_pieces={"Dashed": [0, 1], "Centerline": [0]},
                         traffic_elements={"StopLine": [line[:2]]}, traffic_element_ids={"StopLine": [8]},
                         te_to_centerline=[("StopLine", 0, 0)], te_to_te=[], pose=pose, region=Region(), dims=3,
                         lane_segments=[line, line[1:]], lane_segment_ids=[(100,), (104,)],
                         lane_segment_pieces=[0, 0], lane_to_lane=[(0, 1)])


@pytest.fixture
def fake_converter(monkeypatch):
    import kitscenes.map._lanelet2 as l2
    import kitscenes.map.labels as labels

    made = []

    def converter(self, scene):
        made.append(scene.id)
        return _FakeConverter(scene)

    monkeypatch.setattr(labels, "require_ml_converter", lambda: None)
    monkeypatch.setattr(l2, "available", lambda: (True, ""))
    monkeypatch.setattr(kitscenes.MapLabelSpec, "converter", converter)
    return made


def test_map_labels_in_samples(tmp_path: Path, fake_converter) -> None:
    """SampleSpec(frame="base", map_labels=...): flat polylines per type, collated with a batch index (converter mocked)."""
    root = map_root(tmp_path)
    ds = kitscenes.open(root, splits="train")
    samples = SampleDataset(ds, SampleSpec(frame="base", map_labels=kitscenes.MapLabelSpec(grouping="m3tr")))
    s = samples[0]
    assert s["map"]["available"] and s["map"]["state"] == "present"
    dashed = s["map"]["lines"]["Dashed"]
    assert dashed["xyz"].shape == (5, 3) and dashed["xyz"].dtype == np.float32 and dashed["length"].tolist() == [3, 2]
    assert dashed["ids"].tolist() == [1, 2, 1, 2] and dashed["num_ids"].tolist() == [2, 2]
    assert dashed["piece"].tolist() == [0, 1] and dashed["key"].dtype == np.int64
    assert len(set(dashed["key"].tolist())) == 2
    assert s["map"]["traffic_elements"]["StopLine"]["length"].tolist() == [2]
    assert s["map"]["traffic_elements"]["StopLine"]["ids"].tolist() == [8]
    # every class of the grouping, in every sample (empty where there is none)
    assert tuple(s["map"]["lines"]) == kitscenes.map.GROUPING_CLASSES["m3tr"]
    assert tuple(s["map"]["traffic_elements"]) == kitscenes.map.TE_TYPES
    assert s["map"]["lines"]["RoadBorder"]["xyz"].shape == (0, 3) and s["map"]["lines"]["RoadBorder"]["key"].shape == (0,)
    batch = collate([samples[0], samples[1], samples[2]])
    assert np.asarray(batch["map"]["available"]).tolist() == [True] * 3
    assert np.asarray(batch["map"]["lines"]["Dashed"]["batch"]).tolist() == [0, 0, 1, 1, 2, 2]
    assert tuple(np.asarray(batch["map"]["lines"]["Dashed"]["xyz"]).shape) == (15, 3)
    assert np.asarray(batch["map"]["lines"]["Dashed"]["key"]).tolist() == dashed["key"].tolist() * 3  # same elements
    assert np.asarray(batch["map"]["lines"]["Dashed"]["ids"]).shape == (12,)
    assert set(batch["map"]["lines"]) == set(kitscenes.map.GROUPING_CLASSES["m3tr"])
    assert np.asarray(batch["map"]["lines"]["Virtual"]["batch"]).shape == (0,)
    # topology: indices into the sample's lists; in a batch with the sample as the first column
    topo = s["map"]["topology"]
    assert topo["lane_to_lane"].tolist() == [[0, 1]] and topo["te_to_te"].shape == (0, 4)
    assert topo["te_to_centerline"].tolist() == [[kitscenes.map.TE_TYPES.index("StopLine"), 0, 0]]
    assert s["map"]["lane_segments"]["length"].tolist() == [3, 2] and s["map"]["map_changed"] is False
    assert np.asarray(batch["map"]["topology"]["lane_to_lane"]).tolist() == [[0, 0, 1], [1, 0, 1], [2, 0, 1]]
    assert tuple(np.asarray(batch["map"]["topology"]["te_to_te"]).shape) == (0, 5)
    assert np.asarray(batch["map"]["lane_segments"]["batch"]).tolist() == [0, 0, 1, 1, 2, 2]
    assert np.asarray(batch["map"]["map_changed"]).tolist() == [False] * 3
    assert fake_converter == ["with_map"]  # one converter for the scene, reused for its frames
    assert pickle.loads(pickle.dumps(samples))[1]["map"]["available"]


def test_map_labels_where_the_split_withholds_maps(tmp_path: Path, fake_converter) -> None:
    root = map_root(tmp_path)
    spec = SampleSpec(frame="base", map_labels=kitscenes.MapLabelSpec())
    with pytest.raises(kitscenes.MapUnavailable, match="no_map") as err:
        SampleDataset(kitscenes.open(root), spec)
    assert err.value.state is kitscenes.Availability.WITHHELD_BY_SPLIT
    loose = SampleDataset(kitscenes.open(root), dataclasses.replace(spec, require=False))
    states = {s["meta"]["scene"]: (s["map"]["available"], s["map"]["state"]) for s in loose}
    assert states == {"with_map": (True, "present"), "no_map": (False, "withheld_by_split")}
    batch = collate([loose[0], loose[3]])
    assert np.asarray(batch["map"]["available"]).tolist() == [True, False]
    assert np.asarray(batch["map"]["lines"]["Dashed"]["batch"]).tolist() == [0, 0]
    no_map = loose[3]["map"]  # the same keys without a map, all empty
    assert tuple(no_map["lines"]) == kitscenes.map.GROUPING_CLASSES["default"]
    assert all(len(v["length"]) == 0 for v in no_map["lines"].values())
    assert set(no_map) == set(loose[0]["map"]) and len(no_map["lane_segments"]["length"]) == 0
    assert {k: v.shape for k, v in no_map["topology"].items()} == {"lane_to_lane": (0, 2), "te_to_centerline": (0, 3),
                                                                   "te_to_te": (0, 4)}
    assert np.asarray(batch["map"]["topology"]["lane_to_lane"]).tolist() == [[0, 0, 1]]


def test_map_labels_need_the_converter(tmp_path: Path) -> None:
    from kitscenes.map import has_ml_converter

    if has_ml_converter():
        pytest.skip("the ML converter is installed")
    with pytest.raises(kitscenes.MissingDependencyError):
        SampleDataset(kitscenes.open(map_root(tmp_path), splits="train"), SampleSpec(frame="base", map_labels=kitscenes.MapLabelSpec()))


def test_map_labels_with_the_real_converter(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map import MLConverter, Region

    root = map_root(tmp_path)
    region = Region(ahead=15, behind=5, left=5, right=5)
    ds = kitscenes.open(root, splits="train")
    s = SampleDataset(ds, SampleSpec(frame="base", map_labels=kitscenes.MapLabelSpec(region=region)))[1]
    expected = MLConverter.for_scene(ds[0], region=region)(ds[0].frame(1).pose)
    assert s["map"]["available"] and set(s["map"]["lines"]) == set(expected.lines) != set()
    for name, lines in expected.lines.items():
        got = s["map"]["lines"][name]
        np.testing.assert_allclose(got["xyz"], np.concatenate(lines) if lines else np.zeros((0, 3)), atol=1e-5)
        assert got["length"].tolist() == [20] * len(lines)
        assert got["ids"].tolist() == [i for ids in expected.line_ids[name] for i in ids]
        assert got["piece"].tolist() == expected.line_pieces[name]


def test_map_label_spec_maptr_preset() -> None:
    from kitscenes.map import MAPTR_CLASSES, Region

    spec = kitscenes.MapLabelSpec.maptr()
    assert spec.line_class_names == ("divider", "ped_crossing", "boundary") and spec.te_class_names == ()
    assert dict(spec.classes) == MAPTR_CLASSES and spec.grouping == "maptr" and spec.crosswalks == "polygons"
    assert spec.region == Region() and spec.dims == 3 and spec.max_height_diff == 4.0 and spec.chaining == "frame"
    v2 = kitscenes.MapLabelSpec.maptr(centerline=True, dims=2, max_height_diff=None, chaining="scene")
    assert v2.line_class_names == ("divider", "ped_crossing", "boundary", "centerline")
    assert v2.dims == 2 and v2.max_height_diff is None and v2.chaining == "scene"
    assert hash(spec) == hash(kitscenes.MapLabelSpec.maptr())  # specs stay hashable and comparable
    assert kitscenes.MapLabelSpec(classes=["Dashed"]).classes == (("Dashed", ("Dashed",)),)
    assert pickle.loads(pickle.dumps(spec)) == spec
    for bad in (dict(crosswalks="closed"), dict(chaining="city"), dict(max_height_diff=0.0),
                dict(classes={"divider": ["Dashed"]}), dict(te_classes={"x": ["Nope"]})):
        with pytest.raises(ValueError):
            kitscenes.MapLabelSpec.maptr(**bad)


def test_frames_of_samples_and_map_labels() -> None:
    """Samples are in the base frame by default; the MapTR preset is for the road frame and says so."""
    assert SampleSpec().frame == "base" and kitscenes.MapLabelSpec().frame is None
    maptr = kitscenes.MapLabelSpec.maptr()
    assert maptr.frame == "road" and kitscenes.MapLabelSpec.maptr(frame=None).frame is None
    assert SampleSpec(frame="road", map_labels=maptr).map_labels is maptr
    for frame in ("base", "road"):  # a spec without a frame follows the sample
        assert SampleSpec(frame=frame, map_labels=kitscenes.MapLabelSpec()).frame == frame
    with pytest.raises(ValueError, match=r'SampleSpec\(frame="road"\)'):
        SampleSpec(map_labels=maptr)  # the base-frame default with the road-frame preset
    with pytest.raises(ValueError, match=r'SampleSpec\(frame="base"\)'):
        SampleSpec(frame="road", map_labels=kitscenes.MapLabelSpec(frame="base"))
    with pytest.raises(ValueError, match="frame"):
        kitscenes.MapLabelSpec(frame="world")


def test_map_labels_maptr_preset_with_the_real_converter(tmp_path: Path) -> None:
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from kitscenes.map import Region

    scene = make_scene(tmp_path / "data" / "train" / "with_map", n=3, maps=True)
    write_osm(scene / "maps" / "map.osm")  # roads at 100 m, a crosswalk at y 20..23, a drivable_space_border
    write_poses(scene, 3, speed=1.0, origin=(1.75, 2.0, 102.0))  # driving east, 2 m above the road
    make_scene(tmp_path / "data" / "test" / "no_map", n=2)
    # a box 25 m to the left of the drive reaches the crosswalk
    spec = SampleSpec(frame="road", map_labels=kitscenes.MapLabelSpec.maptr(region=Region(30, 30, 25, 15)), require=False)
    by_scene = {s["meta"]["scene"]: s for s in SampleDataset(kitscenes.open(tmp_path), spec)}
    s = by_scene["with_map"]["map"]
    assert tuple(s["lines"]) == ("divider", "ped_crossing", "boundary") and s["traffic_elements"] == {}
    crossing = s["lines"]["ped_crossing"]  # one closed outline of 20 points
    assert crossing["length"].tolist() == [20] and crossing["ids"].tolist() == [103]
    np.testing.assert_array_equal(crossing["xyz"][0], crossing["xyz"][19])
    assert s["lines"]["boundary"]["ids"].tolist() == [3]  # the drivable_space_border line
    assert s["lines"]["divider"]["length"].tolist() and set(s["lines"]["divider"]["length"].tolist()) == {20}
    batch = collate([by_scene["with_map"], by_scene["no_map"]])
    assert tuple(batch["map"]["lines"]) == ("divider", "ped_crossing", "boundary")
    assert np.asarray(batch["map"]["lines"]["ped_crossing"]["batch"]).tolist() == [0]


def test_scene_chaining_in_samples_needs_the_fixed_converter(tmp_path: Path, fake_converter,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    from kitscenes.map import _lanelet2

    monkeypatch.setattr(_lanelet2, "has_reusable_map_data", lambda version=None: False)
    monkeypatch.setattr(_lanelet2, "_installed_version", lambda: "1.2.3+kitscenes1")
    spec = SampleSpec(frame="road", map_labels=kitscenes.MapLabelSpec.maptr(chaining="scene"))
    with pytest.raises(kitscenes.MissingDependencyError, match=r"kitscenes2"):
        SampleDataset(kitscenes.open(map_root(tmp_path), splits="train"), spec)


def test_map_label_spec_labels_are_the_sample_labels(tmp_path: Path) -> None:
    """MapLabelSpec.labels(frame, frame=...) is sample["map"]: MLConverter's labels, lifted into the road frame."""
    pytest.importorskip("lanelet2.ml_converter", reason="needs a lanelet2 build with the ML converter")
    from collections import OrderedDict

    from kitscenes.map import MLConverter

    scene_dir = make_scene(tmp_path / "data" / "map_change_train" / "s", n=3, maps=True)
    write_osm(scene_dir / "maps" / "map.osm")
    write_poses(scene_dir, 3, speed=1.0, origin=(1.75, 2.0, 102.0))  # on lanelet 100, driving east
    scene = kitscenes.open(tmp_path)[0]
    spec = kitscenes.MapLabelSpec.maptr(centerline=True)
    samples = SampleDataset(kitscenes.open(tmp_path), SampleSpec(map_labels=spec, frame="road"))
    got = spec.labels(scene.frame(1), frame="road", converters=OrderedDict())
    for name in ("lines", "lane_segments", "topology"):
        assert str(got[name]) == str(samples[1]["map"][name])
    expected = MLConverter.for_scene(scene, grouping="maptr", classes=dict(spec.classes), te_classes={},
                                     crosswalks="polygons", max_height_diff=4.0)(scene.frame(1).pose)
    assert got["map_changed"] and expected.map_changed and len(expected.lane_segments) == 2  # (100, 104), (101,)
    assert got["lane_segments"]["ids"].tolist() == [i for ids in expected.lane_segment_ids for i in ids]
    lift = np.array([0, 0, scene.road_height_m], np.float32)
    np.testing.assert_allclose(got["lane_segments"]["xyz"], np.concatenate(expected.lane_segments) + lift, atol=1e-5)
    np.testing.assert_allclose(got["lines"]["centerline"]["xyz"],
                               np.concatenate(expected.lines["centerline"]) + lift, atol=1e-5)
    assert got["topology"]["lane_to_lane"].tolist() == [list(e) for e in expected.lane_to_lane]
    # the preset's frame is the default of labels(); frame="base" is not lifted, nor is a spec without a frame
    assert str(spec.labels(scene.frame(1))["lines"]) == str(got["lines"])
    base = spec.labels(scene.frame(1), frame="base")
    unset = dataclasses.replace(spec, frame=None)
    assert str(unset.labels(scene.frame(1))["lines"]) == str(base["lines"])
    in_base = SampleDataset(kitscenes.open(tmp_path), SampleSpec(map_labels=unset))[1]["map"]  # base, the default
    assert str(in_base["lines"]) == str(base["lines"])
    np.testing.assert_allclose(base["lines"]["centerline"]["xyz"], np.concatenate(expected.lines["centerline"]),
                               atol=1e-5)
    with pytest.raises(ValueError):
        spec.labels(scene.frame(1), frame="world")
    # one entry: the spec is the converter's arguments
    conv = MLConverter.for_scene(scene, spec)
    assert (conv.grouping, conv.crosswalks, conv.max_height_diff) == ("maptr", "polygons", 4.0)
    assert str(conv(scene.frame(1).pose).lines) == str(spec.converter(scene)(scene.frame(1).pose).lines)
    with pytest.raises(ValueError, match="not both"):
        MLConverter.for_scene(scene, spec, dims=2)


def test_map_label_spec_cache_is_no_part_of_its_identity() -> None:
    spec = kitscenes.MapLabelSpec.maptr()
    assert dataclasses.replace(spec, cache=2) == spec and hash(dataclasses.replace(spec, cache=2)) == hash(spec)
    assert dataclasses.replace(spec, dims=2) != spec
